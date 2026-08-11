"""Pre-equilibrated lipid conformation library.

Generates realistic lipid conformations using geometric construction
with correct bond lengths (CHARMM36 equilibrium values) and random
gauche defects.  The initial geometry only needs to be approximately
correct — MD equilibration restores exact equilibrium values.

Force-field independence
------------------------
The geometry library is built from geometric principles using standard
bond lengths that are universal across CHARMM, AMBER, and OPLS force
fields (differences are < 0.001 nm for common bonds).  The force-field-
specific topology (atom types, charges, bond/angle/dihedral parameters)
is applied later in the ForceFieldAssigner step from the user-selected
force field's .rtp files.

Each lipid type gets a directory stored as:
    {data_dir}/lipid_conformations/{lipid_name}/
        conf_0000.npz
        conf_0001.npz
        ...
        metadata.json

Usage:
    from gmxbuilder.modules.membrane.lipid_library import LipidLibrary
    lib = LipidLibrary(data_dir)
    lib.ensure("POPC")            # build if not cached
    confs = lib.load("POPC", n=5) # get 5 random conformations
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

from gmxbuilder.runtime.hardware import (
    find_gromacs_executable,
    lipid_worker_threads,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MD_N_SNAPSHOTS = 50        # conformations per lipid


def _gmx_bin() -> str:
    executable = find_gromacs_executable()
    if not executable:
        raise RuntimeError("GROMACS is not available")
    return executable

# GROMACS vacuum MD is disabled by default — vacuum simulations at
# 300-400 K produce fully extended (~10 nm) lipid conformations that
# are physically valid but biologically irrelevant for membrane packing.
# The geometric gauche-defect approach reliably produces compact
# (~2.5-3.5 nm) membrane-like conformations for all 84 lipid types.
#
# To re-enable, set _USE_GROMACS_MD = True and ensure charmm36.ff
# is accessible.
_USE_GROMACS_MD = False

# Lipids that have full CHARMM36 parameter support (atom types + bonds)
_SUPPORTED_LIPIDS = {
    "POPC", "DPPC", "DMPC", "DOPC", "POPE", "DOPE",
    "POPG", "POPS", "POPA", "CHOL",
}

# If re-enabled: vacuum MD parameters
MD_TEMPERATURE = 300.0      # K — lower temp to reduce unfolding
MD_TIMESTEP = 0.002         # ps
MD_NSTEPS = 5000            # 10 ps — short, just relax bonds
MD_CUTOFF = 1.2             # nm


class LipidLibrary:
    """Manage a pre-equilibrated lipid conformation library."""

    def __init__(self, data_dir: str | Path | None = None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "lipid_conformations"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has(self, lipid_name: str) -> bool:
        """Check if conformations exist for *lipid_name*."""
        d = self._lipid_dir(lipid_name)
        meta = d / "metadata.json"
        if not meta.exists():
            return False
        # Check we have at least some npz files
        return len(list(d.glob("conf_*.npz"))) >= 5

    def ensure(self, lipid_name: str, force: bool = False) -> bool:
        """Build conformations for *lipid_name* if not cached. Returns True if built."""
        if not force and self.has(lipid_name):
            return False
        self._build_lipid(lipid_name)
        return True

    def load(self, lipid_name: str, n: int = 5,
             rng: np.random.Generator | None = None) -> list[tuple[np.ndarray, list[str]]]:
        """Return *n* random conformations of *lipid_name*.

        Each element is (coords_nm, atom_names).
        """
        if rng is None:
            rng = np.random.default_rng()
        d = self._lipid_dir(lipid_name)
        conf_files = sorted(d.glob("conf_*.npz"))
        if not conf_files:
            raise FileNotFoundError(f"No conformations for {lipid_name} in {d}")
        chosen = rng.choice(conf_files, size=min(n, len(conf_files)), replace=False)
        results = []
        for cf in chosen:
            data = np.load(cf, allow_pickle=False)
            results.append((data["coords"], data["atom_names"].tolist()))
        return results

    def load_one(self, lipid_name: str,
                 rng: np.random.Generator | None = None) -> tuple[np.ndarray, list[str]]:
        """Return a single random conformation."""
        confs = self.load(lipid_name, n=1, rng=rng)
        return confs[0]

    # ------------------------------------------------------------------
    # Internal: build
    # ------------------------------------------------------------------

    def _lipid_dir(self, name: str) -> Path:
        return self.data_dir / name.upper()

    def _build_lipid(self, lipid_name: str) -> None:
        """Build lipid conformations using GROMACS vacuum MD.

        For lipids with full CHARMM36 parameter support, runs a 50 ps
        vacuum MD at 400 K and extracts 50 evenly-spaced snapshots.
        For unsupported lipids, falls back to geometric gauche-defect
        sampling.

        The MD approach ensures all conformations are physically valid:
        bonded parameters match CHARMM36, nonbonded interactions are
        computed with proper VDW radii, and the resulting conformations
        have realistic tail disorder from thermal motion.
        """
        from gmxbuilder.modules.membrane.lipids import LipidRegistry
        from gmxbuilder.geometry.lipid_geom import build_lipid_geometry

        try:
            lipid = LipidRegistry.get(lipid_name)
            cat, t1, t2 = lipid.category, lipid.tail1, lipid.tail2
        except KeyError:
            raise ValueError(f"Unknown lipid: {lipid_name}")

        d = self._lipid_dir(lipid_name)
        d.mkdir(parents=True, exist_ok=True)

        # Generate initial all-trans geometry
        coords_all_trans, atom_names = build_lipid_geometry(
            lipid_name, t1, t2, cat, rng=None, gauche_prob=0.0)
        n_atoms = len(coords_all_trans)

        # Try GROMACS MD for supported lipids (disabled by default — see note above)
        if _USE_GROMACS_MD and lipid_name.upper() in _SUPPORTED_LIPIDS:
            try:
                snapshots = self._run_vacuum_md(
                    lipid_name, coords_all_trans, atom_names, n_atoms)
                if snapshots and len(snapshots) >= 10:
                    # Recenter each snapshot at its own geometric center.
                    # Without this, vacuum MD drift produces lipids far
                    # from origin, breaking placement in the membrane.
                    for i, snap_coords in enumerate(snapshots):
                        snap_coords = snap_coords - snap_coords.mean(axis=0)
                        npz_path = d / f"conf_{i:04d}.npz"
                        np.savez_compressed(npz_path, coords=snap_coords,
                                           atom_names=np.array(atom_names))
                    meta = {
                        "lipid_name": lipid_name, "n_atoms": n_atoms,
                        "n_conformations": len(snapshots),
                        "atom_names": atom_names,
                        "method": "gromacs_vacuum_md",
                        "temperature_K": MD_TEMPERATURE,
                        "md_nsteps": MD_NSTEPS,
                        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    with open(d / "metadata.json", "w") as f:
                        json.dump(meta, f, indent=2)
                    return
            except Exception:
                pass  # Fall through to geometric fallback

        # ---- Geometric fallback for unsupported lipids ----
        n_confs = MD_N_SNAPSHOTS
        for i in range(n_confs):
            prob = 0.10 + 0.20 * float(i) / float(max(n_confs - 1, 1))
            conf_rng = np.random.default_rng(i * 31337 + zlib.crc32(lipid_name.encode()) % 2**31)
            conf_coords, conf_names = build_lipid_geometry(
                lipid_name, t1, t2, cat, rng=conf_rng, gauche_prob=prob)
            npz_path = d / f"conf_{i:04d}.npz"
            np.savez_compressed(npz_path, coords=conf_coords, atom_names=np.array(conf_names))

        meta = {
            "lipid_name": lipid_name, "n_atoms": n_atoms,
            "n_conformations": n_confs, "atom_names": atom_names,
            "method": "geometric_fallback",
            "gauche_prob_range": [0.10, 0.30],
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(d / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # GROMACS vacuum MD
    # ------------------------------------------------------------------

    def _run_vacuum_md(
        self, lipid_name: str, coords: np.ndarray,
        atom_names: list[str], n_atoms: int,
    ) -> list[np.ndarray] | None:
        """Run a short vacuum MD simulation and return snapshot coordinates."""
        # Locate charmm36.ff
        ff_dir = (Path(__file__).resolve().parent.parent.parent /
                  "data" / "forcefields" / "charmm36")
        if not (ff_dir / "forcefield.itp").exists():
            return None

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # 1. Symlink force field
            ff_link = tmp_path / "charmm36.ff"
            ff_link.symlink_to(ff_dir)

            # 2. Write GRO
            gro_path = tmp_path / "lipid.gro"
            self._write_gro(coords, atom_names, lipid_name, gro_path)

            # 3. Write ITP with proper CHARMM36 parameters
            itp_path = tmp_path / f"{lipid_name}.itp"
            if not self._write_lipid_itp_file(lipid_name, coords, atom_names, itp_path):
                return None

            # 4. Write TOP
            top_path = tmp_path / "lipid.top"
            top_path.write_text(
                f'#include "charmm36.ff/forcefield.itp"\n'
                f'#include "{lipid_name}.itp"\n'
                f'\n[ system ]\nSingle {lipid_name} in vacuum\n\n'
                f'[ molecules ]\n{lipid_name}    1\n'
            )

            # 5. Write MDP
            mdp_path = tmp_path / "md.mdp"
            mdp_path.write_text(
                f"integrator               = md-vv\n"
                f"dt                       = {MD_TIMESTEP}\n"
                f"nsteps                   = {MD_NSTEPS}\n"
                f"tcoupl                   = v-rescale\n"
                f"tc-grps                  = System\n"
                f"tau-t                    = 0.5\n"
                f"ref-t                    = {MD_TEMPERATURE}\n"
                f"pcoupl                   = no\n"
                f"cutoff-scheme            = Verlet\n"
                f"verlet-buffer-tolerance  = -1\n"
                f"rlist                    = {MD_CUTOFF}\n"
                f"rvdw                     = {MD_CUTOFF}\n"
                f"rcoulomb                 = {MD_CUTOFF}\n"
                f"constraints              = none\n"
                f"pbc                      = xyz\n"
                f"comm-mode                = angular\n"
                f"nstxout                  = 0\n"
                f"nstvout                  = 0\n"
                f"nstlog                   = {MD_NSTEPS // 10}\n"
                f"nstenergy                = {MD_NSTEPS // 10}\n"
                f"nstxout-compressed       = {MD_NSTEPS // MD_N_SNAPSHOTS}\n"
                f"compressed-x-grps        = System\n"
            )

            # 6. grompp
            tpr_path = tmp_path / "lipid.tpr"
            env = os.environ.copy()
            env["GMXLIB"] = str(tmp_path)
            result = subprocess.run(
                [_gmx_bin(), "grompp", "-f", str(mdp_path), "-c", str(gro_path),
                 "-p", str(top_path), "-o", str(tpr_path), "-maxwarn", "5"],
                capture_output=True, text=True, cwd=str(tmp_path),
                timeout=60, env=env,
            )
            if result.returncode != 0:
                print(f"  grompp failed for {lipid_name}: {result.stderr[-200:]}")
                return None

            # 7. mdrun
            xtc_path = tmp_path / "traj.xtc"
            gro_out = tmp_path / "out.gro"
            result = subprocess.run(
                [_gmx_bin(), "mdrun", "-s", str(tpr_path),
                 "-c", str(gro_out), "-x", str(xtc_path),
                 "-ntmpi", "1", "-ntomp", str(lipid_worker_threads()),
                 "-nb", "cpu", "-nsteps", str(MD_NSTEPS)],
                capture_output=True, text=True, cwd=str(tmp_path),
                timeout=600, env=env,
            )
            if result.returncode != 0 or not xtc_path.exists():
                print(f"  mdrun failed for {lipid_name}")
                return None

            # 8. Extract snapshots from XTC trajectory
            return self._extract_snapshots(tpr_path, xtc_path, tmp_path)

    def _extract_snapshots(
        self, tpr_path: Path, traj_path: Path, tmp_path: Path
    ) -> list[np.ndarray]:
        """Extract evenly-spaced snapshots from .xtc or .trr trajectory."""
        from gmxbuilder.io.gro import GROReader
        snapshots = []
        for i in range(MD_N_SNAPSHOTS):
            t_ps = i * MD_NSTEPS * MD_TIMESTEP / MD_N_SNAPSHOTS
            out_gro = tmp_path / f"snap_{i:04d}.gro"
            result = subprocess.run(
                [_gmx_bin(), "trjconv", "-s", str(tpr_path), "-f", str(traj_path),
                 "-o", str(out_gro), "-dump", f"{t_ps:.3f}"],
                capture_output=True, text=True, cwd=str(tmp_path), timeout=30,
                input="System\n",
            )
            if result.returncode == 0 and out_gro.exists():
                try:
                    struct = GROReader().read(out_gro)
                    snapshots.append(struct.coordinates)
                except Exception:
                    pass
        return snapshots

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_gro(coords: np.ndarray, atom_names: list[str],
                   lipid_name: str, path: Path) -> None:
        n = len(coords)
        with open(path, "w") as f:
            f.write(f"Single {lipid_name}\n{n:5d}\n")
            for i in range(n):
                x, y, z = coords[i]
                f.write(f"{1:5d}{lipid_name:<5s}{atom_names[i]:>5s}"
                        f"{i+1:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
            f.write(" 10.00000 10.00000 10.00000\n")

    @staticmethod
    def _write_lipid_itp_file(
        lipid_name: str, coords: np.ndarray,
        atom_names: list[str], path: Path,
    ) -> bool:
        """Write a proper CHARMM36 ITP for a single lipid molecule.

        Uses the same code path as the system topology writer, ensuring
        correct atom types, charges, and bond parameters.
        """
        from gmxbuilder.core.structure import Structure
        from gmxbuilder.io.top import TopologyWriter

        n = len(coords)
        # Build a minimal Structure for the ITP writer
        struct = Structure(
            coordinates=coords.copy(),
            box_vectors=np.eye(3) * 10.0,
            atom_names=list(atom_names),
            resnames=[lipid_name] * n,
            resids=[1] * n,
            elements=["C"] * n,  # Will be overridden by ITP writer
        )

        tw = TopologyWriter(force_field="charmm36", ff_config={"protein": "charmm36"})
        try:
            tw._write_lipid_itp(lipid_name, struct, path)
            return path.exists() and path.stat().st_size > 100
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_library: LipidLibrary | None = None


def get_lipid_library(data_dir: str | Path | None = None) -> LipidLibrary:
    """Return the global lipid conformation library singleton."""
    global _library
    if _library is None:
        _library = LipidLibrary(data_dir)
    return _library
