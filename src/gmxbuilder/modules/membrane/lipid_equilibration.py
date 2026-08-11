"""Offline explicit-solvent equilibration for the Step 5 lipid library."""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Iterator

import numpy as np

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROReader, GROWriter
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.membrane.equilibrated_library import (
    ACCEPTED_METHOD,
    MIN_CONFORMERS,
    SCHEMA_VERSION,
    EquilibratedLipidLibrary,
    lipid_parameter_family,
    topology_signature,
)
from gmxbuilder.modules.membrane.lipid_orientation import (
    MAX_TAIL_CORE_GAP_NM,
    MIN_INWARD_COSINE,
    MIN_INWARD_PROJECTION_NM,
    infer_lipid_orientation,
    orient_lipid_to_outward_normal,
    outward_orientation,
)


_gpu_device: ContextVar[int | None] = ContextVar(
    "gmxbuilder_lipid_gpu_device", default=None
)


@contextmanager
def lipid_gpu_device(device_id: int | None) -> Iterator[None]:
    """Bind GROMACS mdrun calls in this execution to one visible GPU."""
    token = _gpu_device.set(None if device_id is None else int(device_id))
    try:
        yield
    finally:
        _gpu_device.reset(token)


def find_gromacs() -> str:
    from gmxbuilder.runtime.hardware import find_gromacs_executable

    candidate = find_gromacs_executable()
    if candidate:
        return candidate
    raise RuntimeError("GROMACS was not found; set GMX_BIN")


def _outer_headgroup_anchor(
    coordinates: np.ndarray,
    atom_names: list[str],
    box_midplane_z: float,
    *,
    upper_leaflet: bool | None = None,
) -> tuple[int, bool]:
    """Return an outward polar anchor and whether the lipid is upper-leaflet."""
    elements = [next((char for char in name.upper() if char.isalpha()), "") for name in atom_names]
    candidates = [index for index, element in enumerate(elements) if element in {"O", "N", "P", "S"}]
    if not candidates:
        candidates = [index for index, element in enumerate(elements) if element != "H"]
    if not candidates:
        candidates = list(range(len(atom_names)))
    # Classify the leaflet by the chemical head region, not by the whole
    # molecule COM (which is tail-dominated and misclassifies inverted lipids).
    upper = (
        bool(upper_leaflet)
        if upper_leaflet is not None
        else float(coordinates[candidates, 2].mean()) >= box_midplane_z
    )
    stripped = [str(name).strip() for name in atom_names]
    if "P" in stripped:
        return stripped.index("P"), upper
    z_values = coordinates[candidates, 2]
    local_index = int(np.argmax(z_values) if upper else np.argmin(z_values))
    return candidates[local_index], upper


def _simulation_lipid_resname_map(
    lipid_names: set[str], force_field: str, lipid_ff: str,
) -> dict[str, str]:
    """Map five-character GROMACS output residue names to registry names."""
    mapping = {name[:5].upper(): name for name in lipid_names}
    if lipid_ff == "gaff2":
        from gmxbuilder.modules.forcefield.gaff_backend import prepare_gaff_lipid
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        for name in lipid_names:
            lipid = LipidRegistry.get(name)
            template = prepare_gaff_lipid(name, lipid.smiles, lipid.charge)
            mapping[template.name[:5].upper()] = name
    return mapping


class LipidEquilibrationBuilder:
    """Build a validated conformer entry from a solvated bilayer."""

    def __init__(self, library: EquilibratedLipidLibrary | None = None, gmx: str | None = None):
        from gmxbuilder.runtime.hardware import lipid_worker_threads

        self.library = library or EquilibratedLipidLibrary()
        self.gmx = gmx or find_gromacs()
        self.threads = lipid_worker_threads()

    @staticmethod
    def _mdp(stage: str, nsteps: int, temperature: float) -> str:
        common = (
            "cutoff-scheme = Verlet\n"
            "nstlist = 20\nrlist = 1.2\nrcoulomb = 1.2\nrvdw = 1.2\n"
            "coulombtype = PME\nvdwtype = Cut-off\npbc = xyz\n"
        )
        if stage == "em":
            return "integrator = steep\nemtol = 1000\nemstep = 0.01\n" f"nsteps = {nsteps}\n" + common
        pressure = ""
        if stage == "npt":
            pressure = (
                "pcoupl = C-rescale\npcoupltype = Semiisotropic\n"
                "tau-p = 5.0\nref-p = 1.0 1.0\ncompressibility = 4.5e-5 4.5e-5\n"
            )
        return (
            "integrator = md\ndt = 0.002\n" f"nsteps = {nsteps}\n"
            "constraints = h-bonds\nconstraint-algorithm = lincs\n"
            "tcoupl = V-rescale\ntc-grps = System\ntau-t = 1.0\n"
            f"ref-t = {temperature:.2f}\ngen-vel = {'yes' if stage == 'nvt' else 'no'}\n"
            f"gen-temp = {temperature:.2f}\ngen-seed = 20260713\n"
            + pressure + common
        )

    def _run(self, args: list[str], cwd: Path, *, input_text: str | None = None, timeout: int = 3600) -> None:
        result = subprocess.run(
            args, cwd=cwd, input=input_text, text=True, capture_output=True, timeout=timeout,
        )
        if result.returncode:
            output = (result.stdout + "\n" + result.stderr)[-6000:]
            raise RuntimeError(f"Command failed ({' '.join(args)}):\n{output}")

    @staticmethod
    def _assert_finite_minimization(log_path: Path) -> None:
        """Reject an EM output that GROMACS wrote after infinite forces."""
        text = log_path.read_text(errors="replace")
        potential_matches = re.findall(r"Potential Energy\s*=\s*([^\s]+)", text)
        force_matches = re.findall(r"Maximum force\s*=\s*([^\s]+)", text)
        if not potential_matches or not force_matches:
            raise RuntimeError(f"Energy minimization diagnostics are missing from {log_path.name}")
        try:
            potential = float(potential_matches[-1])
            maximum_force = float(force_matches[-1])
        except ValueError as exc:
            raise RuntimeError(f"Energy minimization diagnostics are invalid in {log_path.name}") from exc
        if not math.isfinite(potential) or not math.isfinite(maximum_force):
            raise RuntimeError(
                f"Energy minimization {log_path.name} left non-finite energy or force; "
                "the structure contains unresolved atomic overlaps"
            )

    @staticmethod
    def _solvent_padding(net_charge_per_lipid: int) -> float:
        """Reserve enough water for neutralising extreme pure-anion bilayers."""
        return 2.0 if abs(int(net_charge_per_lipid)) >= 4 else 1.2

    @staticmethod
    def _ion_minimization_mdp(*, test_mode: bool) -> str:
        """Use conservative steepest descent after close ion placement."""
        steps = 10000 if test_mode else 20000
        return LipidEquilibrationBuilder._mdp("em", steps, 310.0).replace(
            "emstep = 0.01", "emstep = 0.001"
        )

    def _mdrun(self, stage: str, work: Path, *, timeout: int) -> str:
        """Run one MD stage, preferring CUDA but retrying safely on CPU."""
        base = [
            self.gmx, "mdrun", "-deffnm", stage,
            "-ntmpi", "1", "-ntomp", str(self.threads),
        ]
        if os.environ.get("GMXBUILDER_LIPID_LIBRARY_GPU", "1") == "1":
            try:
                gpu_args = ["-nb", "gpu", "-pme", "gpu"]
                selected_gpu = _gpu_device.get()
                if selected_gpu is not None:
                    gpu_args.extend(["-gpu_id", str(selected_gpu)])
                self._run(base + gpu_args, work, timeout=timeout)
                return stage
            except RuntimeError:
                # Some CUDA/GROMACS combinations complete the requested steps
                # but fail during device-buffer teardown. A final GRO is
                # written only after all requested integration steps; let the
                # next grompp validate it instead of repeating a long run.
                gpu_output = work / f"{stage}.gro"
                if gpu_output.is_file() and gpu_output.stat().st_size > 100:
                    return stage
                # Otherwise re-run to a separate basename so partial GPU files
                # are never consumed.
                cpu_stage = stage + "_cpu"
                self._run(
                    [self.gmx, "mdrun", "-s", f"{stage}.tpr", "-deffnm", cpu_stage,
                     "-ntmpi", "1", "-ntomp", str(self.threads),
                     "-nb", "cpu", "-pme", "cpu"],
                    work,
                    timeout=timeout,
                )
                return cpu_stage
        self._run(base + ["-nb", "cpu", "-pme", "cpu"], work, timeout=timeout)
        return stage

    def _genion_with_retry(self, work: Path) -> float:
        """Add neutralising/salt ions with bounded solvent-packing fallback."""
        topology = work / "topol.top"
        original_topology = topology.read_text()
        last_error: RuntimeError | None = None
        for rmin in (0.40, 0.35, 0.30):
            topology.write_text(original_topology)
            (work / "ionized.gro").unlink(missing_ok=True)
            try:
                self._run(
                    [
                        self.gmx, "genion", "-s", "ions.tpr", "-o", "ionized.gro",
                        "-p", "topol.top", "-neutral", "-conc", "0.15",
                        "-pname", "NA", "-nname", "CL", "-rmin", f"{rmin:.2f}",
                    ],
                    work,
                    input_text="SOL\n",
                    timeout=600,
                )
                return rmin
            except RuntimeError as exc:
                last_error = exc
                if "No more replaceable solvent" not in str(exc):
                    topology.write_text(original_topology)
                    raise
        topology.write_text(original_topology)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _repack_bootstrap_bilayer(system: System, spacing: float = 1.2) -> None:
        """Rigidly place bootstrap lipids on staggered, clash-free lattices."""
        membrane_components = [
            component for component in system.components
            if component.kind == ComponentKind.MEMBRANE
        ]
        if len(membrane_components) != 1:
            raise RuntimeError("Offline lipid equilibration requires one membrane component")
        metadata = membrane_components[0].metadata
        sizes = [int(value) for value in metadata.get("lipid_sizes", [])]
        upper_count = int(metadata.get("n_lipids_upper", 0))
        lower_count = int(metadata.get("n_lipids_lower", 0))
        if len(sizes) != upper_count + lower_count or min(upper_count, lower_count) <= 0:
            raise RuntimeError("Membrane lipid partition metadata is inconsistent")
        offsets = np.cumsum([0] + sizes)
        maximum_xy_span = max(
            float(np.ptp(system.coordinates[offsets[index]:offsets[index + 1], :2], axis=0).max())
            for index in range(len(sizes))
        )
        spacing = max(float(spacing), maximum_xy_span + 0.15)
        side = int(np.ceil(np.sqrt(max(upper_count, lower_count))))
        axis = (np.arange(side, dtype=float) - (side - 1) / 2.0) * spacing
        grid = np.asarray([(x, y) for y in axis for x in axis], dtype=float)
        box_xy = side * spacing
        lower_grid = grid.copy()
        lower_grid[:, :2] += spacing / 2.0
        lower_grid = (lower_grid + box_xy / 2.0) % box_xy - box_xy / 2.0

        for molecule_index in range(len(sizes)):
            indices = slice(offsets[molecule_index], offsets[molecule_index + 1])
            current_xy = system.coordinates[indices, :2].mean(axis=0)
            if molecule_index < upper_count:
                target_xy = grid[molecule_index]
                # Keep the two bootstrap tail slabs apart during the vacuum
                # precompression.  At 0.15 nm, long anionic lipids can put an
                # upper-leaflet terminal carbon within 0.01 nm of a lower-
                # leaflet hydrogen before the first EM.  The explicit-solvent
                # NVT/NPT stages subsequently close this temporary core gap.
                z_shift = 0.55
            else:
                target_xy = lower_grid[molecule_index - upper_count]
                z_shift = -0.55
            system.coordinates[indices, :2] += target_xy - current_xy
            system.coordinates[indices, 2] += z_shift
        # Highly polyunsaturated chains can interdigitate far beyond the
        # nominal midplane.  Regridding changes their XY neighbours, so a
        # fixed shift alone can still create sub-0.02-nm cross-leaflet pairs.
        # Separate the complete atomic slabs before the first vacuum EM; NPT
        # later restores the physical DHH and core contact.
        upper_atoms = slice(0, int(offsets[upper_count]))
        lower_atoms = slice(int(offsets[upper_count]), int(offsets[-1]))
        upper_inner = float(system.coordinates[upper_atoms, 2].min())
        lower_inner = float(system.coordinates[lower_atoms, 2].max())
        minimum_slab_gap = 0.18
        additional_z = max(
            (lower_inner + minimum_slab_gap - upper_inner) / 2.0,
            0.0,
        )
        if additional_z > 0.0:
            system.coordinates[upper_atoms, 2] += additional_z
            system.coordinates[lower_atoms, 2] -= additional_z
        dimensions = system.structure.dimensions()
        system.structure.box_vectors = np.diag([
            box_xy, box_xy, float(dimensions[2]) + 2.0 * additional_z + 0.3,
        ])

    @staticmethod
    def _reimage_bilayer_z(system: System) -> None:
        """Move whole lipids to the nearest intended leaflet periodic image."""
        component = next(
            item for item in system.components if item.kind == ComponentKind.MEMBRANE
        )
        sizes = [int(value) for value in component.metadata.get("lipid_sizes", [])]
        upper_count = int(component.metadata.get("n_lipids_upper", 0))
        lower_count = int(component.metadata.get("n_lipids_lower", 0))
        if len(sizes) != upper_count + lower_count or not sizes:
            raise RuntimeError("Membrane lipid partition metadata is inconsistent")
        offsets = np.cumsum([0] + sizes)
        box_z = float(system.structure.dimensions()[2])
        if box_z <= 0.0:
            raise RuntimeError("Membrane box Z dimension is invalid")
        dhh = float(
            component.metadata.get(
                "bilayer_thickness",
                component.metadata.get("bilayer_thickness_nominal", 3.8),
            )
        )
        for molecule_index in range(len(sizes)):
            start = int(offsets[molecule_index])
            end = int(offsets[molecule_index + 1])
            upper = molecule_index < upper_count
            molecule_names = [
                str(value).strip() for value in system.structure.atom_names[start:end]
            ]
            anchor_index, _ = _outer_headgroup_anchor(
                system.coordinates[start:end],
                molecule_names,
                0.0,
                upper_leaflet=upper,
            )
            anchor_z = float(system.coordinates[start + anchor_index, 2])
            target_z = (dhh / 2.0) * (1.0 if upper else -1.0)
            image_shift = round((target_z - anchor_z) / box_z) * box_z
            system.coordinates[start:end, 2] += image_shift

    def _precompress_bilayer(
        self,
        system: System,
        work: Path,
        force_field: str,
        water_model: str,
        target_apl: float,
        *,
        test_mode: bool,
    ) -> None:
        """Shrink a safe lattice to target APL using rigid-centre EM cycles."""
        component = next(
            item for item in system.components if item.kind == ComponentKind.MEMBRANE
        )
        sizes = [int(value) for value in component.metadata["lipid_sizes"]]
        upper_count = int(component.metadata["n_lipids_upper"])
        target_xy = float(np.sqrt(upper_count * target_apl))
        current = work / "compress_00.gro"
        GROWriter.write(system.structure, current)
        topology = work / "compress.top"
        TopologyWriter(
            force_field,
            ff_config={
                "protein": force_field,
                "lipid_ff": system.metadata.get("lipid_ff", force_field),
                "water_model": water_model,
            },
        ).write_top(system.structure, topology, system_name="Lipid library precompression")
        (work / "compress.mdp").write_text(
            self._mdp("em", 1000 if test_mode else 3000, 310.0)
        )
        offsets = np.cumsum([0] + sizes)
        cycle = 0
        initial_xy = float(GROReader().read(current).dimensions()[0])
        # The safe rigid-centre step contracts by at most four percent.  Size
        # the guard from the actual bootstrap lattice instead of retaining
        # the old 24-cycle limit that belonged to an eight-percent step.
        required_cycles = max(
            0,
            int(math.ceil(math.log(target_xy * 1.01 / initial_xy) / math.log(0.96))),
        )
        max_cycles = min(max(required_cycles + 2, 24), 64)
        while float(GROReader().read(current).dimensions()[0]) > target_xy * 1.01:
            cycle += 1
            if cycle > max_cycles:
                raise RuntimeError(
                    "Precompression did not reach the target APL in "
                    f"{max_cycles} cycles"
                )
            tpr = work / f"compress_{cycle:02d}.tpr"
            deffnm = f"compress_em_{cycle:02d}"
            self._run(
                [self.gmx, "grompp", "-f", "compress.mdp", "-c", current.name,
                 "-p", topology.name, "-o", tpr.name, "-maxwarn", "1"],
                work,
            )
            self._run(
                [self.gmx, "mdrun", "-s", tpr.name, "-deffnm", deffnm,
                 "-ntmpi", "1", "-ntomp", str(self.threads), "-nb", "cpu"],
                work,
                timeout=3600,
            )
            self._assert_finite_minimization(work / f"{deffnm}.log")
            minimized = GROReader().read(work / f"{deffnm}.gro")
            old_xy = float(minimized.dimensions()[0])
            # Polyunsaturated/branched lipids can create a hard contact when
            # all molecule centres move by 8% in one step.  Four-percent
            # rigid-centre increments keep every intermediate EM finite while
            # still reaching the target well within the 24-cycle guard.
            scale = max(target_xy / old_xy, 0.96)
            molecule_centers = np.asarray([
                minimized.coordinates[offsets[index]:offsets[index + 1], :2].mean(axis=0)
                for index in range(len(sizes))
            ])
            box_center = molecule_centers.mean(axis=0)
            for molecule_index in range(len(sizes)):
                indices = slice(offsets[molecule_index], offsets[molecule_index + 1])
                center = molecule_centers[molecule_index]
                target_center = box_center + (center - box_center) * scale
                minimized.coordinates[indices, :2] += target_center - center
            dimensions = minimized.dimensions()
            minimized.box_vectors = np.diag([
                dimensions[0] * scale,
                dimensions[1] * scale,
                dimensions[2],
            ])
            current = work / f"compress_{cycle:02d}.gro"
            GROWriter.write(minimized, current)

        # The loop minimises before each shrink. The target-size structure
        # therefore needs one final full EM before water is introduced.
        (work / "compress_final.mdp").write_text(self._mdp("em", 5000, 310.0))
        self._run(
            [self.gmx, "grompp", "-f", "compress_final.mdp", "-c", current.name,
             "-p", topology.name, "-o", "compress_final.tpr", "-maxwarn", "1"],
            work,
        )
        self._run(
            [self.gmx, "mdrun", "-s", "compress_final.tpr", "-deffnm", "compress_final",
             "-ntmpi", "1", "-ntomp", str(self.threads), "-nb", "cpu"],
            work,
            timeout=7200,
        )
        self._assert_finite_minimization(work / "compress_final.log")
        compressed = GROReader().read(work / "compress_final.gro")
        system.structure.coordinates = compressed.coordinates
        system.structure.box_vectors = compressed.box_vectors
        self._reimage_bilayer_z(system)

    def build(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None = None,
        *,
        temperature: float = 310.0,
        npt_ps: float = 1000.0,
        test_mode: bool = False,
        force: bool = False,
    ) -> Path:
        from gmxbuilder.modules.membrane.builder import MembraneBuilder
        from gmxbuilder.modules.membrane.lipids import LipidRegistry
        from gmxbuilder.modules.solvation.solvate import SolvationBuilder

        name = lipid_name.strip().upper()
        lipid = LipidRegistry.get(name)
        # Some amphiphiles are not stable or biologically representative as a
        # neat 100% bilayer (sterols, lysolipids, DAG, bulky glycolipids and
        # highly charged phosphoinositides).  Pre-equilibrate them at 40 mol%
        # in a POPC host so extracted conformers experience a sealed bilayer
        # while still providing >=20 molecules per leaflet for validation.
        host_categories = {"ST", "LPC", "DG", "CER", "GM1", "PIP"}
        host_lipid = (
            LipidRegistry.get("POPC") if lipid.category in host_categories else None
        )
        membrane_lipid_names = {name}
        if host_lipid is not None:
            membrane_lipid_names.add(host_lipid.name)
            membrane_config = {
                "lipid_composition": {
                    "upper": [
                        {"name": host_lipid.name, "ratio": 60},
                        {"name": name, "ratio": 40},
                    ],
                    "lower": [
                        {"name": host_lipid.name, "ratio": 60},
                        {"name": name, "ratio": 40},
                    ],
                },
                "n_lipids_per_leaflet": 64,
                "seed": 20260713,
            }
            target_apl = 0.60 * host_lipid.area_per_lipid + 0.40 * lipid.area_per_lipid
            target_dhh = (
                0.60 * host_lipid.bilayer_thickness
                + 0.40 * lipid.bilayer_thickness
            )
        else:
            membrane_config = {
                "lipid_type": name,
                "n_lipids_per_leaflet": 64,
                "seed": 20260713,
            }
            target_apl = float(lipid.area_per_lipid)
            target_dhh = float(lipid.bilayer_thickness)
        if lipid_ff is None and force_field.startswith("amber"):
            from gmxbuilder.modules.forcefield.lipid_policy import amber_lipid_backend

            lipid_ff, reason = amber_lipid_backend(sorted(membrane_lipid_names))
            if lipid_ff is None:
                raise ValueError(reason)
        lipid_ff = lipid_ff or force_field
        if not force and self.library.has(name, force_field, lipid_ff):
            return self.library.inspect(name, force_field, lipid_ff).path

        if lipid_ff == "gaff2" and not force_field.startswith("amber"):
            raise ValueError("GAFF2 lipids require an Amber protein force-field family")
        if force_field == "oplsaa":
            raise ValueError("No general OPLS lipid parameterization backend is installed")
        simulation_resnames = _simulation_lipid_resname_map(
            membrane_lipid_names, force_field, lipid_ff,
        )

        output_dir = self.library.entry_dir(name, force_field, lipid_ff, writable=True)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        build_started = time.time()
        with tempfile.TemporaryDirectory(prefix=f"gmxbuilder-{name.lower()}-") as temp:
            work = Path(temp)
            initial = System(
                Structure(np.empty((0, 3)), np.eye(3) * 10.0),
                metadata={
                    "force_field": force_field,
                    "lipid_ff": lipid_ff,
                    "water_model": "tip3p",
                    "selected_lipid_names": sorted(membrane_lipid_names),
                    "seed": 20260713,
                },
            )
            membrane = MembraneBuilder(use_equilibrated_library=False).run(
                initial, membrane_config,
            )
            if not membrane.success:
                raise RuntimeError("Bootstrap membrane failed: " + "; ".join(membrane.log))
            self._repack_bootstrap_bilayer(membrane.system)
            self._precompress_bilayer(
                membrane.system,
                work,
                force_field,
                "tip3p",
                target_apl,
                test_mode=test_mode,
            )
            solvated = SolvationBuilder().run(
                membrane.system, {
                    "water_model": "tip3p",
                    "box_padding": self._solvent_padding(lipid.charge),
                    "seed": 20260713,
                },
            )
            if not solvated.success:
                raise RuntimeError("Solvation failed: " + "; ".join(solvated.log))

            GROWriter.write(solvated.system.structure, work / "solvated.gro")
            TopologyWriter(
                force_field,
                ff_config={
                    "protein": force_field,
                    "lipid_ff": lipid_ff,
                    "water_model": "tip3p",
                },
            ).write_top(solvated.system.structure, work / "topol.top", system_name=f"{name} library")

            # Even smoke mode must perform a real minimisation: the bootstrap
            # bilayer is densely packed and a 100-step EM can leave enormous
            # LJ contacts that make an MD smoke test meaningless.
            (work / "em.mdp").write_text(self._mdp("em", 5000, temperature))
            self._run([self.gmx, "grompp", "-f", "em.mdp", "-c", "solvated.gro", "-p", "topol.top", "-o", "ions.tpr", "-maxwarn", "1"], work)
            # The offline library also builds deliberately extreme pure
            # anionic bilayers.  GROMACS' 0.6-nm default can exhaust the thin
            # solvent slabs before all neutralising ions fit; 0.4 nm remains
            # outside normal first-shell contact while allowing those valid
            # high-charge test systems to be neutralised.
            genion_rmin = self._genion_with_retry(work)
            (work / "ion_em.mdp").write_text(
                self._ion_minimization_mdp(test_mode=test_mode)
            )
            self._run([self.gmx, "grompp", "-f", "ion_em.mdp", "-c", "ionized.gro", "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "1"], work)
            self._run([
                self.gmx, "mdrun", "-deffnm", "em",
                "-ntmpi", "1", "-ntomp", str(self.threads),
            ], work)
            self._assert_finite_minimization(work / "em.log")

            nvt_steps = 100 if test_mode else 50000
            npt_steps = 250 if test_mode else max(50000, int(npt_ps * 500.0))
            (work / "nvt.mdp").write_text(self._mdp("nvt", nvt_steps, temperature))
            (work / "npt.mdp").write_text(self._mdp("npt", npt_steps, temperature))
            self._run([self.gmx, "grompp", "-f", "nvt.mdp", "-c", "em.gro", "-p", "topol.top", "-o", "nvt.tpr", "-maxwarn", "1"], work)
            nvt_output = self._mdrun("nvt", work, timeout=3600)
            self._run([self.gmx, "grompp", "-f", "npt.mdp", "-c", f"{nvt_output}.gro", "-p", "topol.top", "-o", "npt.tpr", "-maxwarn", "1"], work)
            npt_output = self._mdrun(
                "npt", work, timeout=7200 if test_mode else 172800,
            )

            whole = work / "whole.gro"
            self._run(
                [self.gmx, "trjconv", "-s", "npt.tpr", "-f", f"{npt_output}.gro", "-o", str(whole), "-pbc", "mol", "-center"],
                work, input_text="System\nSystem\n", timeout=600,
            )
            structure = GROReader().read(whole)
            validation_groups: list[list[int]] = []
            current: list[int] = []
            previous = None
            for index, (resname, resid) in enumerate(zip(structure.resnames, structure.resids)):
                key = (resname, resid)
                canonical_resname = simulation_resnames.get(
                    str(resname).strip().upper(), str(resname).strip().upper(),
                )
                if canonical_resname not in membrane_lipid_names:
                    if current:
                        validation_groups.append(current)
                        current = []
                    previous = None
                    continue
                if previous is not None and key != previous:
                    validation_groups.append(current)
                    current = []
                current.append(index)
                previous = key
            if current:
                validation_groups.append(current)
            if len(validation_groups) % 2:
                raise RuntimeError(
                    "Equilibrated membrane has an odd number of lipid molecules"
                )
            leaflet_split = len(validation_groups) // 2
            validation_records = [
                (indices, index < leaflet_split)
                for index, indices in enumerate(validation_groups)
            ]
            target_records = [
                (indices, upper_leaflet)
                for indices, upper_leaflet in validation_records
                if simulation_resnames.get(
                    str(structure.resnames[indices[0]]).strip().upper(),
                    str(structure.resnames[indices[0]]).strip().upper(),
                ) == name
            ]
            if len(target_records) < MIN_CONFORMERS * 2:
                raise RuntimeError(
                    f"Only {len(target_records)} intact {name} molecules were extracted; "
                    f"need at least {MIN_CONFORMERS} in each leaflet"
                )

            box_z = float(structure.dimensions()[2])
            prepared: list[tuple[np.ndarray, list[str]]] = []
            anchor_positions = []
            leaflet_flags: list[bool] = []
            for indices, upper_leaflet in target_records:
                coords = structure.coordinates[indices].copy()
                atom_names = [structure.atom_names[index].strip() for index in indices]
                anchor_index, _ = _outer_headgroup_anchor(
                    coords, atom_names, box_z / 2.0,
                    upper_leaflet=upper_leaflet,
                )
                leaflet_flags.append(upper_leaflet)
                anchor = coords[anchor_index].copy()
                anchor_positions.append(anchor.copy())
                coords -= anchor
                if not upper_leaflet:
                    coords[:, 2] *= -1.0
                # Store every accepted conformation in the canonical upper-
                # leaflet frame.  This rigid rotation preserves the NPT
                # internal geometry while making the on-disk contract explicit.
                coords = orient_lipid_to_outward_normal(
                    coords, atom_names, upper=True,
                )
                coords -= coords[anchor_index]
                prepared.append((coords, atom_names))

            atom_names = prepared[0][1]
            if any(names != atom_names for _, names in prepared):
                raise RuntimeError("Extracted lipid atom order is inconsistent")
            # Deterministic, even sampling from both leaflets.
            selected = np.linspace(0, len(prepared) - 1, 50, dtype=int)
            staging = output_dir.with_name(output_dir.name + ".building")
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            for number, index in enumerate(selected):
                coords, names = prepared[int(index)]
                np.savez_compressed(
                    staging / f"conf_{number:04d}.npz",
                    coords=coords,
                    atom_names=np.asarray(names),
                )
            signature = topology_signature(atom_names, force_field, lipid_ff)
            box_dimensions = structure.dimensions()
            upper_count = int(sum(leaflet_flags))
            lower_count = len(leaflet_flags) - upper_count
            area_per_lipid = float(
                box_dimensions[0] * box_dimensions[1] / 64.0
            )
            expected_apl = target_apl
            apl_ratio = area_per_lipid / expected_apl
            head_z = np.asarray([position[2] for position in anchor_positions], dtype=float)
            flags = np.asarray(leaflet_flags, dtype=bool)
            upper_heads = head_z[flags]
            lower_heads = head_z[~flags]
            def _circular_mean(values: np.ndarray) -> float:
                angles = values * (2.0 * np.pi / float(box_dimensions[2]))
                angle = float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))
                return (angle % (2.0 * np.pi)) * float(box_dimensions[2]) / (2.0 * np.pi)

            if len(upper_heads) and len(lower_heads):
                upper_mean = _circular_mean(upper_heads)
                lower_mean = _circular_mean(lower_heads)
                period = float(box_dimensions[2])
                head_delta = (upper_mean - lower_mean + period / 2.0) % period - period / 2.0
                measured_dhh = abs(float(head_delta))
            else:
                measured_dhh = float("nan")
            dhh_ratio = measured_dhh / target_dhh
            # Validate the entire simulated bilayer, including the POPC host
            # used for sterols that cannot form a stable pure bilayer.
            validation_projections: list[float] = []
            validation_cosines: list[float] = []
            upper_tail_z: list[np.ndarray] = []
            lower_tail_z: list[np.ndarray] = []
            for indices, validation_upper in validation_records:
                validation_coords = structure.coordinates[indices]
                validation_names = [
                    structure.atom_names[index].strip() for index in indices
                ]
                validation_profile = infer_lipid_orientation(
                    validation_coords, validation_names,
                )
                validation_projection, validation_cosine = outward_orientation(
                    validation_profile, upper=validation_upper,
                )
                validation_projections.append(validation_projection)
                validation_cosines.append(validation_cosine)
                validation_anchor, _ = _outer_headgroup_anchor(
                    validation_coords,
                    validation_names,
                    box_z / 2.0,
                    upper_leaflet=validation_upper,
                )
                head_plane = measured_dhh / 2.0 * (1.0 if validation_upper else -1.0)
                tail_relative = (
                    validation_coords[validation_profile.tail_indices, 2]
                    - validation_coords[validation_anchor, 2]
                )
                (upper_tail_z if validation_upper else lower_tail_z).append(
                    head_plane + tail_relative
                )
            projections = np.asarray(validation_projections, dtype=float)
            cosines = np.asarray(validation_cosines, dtype=float)
            orientation_passed = bool(
                len(projections)
                and np.all(projections >= MIN_INWARD_PROJECTION_NM)
                and np.all(cosines >= MIN_INWARD_COSINE)
                and upper_count >= MIN_CONFORMERS
                and lower_count >= MIN_CONFORMERS
            )
            if upper_tail_z and lower_tail_z:
                upper_inner = float(np.percentile(np.concatenate(upper_tail_z), 1.0))
                lower_inner = float(np.percentile(np.concatenate(lower_tail_z), 99.0))
                tail_core_gap = upper_inner - lower_inner
            else:
                tail_core_gap = float("nan")
            core_passed = bool(
                np.isfinite(tail_core_gap)
                and tail_core_gap <= MAX_TAIL_CORE_GAP_NM
            )
            production_quality = (
                not test_mode
                and npt_steps * 0.002 >= 500.0
                and np.isfinite([area_per_lipid, measured_dhh]).all()
                and 0.75 <= apl_ratio <= 1.35
                and 0.70 <= dhh_ratio <= 1.30
                and orientation_passed
                and core_passed
            )
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "status": "ready",
                "method": ACCEPTED_METHOD,
                "lipid_name": name,
                "canonical_smiles": lipid.smiles,
                "force_field": force_field,
                "lipid_ff": lipid_ff,
                "parameter_family": lipid_parameter_family(force_field, lipid_ff),
                "topology_sha256": signature,
                "atom_names": atom_names,
                "n_conformations": len(selected),
                "temperature_K": temperature,
                "salt_molar": 0.15,
                "genion_rmin_nm": genion_rmin,
                "npt_ps": npt_steps * 0.002,
                "test_mode": bool(test_mode),
                "equilibration_host": (
                    {
                        "lipid_name": host_lipid.name,
                        "ratio_percent": 60,
                        "target_ratio_percent": 40,
                    }
                    if host_lipid is not None else None
                ),
                "quality": {
                    "passed": bool(production_quality),
                    "reason": (
                        "APL, DHH, lipid orientation and hydrophobic-core seal passed production NPT gates"
                        if production_quality else
                        "test mode or a production APL/DHH/orientation/core-seal gate failed"
                    ),
                    "area_per_lipid_nm2": area_per_lipid,
                    "expected_area_per_lipid_nm2": expected_apl,
                    "apl_ratio": apl_ratio,
                    "dhh_nm": measured_dhh,
                    "expected_dhh_nm": target_dhh,
                    "dhh_ratio": dhh_ratio,
                    "orientation": {
                        "passed": orientation_passed,
                        "n_lipids_checked": int(len(projections)),
                        "upper_lipids": upper_count,
                        "lower_lipids": lower_count,
                        "minimum_inward_projection_nm": (
                            float(projections.min()) if len(projections) else None
                        ),
                        "minimum_inward_cosine": (
                            float(cosines.min()) if len(cosines) else None
                        ),
                    },
                    "hydrophobic_core": {
                        "passed": core_passed,
                        "tail_core_gap_nm": tail_core_gap,
                        "maximum_tail_core_gap_nm": MAX_TAIL_CORE_GAP_NM,
                    },
                },
                "elapsed_s": round(time.time() - build_started, 2),
            }
            (staging / "metadata.json").write_text(json.dumps(metadata, indent=2))
            if output_dir.exists():
                shutil.rmtree(output_dir)
            staging.replace(output_dir)
            if not test_mode and not production_quality:
                raise RuntimeError(
                    f"Production quality gates failed for {name}: "
                    f"APL ratio {apl_ratio:.3f}, DHH ratio {dhh_ratio:.3f}, "
                    f"minimum inward projection "
                    f"{float(projections.min()) if len(projections) else float('nan'):.3f} nm, "
                    f"minimum inward cosine "
                    f"{float(cosines.min()) if len(cosines) else float('nan'):.3f}, "
                    f"tail-core gap {tail_core_gap:.3f} nm. Diagnostics were retained in "
                    f"{output_dir / 'metadata.json'}"
                )
        return output_dir
