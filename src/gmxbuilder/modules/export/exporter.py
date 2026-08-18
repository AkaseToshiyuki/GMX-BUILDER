"""Module 5: Export — write complete GROMACS simulation package."""

from __future__ import annotations

from pathlib import Path
import json
import re
import shutil
import zipfile

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.system import System
from gmxbuilder.pipeline.base import BaseModule, ModuleResult
from gmxbuilder.io.gro import GROWriter
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.io.mdp import MDPWriter
from gmxbuilder.modules import register_module


@register_module
class ExportModule(BaseModule):
    """Write the final GROMACS simulation package.

    Produces:
    - input.gro           — system coordinates
    - topol.top           — master topology
    - index.ndx           — system-specific index groups
    - root-level *.itp    — force-field and per-molecule topologies
    - mdp/*.mdp           — simulation parameter files
    - run_md.sh           — launcher for the generated stage set
    """

    name = "export"
    description = "Write complete GROMACS simulation package (.gro, .top, .ndx, .itp, .mdp)"

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config,
            {
                "output_dir",
                "system_name",
                "write_mdp",
                "mdp_params",
                "simparams",
                "execution_hardware",
                "seed",
            },
        )
        if "write_mdp" in config and not isinstance(config["write_mdp"], bool):
            raise ModuleConfigError("export.write_mdp must be true or false")
        for key in ("mdp_params", "simparams", "execution_hardware"):
            if key in config and not isinstance(config[key], dict):
                raise ModuleConfigError(f"export.{key} must be an object")
        system_name = config.get("system_name")
        if system_name is not None:
            if not isinstance(system_name, str) or not system_name.strip():
                raise ModuleConfigError("export.system_name must be a non-empty string")
            if not all(character.isalnum() or character in "_-" for character in system_name):
                raise ModuleConfigError(
                    "export.system_name may contain only letters, numbers, '_' and '-'"
                )
        return True

    def run(self, system: System, config: dict) -> ModuleResult:
        output_dir = Path(config.get("output_dir", "./output"))
        system_name = config.get("system_name", "system")
        write_mdp = config.get("write_mdp", True)
        log = []

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "input.pdb").unlink(missing_ok=True)
        (output_dir / "run_md.sh").unlink(missing_ok=True)
        if (output_dir / "mdp").is_dir():
            shutil.rmtree(output_dir / "mdp")

        # ---- 0. Write index (.ndx) file ----
        ndx_path = output_dir / "index.ndx"
        self._write_index(system, ndx_path)
        log.append("Wrote index.ndx")

        # ---- 1. Write .gro + .pdb (input.gro + input.pdb) ----
        gro_path = output_dir / "input.gro"
        GROWriter.write(system.structure, gro_path, title=f"GMXBUILDER: {system_name}")
        log.append("Wrote input.gro")
        # Write PDB for visual reference (skip if >99,999 atoms — PDB format limit)
        if system.num_atoms <= 99999:
            from gmxbuilder.io.pdb import PDBWriter

            pdb_path = output_dir / "input.pdb"
            PDBWriter.write(system.structure, pdb_path, title=f"GMXBUILDER: {system_name}")
            log.append("Wrote input.pdb")
        else:
            log.append(
                f"Skipped input.pdb ({system.num_atoms} atoms exceeds PDB 99999-atom serial limit; use input.gro instead)"
            )

        # ---- 2. Write topology and flat root-level parameter files ----
        top_path = output_dir / "topol.top"
        # Read the force field from system metadata (set by ForceFieldAssigner)
        ff_name = system.metadata.get("force_field", config.get("protein", "amber14sb"))
        ff_config = {
            "protein": ff_name,
            "lipid_ff": system.metadata.get("lipid_ff", ff_name),
            "water_model": system.metadata.get(
                "water_model", system.metadata.get("ff_water_model", "tip3p")
            ),
            "ligand_parameters": system.metadata.get("ligand_parameters", {}),
            "native_nucleic_topologies": system.metadata.get("native_nucleic_topologies", []),
        }
        tw = TopologyWriter(force_field=ff_name, ff_config=ff_config)
        tw.write_top(system.structure, top_path, system_name=system_name, topology=system.topology)
        log.append("Wrote topol.top")

        # ---- 3. Write MDP files ----
        written: list[Path] = []
        execution_hardware: dict[str, object] = {}
        if write_mdp:
            mdp_dir = output_dir / "mdp"
            mdp_writer = MDPWriter()

            # simparams comes from system.metadata (set by web server or CLI)
            sim = system.metadata.get("simparams", config.get("simparams", {}))
            requested = dict(config.get("mdp_params", {}))
            from gmxbuilder.core.enums import ComponentKind

            has_membrane = bool(system.component_by_kind(ComponentKind.MEMBRANE))
            has_protein = bool(system.component_by_kind(ComponentKind.PROTEIN))
            has_nucleic = bool(system.component_by_kind(ComponentKind.NUCLEIC_ACID))
            lipid_ff = str(system.metadata.get("lipid_ff", "")).lower()
            has_lipid_dihedral_restraints = has_membrane and lipid_ff not in {"gaff2", "lipid21"}
            # Workflow metadata, execution hardware, and per-stage MDP values
            # are separate contracts. The completed system supplies only the
            # scientific context that a browser must not be allowed to forge.
            mdp_context = {
                "force_field": ff_name,
                "force_field_family": (
                    "charmm"
                    if str(ff_name).lower().startswith("charmm")
                    else "opls"
                    if str(ff_name).lower().startswith("opls")
                    else "amber"
                ),
                "has_membrane": has_membrane,
                "n_tc_groups": 3 if has_membrane and has_protein else 2,
                # The existing POSRES macro controls all solute macromolecule
                # ITPs, including native DNA/RNA position restraints.
                "protein_position_restraints": has_protein or has_nucleic,
                "lipid_position_restraints": has_membrane,
                "lipid_dihedral_restraints": has_lipid_dihedral_restraints,
            }
            raw_sim = {**requested, **dict(sim or {})}
            normalized_sim = mdp_writer.normalize_simulation_config(raw_sim, mdp_context)
            minimization = normalized_sim["minimization"]
            eq_stages = normalized_sim["eq_stages"]
            prod_iters = normalized_sim["prod_iters"]
            requested_dih = eq_stages or []
            if eq_stages:
                enabled_indices = [
                    index for index, stage in enumerate(eq_stages) if stage.get("enabled", True)
                ]
                if enabled_indices and enabled_indices[-1] < len(eq_stages) - 1:
                    last_enabled = eq_stages[enabled_indices[-1]]
                    remaining_restraints = {
                        key: float(last_enabled.get(key, 0.0))
                        for key in ("bb", "sc", "lipid", "dih")
                        if float(last_enabled.get(key, 0.0)) > 0.0
                    }
                    if remaining_restraints:
                        log.append(
                            "WARNING: later equilibration stages were disabled while the "
                            "last enabled stage still has restraints "
                            f"{remaining_restraints}; production removes them immediately"
                        )
            if not has_lipid_dihedral_restraints and any(
                stage.get("enabled", True) and float(stage.get("dih", 0)) > 0
                for stage in requested_dih
            ):
                log.append(
                    "WARNING: lipid dihedral restraints were requested but are unavailable "
                    f"for the exact {lipid_ff or 'selected'} lipid topology; the generated "
                    "MDP omits DIHRES instead of silently declaring an unused macro"
                )

            written = mdp_writer.generate_all(
                mdp_dir,
                mdp_context,
                eq_stages=eq_stages,
                prod_iters=prod_iters,
                minimization=minimization,
            )
            log.append(f"Wrote {len(written)} .mdp files to mdp/")
            from gmxbuilder.runtime.hardware import normalize_simulation_hardware

            execution_hardware = normalize_simulation_hardware(
                config.get("execution_hardware")
            )

        # ---- 3.5 Write run script + README (one-click launcher) ----
        readme_path = output_dir / "README.txt"
        self._write_readme(
            readme_path,
            system_name,
            system.metadata.get("seed", 42),
            ff_name,
            ff_config["water_model"],
            written,
            execution_hardware,
        )
        log.append("Wrote README.txt")
        from gmxbuilder.runtime.citations import atomistic_citations

        citations_path = output_dir / "CITATIONS.json"
        citations_path.write_text(
            json.dumps(atomistic_citations(system.metadata), indent=2) + "\n",
            encoding="utf-8",
        )
        log.append("Wrote CITATIONS.json")
        if write_mdp:
            run_script_path = output_dir / "run_md.sh"
            self._write_run_script(
                run_script_path,
                system_name,
                system.metadata.get("seed", 42),
                written,
                execution_hardware,
            )
            log.append("Wrote run_md.sh")

        # ---- 4. Write ZIP archive ----
        zip_path = output_dir / f"{system_name}.zip"
        self._write_archive(
            output_dir,
            zip_path,
            written,
            include_run_script=write_mdp,
        )
        log.append(f"Wrote {zip_path.name}")

        return ModuleResult(
            success=True,
            system=system,
            log=log
            + [
                f"  input.gro — {system.num_atoms} atoms",
                "  topol.top — simulation-ready topology",
                "  index.ndx — index groups (System/SOLU/MEMB/SOLV)",
                f"  mdp/ — {len(written)} generated MD parameter files",
                "  run_md.sh — executable one-click GROMACS launcher",
                f"  {system_name}.zip — complete package",
            ],
        )

    @staticmethod
    def _topology_members(output_dir: Path) -> set[Path]:
        """Resolve only safe, reachable local topology includes."""
        root = output_dir.resolve()
        pending = [output_dir / "topol.top"]
        members: set[Path] = set()
        include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"')
        while pending:
            path = pending.pop()
            resolved = path.resolve()
            if resolved in members or root not in (resolved, *resolved.parents):
                continue
            if not resolved.is_file() or resolved.is_symlink():
                continue
            members.add(resolved)
            for line in resolved.read_text(errors="replace").splitlines():
                match = include_pattern.match(line)
                if match:
                    pending.append(output_dir / match.group(1))
        return members

    @classmethod
    def _write_archive(
        cls,
        output_dir: Path,
        zip_path: Path,
        written_mdp: list[Path],
        *,
        include_run_script: bool,
    ) -> None:
        """Archive the current run manifest, never unrelated stale files."""
        candidates = {
            output_dir / "input.gro",
            output_dir / "input.pdb",
            output_dir / "index.ndx",
            output_dir / "README.txt",
            output_dir / "CITATIONS.json",
            *cls._topology_members(output_dir),
            *written_mdp,
        }
        if include_run_script:
            candidates.add(output_dir / "run_md.sh")
        members = sorted(
            path.resolve() for path in candidates if path.is_file() and not path.is_symlink()
        )
        zip_path.unlink(missing_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in members:
                archive.write(path, path.relative_to(output_dir.resolve()))

    @staticmethod
    def _index_groups(system) -> dict[str, list[int]]:
        """Return non-overlapping, scientifically named 1-based index groups.

        Uses System components for reliable classification:
        MEMBRANE → MEMB, SOLVENT+IONS → SOLV, everything else → SOLU.
        Ligands and cofactors are solutes, not membrane atoms.
        """
        from gmxbuilder.core.enums import ComponentKind

        solu: list[int] = []
        memb: list[int] = []
        solv: list[int] = []

        for comp in system.components:
            indices = [int(i) + 1 for i in comp.atom_indices]  # 1-based
            if comp.kind == ComponentKind.MEMBRANE:
                memb.extend(indices)
            elif comp.kind in (ComponentKind.SOLVENT, ComponentKind.IONS):
                solv.extend(indices)
            else:
                solu.extend(indices)

        solu.sort()
        memb.sort()
        solv.sort()
        groups = {
            "System": list(range(1, system.num_atoms + 1)),
            "SOLU": solu,
        }
        if memb:
            groups["MEMB"] = memb
        groups["SOLV"] = solv
        if memb:
            groups["SOLU_MEMB"] = sorted(solu + memb)
        return groups

    @classmethod
    def _write_index(cls, system, path):
        """Generate the exact index groups referenced by generated MDP files."""
        groups = cls._index_groups(system)

        with open(path, "w") as fh:
            fh.write("; GMXBUILDER — index file\n")
            for name, indices in groups.items():
                fh.write(f"\n[ {name} ]\n")
                for j in range(0, len(indices), 15):
                    fh.write(" ".join(str(x) for x in indices[j : j + 15]) + "\n")

    @staticmethod
    def _write_readme(
        path: Path,
        system_name: str,
        seed: int,
        force_field: str,
        water_model: str,
        mdp_paths: list[Path],
        execution_hardware: dict[str, object] | None = None,
    ) -> None:
        """Generate a README with usage instructions and file descriptions."""
        mdp_listing = (
            "\n".join(
                f"    {mdp_path.name:<20s} — generated simulation stage" for mdp_path in mdp_paths
            )
            or "    (MDP generation was disabled)"
        )
        coordinate_listing = [
            "    input.gro             — starting coordinates",
        ]
        if (path.parent / "input.pdb").is_file():
            coordinate_listing.append(
                "    input.pdb             — optional visualization coordinates"
            )
        coordinate_listing.extend(
            [
                "    topol.top             — master topology",
                "    index.ndx             — index groups referenced by the MDP files",
            ]
        )
        parameter_suffixes = {
            ".arn",
            ".atp",
            ".hdb",
            ".itp",
            ".r2b",
            ".rtp",
            ".tdb",
        }
        parameter_files = sorted(
            candidate.name
            for candidate in path.parent.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in parameter_suffixes
        )
        parameter_listing = (
            "\n".join(f"    {name}" for name in parameter_files)
            or "    (no separate parameter files were generated)"
        )
        production_names = [
            mdp_path.stem for mdp_path in mdp_paths if mdp_path.stem.startswith("production")
        ]
        final_production = production_names[-1] if production_names else "production"
        hardware = execution_hardware or {}
        cpu_threads = int(hardware.get("cpu_threads", 1))
        mpi_ranks = int(hardware.get("mpi_ranks", 1))
        omp_threads = int(hardware.get("omp_threads", 1))
        mpi_mode = str(hardware.get("mode", "thread-mpi"))
        gpu_ids = ",".join(str(value) for value in hardware.get("gpu_ids", []))
        gpu_description = gpu_ids if hardware.get("use_gpu") else "disabled"
        readme = f"""==================================================================
  GMXBUILDER — GROMACS Simulation Package
==================================================================

  System:      {system_name}
  Force field: {force_field}
  Water model: {water_model.upper()}
  Seed:        {seed}
  Date:        {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}

==================================================================
  Quick Start
==================================================================

  1. Make the run script executable:
       chmod +x run_md.sh

  2. Run the full simulation pipeline:
       ./run_md.sh

     The generated hardware defaults can be overridden with the
     environment variables listed below.

  3. Monitor progress:
       tail -f {final_production}.log

==================================================================
  File Listing
==================================================================

  Coordinates & Topology
{chr(10).join(coordinate_listing)}

  Force-field and molecule parameters
  These files are stored in the package root and are included by topol.top.
  The exact set depends on the selected force field and system composition.
{parameter_listing}

  MD Parameters (mdp/)
{mdp_listing}

  Scripts
    run_md.sh             — One-click MD simulation launcher
    README.txt            — This file

==================================================================
  Running Individual Stages
==================================================================

  To run a single stage manually (e.g., minimisation):

    gmx grompp -f mdp/mini.mdp \\
               -c input.gro \\
               -r input.gro \\
               -p topol.top \\
               -o mini.tpr
    gmx mdrun -deffnm mini -c mini.gro

  To continue production from a checkpoint:

    gmx mdrun -s {final_production}.tpr -cpi {final_production}.cpt -append

==================================================================
  Environment Variables
==================================================================

  GMX_BIN      GROMACS executable             [default: {hardware.get("gmx_command", "gmx")}]
  MPI_MODE     thread-mpi | external-mpi      [default: {mpi_mode}]
  CPU_THREADS  total CPU threads              [default: {cpu_threads}]
  MPI_RANKS    MPI or thread-MPI ranks        [default: {mpi_ranks}]
  OMP_THREADS  OpenMP threads per rank        [default: {omp_threads}]
  GPU_COUNT    number of selected GPUs        [default: {hardware.get("gpu_count", 0)}]
  GPU_IDS      comma-separated logical IDs    [default: {gpu_description}]
  USE_GPU      1 to expose GPU_IDS to mdrun   [default: {1 if hardware.get("use_gpu") else 0}]
  MPI_LAUNCHER mpirun | mpiexec | srun        [default: {hardware.get("mpi_launcher", "mpirun")}]

  CPU_THREADS must equal MPI_RANKS × OMP_THREADS. For external MPI,
  use a GROMACS executable compiled with external MPI (usually gmx_mpi).

==================================================================
"""
        path.write_text(readme)

    @staticmethod
    def _write_run_script(
        path: Path,
        system_name: str,
        seed: int,
        _mdp_paths: list[Path],
        execution_hardware: dict[str, object] | None = None,
    ) -> None:
        """Generate a one-click bash script.

        The script runs the full workflow:
          1. Energy minimization
          2. Every generated equilibration stage
          3. Every generated production iteration

        Thread-MPI and external-MPI launch paths are intentionally separate:
        ``-ntmpi`` is valid only for the thread-MPI GROMACS executable.
        """
        from gmxbuilder.runtime.hardware import normalize_simulation_hardware

        hardware = normalize_simulation_hardware(execution_hardware)
        gpu_ids = ",".join(str(value) for value in hardware["gpu_ids"])
        gro = "input.gro"
        top = "topol.top"
        ndx = "index.ndx"

        script = f"""#!/usr/bin/env bash
# =============================================================================
#  GMXBUILDER — One-click MD simulation script
#  System:  {system_name}
#  Seed:    {seed}
#  Date:    {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}
# =============================================================================
#
#  Usage:
#    chmod +x run_md.sh
#    ./run_md.sh
#
#  Environment variables (optional):
#    CPU_THREADS=8            total CPU threads
#    MPI_RANKS=2              MPI ranks
#    OMP_THREADS=4            OpenMP threads per rank
#    MPI_MODE="thread-mpi"    thread-mpi | external-mpi
#    GPU_COUNT=1               number of selected GPUs
#    GPU_IDS="0"              comma-separated logical GPU IDs
#    USE_GPU=1                expose GPU_IDS to mdrun
# =============================================================================

set -euo pipefail

# ---- Configuration ----
DEFAULT_CPU_THREADS={hardware["cpu_threads"]}
DEFAULT_MPI_RANKS={hardware["mpi_ranks"]}
DEFAULT_GPU_COUNT={hardware["gpu_count"]}
GMX_BIN="${{GMX_BIN:-{hardware["gmx_command"]}}}"
MPI_MODE="${{MPI_MODE:-{hardware["mode"]}}}"
MPI_LAUNCHER="${{MPI_LAUNCHER:-{hardware["mpi_launcher"]}}}"
MPI_RANKS="${{MPI_RANKS:-${{NTMPI:-$DEFAULT_MPI_RANKS}}}}"
OMP_THREADS="${{OMP_THREADS:-${{NTOMP:-}}}}"
CPU_THREADS="${{CPU_THREADS:-}}"
GPU_IDS="${{GPU_IDS:-{gpu_ids}}}"
GPU_COUNT="${{GPU_COUNT:-$DEFAULT_GPU_COUNT}}"
USE_GPU="${{USE_GPU:-{1 if hardware["use_gpu"] else 0}}}"
PIN="${{PIN:-{hardware["pin"]}}}"

if ! [[ "$MPI_RANKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MPI_RANKS must be a positive integer" >&2
    exit 2
fi
if [ -z "$CPU_THREADS" ]; then
    if [ -n "$OMP_THREADS" ]; then
        CPU_THREADS=$((MPI_RANKS * OMP_THREADS))
    else
        CPU_THREADS=$DEFAULT_CPU_THREADS
    fi
fi
if ! [[ "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "CPU_THREADS must be a positive integer" >&2
    exit 2
fi
if [ -z "$OMP_THREADS" ]; then
    if (( CPU_THREADS % MPI_RANKS != 0 )); then
        echo "MPI_RANKS must divide CPU_THREADS exactly" >&2
        exit 2
    fi
    OMP_THREADS=$((CPU_THREADS / MPI_RANKS))
fi
if ! [[ "$OMP_THREADS" =~ ^[1-9][0-9]*$ ]] ||
   (( MPI_RANKS * OMP_THREADS != CPU_THREADS )); then
    echo "CPU_THREADS must equal MPI_RANKS times OMP_THREADS" >&2
    exit 2
fi
if [[ "$MPI_MODE" != "thread-mpi" && "$MPI_MODE" != "external-mpi" ]]; then
    echo "MPI_MODE must be thread-mpi or external-mpi" >&2
    exit 2
fi
if [[ "$PIN" != "auto" && "$PIN" != "on" && "$PIN" != "off" ]]; then
    echo "PIN must be auto, on, or off" >&2
    exit 2
fi

MDRUN=("$GMX_BIN" mdrun -ntomp "$OMP_THREADS" -pin "$PIN")
if [ "$MPI_MODE" = "thread-mpi" ]; then
    MDRUN+=(-ntmpi "$MPI_RANKS")
else
    case "$MPI_LAUNCHER" in
        mpirun|mpiexec)
            MDRUN=("$MPI_LAUNCHER" -np "$MPI_RANKS" "${{MDRUN[@]}}")
            ;;
        srun)
            MDRUN=(srun -n "$MPI_RANKS" "${{MDRUN[@]}}")
            ;;
        *)
            echo "MPI_LAUNCHER must be mpirun, mpiexec, or srun" >&2
            exit 2
            ;;
    esac
fi
if [ "$USE_GPU" = "1" ]; then
    if ! [[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "GPU_IDS must be a comma-separated list of logical IDs" >&2
        exit 2
    fi
    if ! [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
        echo "GPU_COUNT must be a positive integer when GPU execution is enabled" >&2
        exit 2
    fi
    IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
    if (( ${{#GPU_ID_ARRAY[@]}} != GPU_COUNT )); then
        echo "GPU_COUNT must equal the number of entries in GPU_IDS" >&2
        exit 2
    fi
    if (( MPI_RANKS < GPU_COUNT )); then
        echo "MPI_RANKS must be at least GPU_COUNT; one rank cannot drive multiple GPUs" >&2
        exit 2
    fi
    declare -A SEEN_GPU_IDS=()
    for GPU_ID in "${{GPU_ID_ARRAY[@]}}"; do
        if [[ -n "${{SEEN_GPU_IDS[$GPU_ID]+x}}" ]]; then
            echo "GPU_IDS must not contain duplicate IDs" >&2
            exit 2
        fi
        SEEN_GPU_IDS[$GPU_ID]=1
    done
    MDRUN+=(-gpu_id "$GPU_IDS")
elif [ "$USE_GPU" != "0" ]; then
    echo "USE_GPU must be 0 or 1" >&2
    exit 2
fi

TOP="${{1:-{top}}}"
GRO="${{2:-{gro}}}"
NDX="${{3:-{ndx}}}"

echo "============================================"
echo " GMXBUILDER — MD Simulation"
echo " System:      {system_name}"
echo " MPI mode:    $MPI_MODE"
echo " CPU threads: $CPU_THREADS"
echo " MPI ranks:   $MPI_RANKS"
echo " OMP threads: $OMP_THREADS"
if [ "$USE_GPU" = "1" ]; then
    echo " GPUs:        $GPU_COUNT ($GPU_IDS)"
else
    echo " GPUs:        disabled"
fi
echo "============================================"
echo ""

# ---- Stage 0: Energy Minimisation ----
echo "[Stage 0] Energy Minimisation (steepest descent)"
$GMX_BIN grompp -f mdp/mini.mdp -c $GRO -r $GRO -p $TOP -o mini.tpr -po mini_out.mdp
"${{MDRUN[@]}}" -deffnm mini -c mini.gro
echo "  → mini.gro"
echo ""

# ---- Generated equilibration stages ----
PREV_GRO="mini.gro"
PREV_CPT=""
shopt -s nullglob
EQ_MDPS=(mdp/equili_*.mdp)
IFS=$'\n' EQ_MDPS=($(printf '%s\n' "${{EQ_MDPS[@]}}" | sort -V)); unset IFS
for MDP in "${{EQ_MDPS[@]}}"; do
    STAGE=$(basename "$MDP" .mdp)
    echo "[Equilibration] $STAGE"
    CPT_ARGS=()
    if [ -n "$PREV_CPT" ]; then CPT_ARGS=(-t "$PREV_CPT"); fi
    $GMX_BIN grompp -f "$MDP" -c "$PREV_GRO" -r "$GRO" "${{CPT_ARGS[@]}}" -p "$TOP" -n "$NDX" -o "$STAGE.tpr" -po "${{STAGE}}_out.mdp"
    "${{MDRUN[@]}}" -deffnm "$STAGE"
    PREV_GRO="$STAGE.gro"
    PREV_CPT="$STAGE.cpt"
done

# ---- Generated production iterations ----
PROD_MDPS=(mdp/production*.mdp)
IFS=$'\n' PROD_MDPS=($(printf '%s\n' "${{PROD_MDPS[@]}}" | sort -V)); unset IFS
for MDP in "${{PROD_MDPS[@]}}"; do
    STAGE=$(basename "$MDP" .mdp)
    echo "[Production] $STAGE"
    CPT_ARGS=()
    if [ -n "$PREV_CPT" ]; then CPT_ARGS=(-t "$PREV_CPT"); fi
    $GMX_BIN grompp -f "$MDP" -c "$PREV_GRO" "${{CPT_ARGS[@]}}" -p "$TOP" -n "$NDX" -o "$STAGE.tpr" -po "${{STAGE}}_out.mdp"
    "${{MDRUN[@]}}" -deffnm "$STAGE"
    PREV_GRO="$STAGE.gro"
    PREV_CPT="$STAGE.cpt"
done

echo "============================================"
echo " Simulation complete!"
echo ""
echo " Output files:"
echo "   *.gro        — coordinates at each stage"
echo "   *.xtc        — compressed trajectory"
echo "   *.edr        — energy data"
echo "   *.log        — GROMACS log"
echo "   *.cpt        — checkpoint (for restart)"
echo ""
echo " To restart from a checkpoint:"
echo "   gmx mdrun -s ${{STAGE}}.tpr -cpi ${{STAGE}}.cpt -append"
echo "============================================"
"""
        path.write_text(script)
        # Make executable
        path.chmod(0o755)
