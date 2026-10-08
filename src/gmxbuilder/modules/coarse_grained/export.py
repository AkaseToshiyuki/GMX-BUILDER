"""Self-contained export for Martini 3 systems; no atomistic exporter reuse."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.io.gro import GROWriter
from gmxbuilder.io.mdp import derive_velocity_seed
from gmxbuilder.io.pdb import PDBWriter
from gmxbuilder.modules.coarse_grained.assets import load_manifest, materialize_assets
from gmxbuilder.modules.coarse_grained.common import run_checked, write_topology_texts
from gmxbuilder.modules.coarse_grained.protocol import (
    normalize_protocol,
    write_index,
    write_mdp_files,
    write_run_script,
)
from gmxbuilder.modules.coarse_grained.workflow import CGWorkflowAdmission
from gmxbuilder.modules.export.layout import STRUCTURE_DIR, TOPOLOGY_DIR
from gmxbuilder.modules.export.naming import (
    confined_archive_path,
    record_authoritative_archive,
    validated_system_name,
)
from gmxbuilder.pipeline.base import BaseModule, ModuleResult
from gmxbuilder.pipeline.provenance import build_manifest, file_inventory


class CGExportModule(CGWorkflowAdmission, BaseModule):
    name = "cg_export"
    description = "Export a simulation-ready Martini 3 GROMACS package"

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config,
            {
                "output_dir",
                "system_name",
                "write_mdp",
                "execution_hardware",
                "seed",
                "_task_dir",
                "_step_dir",
            },
        )
        if not config.get("output_dir"):
            raise ModuleConfigError("CG export output directory is missing")
        validated_system_name(config.get("system_name"), default="martini3_system")
        return True

    def run(self, system, config: dict) -> ModuleResult:
        config = self.admit(system, config)
        if not system.metadata.get("system_confirmed"):
            raise ModuleConfigError("The final CG system has not been confirmed")
        from gmxbuilder.modules.solvation.membrane_exclusion import assert_membrane_water_free

        assert_membrane_water_free(system)
        output_dir = Path(str(config["output_dir"])).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        system_name = validated_system_name(config.get("system_name"), default="martini3_system")
        structure_dir = output_dir / STRUCTURE_DIR
        topology_dir = output_dir / TOPOLOGY_DIR
        for name in ("input.gro", "input.pdb", "index.ndx", "topol.top"):
            (output_dir / name).unlink(missing_ok=True)
        shutil.rmtree(output_dir / "toppar", ignore_errors=True)
        structure_dir.mkdir(parents=True, exist_ok=True)
        topology_dir.mkdir(parents=True, exist_ok=True)

        GROWriter.write(system.structure, structure_dir / "input.gro", "GMXBUILDER Martini 3")
        PDBWriter.write(
            system.structure,
            structure_dir / "input.pdb",
            "GMXBUILDER Martini 3",
            wrap_ids_for_viewer=True,
        )
        topology = str(system.metadata.get("cg_master_topology", ""))
        if not topology:
            raise ModuleConfigError("Final CG topology text is missing")
        (topology_dir / "topol.top").write_text(topology, encoding="utf-8")
        # ``toppar`` moves *with* topol.top rather than being renamed. The
        # master topology is COBY's own output and its include lines are not
        # rewritten here; keeping the two in the same relative position is what
        # makes that safe. ``toppar`` is also what a Martini user expects to
        # find, so the atomistic ``forcefield`` name is not imposed on it.
        toppar = topology_dir / "toppar"
        materialize_assets(toppar)
        protein_texts = {
            name: text
            for name, text in dict(system.metadata.get("cg_topology_texts") or {}).items()
            if name.endswith(".itp") and name != "martini_v3.0.0.itp"
        }
        write_topology_texts(protein_texts, toppar)
        write_index(system, structure_dir / "index.ndx")

        write_mdp = config.get("write_mdp", True) is not False
        sim = normalize_protocol(
            system.metadata.get("simparams"),
            has_membrane=system.metadata.get("cg_environment") == "bilayer",
            velocity_seed=derive_velocity_seed(system.metadata.get("seed", config.get("seed", 42))),
        )
        stages: list[tuple[str, str]] = []
        if write_mdp:
            stages = write_mdp_files(output_dir / "mdp", sim)
            write_run_script(
                output_dir / "run_md.sh",
                stages,
                sim,
                config.get("execution_hardware"),
            )
        if write_mdp:
            self._validate_with_gromacs(output_dir, sim)
        manifest = load_manifest()
        readme = [
            "GMXBUILDER Martini 3 simulation package",
            "",
            f"System: {system_name}",
            f"Force field: {manifest['force_field']}",
            f"Bundle: {manifest['bundle_id']}",
            "Water: regular Martini W",
            f"Velocity seed: {sim['velocity_seed']}",
            "",
            *(
                ["Run: ./run_md.sh", "Override GROMACS command: GMX=/path/to/gmx ./run_md.sh"]
                if write_mdp
                else ["Dry bilayer geometry package: no solvated simulation protocol is included."]
            ),
            "",
            (
                "Scientific boundary: this package is coarse-grained and must not be mixed with "
                "atomistic parameters."
            ),
            (
                "Review equilibration and production length for the scientific question before "
                "production use."
            ),
        ]
        (output_dir / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
        (output_dir / "CITATIONS.json").write_text(
            json.dumps(manifest["citations"], indent=2) + "\n", encoding="utf-8"
        )
        package_manifest = {
            "schema": 1,
            "resolution": "coarse-grained",
            "force_field": "Martini 3",
            "bundle_id": manifest["bundle_id"],
            "beads": system.num_atoms,
            "coordinate_source": "exact cg_system Check checkpoint",
            "simulation_ready": write_mdp,
            "gromacs_validation": "grompp-maxwarn-0" if write_mdp else "not-requested",
            "protocol": sim if write_mdp else None,
        }
        # The same record the atomistic exporter writes -- what was asked for
        # at each step, what came out, how long it took, and every file with
        # its size. What it deliberately does *not* claim is replay: a Martini
        # system is built from command-line options rather than a
        # PipelineConfig, so there is no `modules` block to feed back in, and
        # promising one would be a lie a reviewer would discover the hard way.
        package_manifest["replayable"] = False
        package_manifest["provenance"] = build_manifest(
            system,
            system_name=system_name,
            seed=int(system.metadata.get("seed", config.get("seed", 42))),
            files=file_inventory(output_dir),
        )["provenance"]
        (output_dir / "manifest.json").write_text(
            json.dumps(package_manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
        archive_path = confined_archive_path(output_dir, system_name)
        if archive_path.exists():
            archive_path.unlink()
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in self._archive_members(output_dir, write_mdp=write_mdp):
                archive.write(path, path.relative_to(output_dir).as_posix())
        record_authoritative_archive(archive_path)
        output = system.copy()
        output.metadata["export_archive"] = archive_path.name
        return ModuleResult(
            True,
            output,
            [
                "Packaged exact confirmed Martini 3 coordinates without rebuilding",
                (
                    "Strict GROMACS grompp topology/MDP validation passed with maxwarn=0"
                    if write_mdp
                    else "Dry geometry export requested; GROMACS validation was not required"
                ),
                (
                    f"Wrote {len(stages)} serial GROMACS stages and one-click run script"
                    if write_mdp
                    else "Wrote a geometry/topology-only dry bilayer package"
                ),
                f"Archive: {archive_path.name}",
            ],
        )

    @staticmethod
    def _archive_members(output_dir: Path, *, write_mdp: bool) -> list[Path]:
        """Return only files produced for the current Martini export."""
        members = {
            output_dir / STRUCTURE_DIR / "input.gro",
            output_dir / STRUCTURE_DIR / "input.pdb",
            output_dir / STRUCTURE_DIR / "index.ndx",
            output_dir / TOPOLOGY_DIR / "topol.top",
            output_dir / "README.txt",
            output_dir / "CITATIONS.json",
            output_dir / "manifest.json",
        }
        top = output_dir / TOPOLOGY_DIR / "topol.top"
        include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"')
        pending = [top]
        while pending:
            current = pending.pop()
            if not current.is_file() or current.is_symlink():
                continue
            for line in current.read_text(errors="replace").splitlines():
                match = include_pattern.match(line)
                if not match:
                    continue
                included = (current.parent / match.group(1)).resolve()
                if output_dir.resolve() not in included.parents:
                    raise ModuleConfigError("Martini topology include escapes the export directory")
                if included not in members:
                    members.add(included)
                    pending.append(included)
        if write_mdp:
            members.add(output_dir / "run_md.sh")
            members.update((output_dir / "mdp").glob("*.mdp"))
        return sorted(path for path in members if path.is_file() and not path.is_symlink())

    @staticmethod
    def _validate_with_gromacs(output_dir: Path, sim: dict) -> None:
        """Validate coordinate/topology order without retaining temporary TPR files."""
        from gmxbuilder.runtime.hardware import find_gromacs_executable

        executable = find_gromacs_executable()
        if not executable:
            raise ModuleConfigError("GROMACS is required to validate a Martini 3 export")
        with tempfile.TemporaryDirectory(
            prefix="gmxbuilder-cg-grompp-", dir=output_dir.parent
        ) as temporary:
            scratch = Path(temporary)
            mini_mdp = output_dir / "mdp" / "mini.mdp"
            run_checked(
                [
                    executable,
                    "grompp",
                    "-f",
                    str(mini_mdp),
                    "-c",
                    f"{STRUCTURE_DIR}/input.gro",
                    "-p",
                    f"{TOPOLOGY_DIR}/topol.top",
                    "-n",
                    f"{STRUCTURE_DIR}/index.ndx",
                    "-o",
                    str(scratch / "validation.tpr"),
                    "-po",
                    str(scratch / "validation.mdp"),
                    "-maxwarn",
                    "0",
                ],
                cwd=output_dir,
                timeout=120.0,
            )
