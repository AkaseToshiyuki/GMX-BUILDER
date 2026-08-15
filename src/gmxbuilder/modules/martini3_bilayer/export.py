"""Self-contained export for Martini 3 systems; no atomistic exporter reuse."""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from pathlib import Path

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.io.gro import GROWriter
from gmxbuilder.io.pdb import PDBWriter
from gmxbuilder.modules.coarse_grained.assets import load_manifest, materialize_assets
from gmxbuilder.modules.coarse_grained.common import run_checked, write_topology_texts
from gmxbuilder.modules.coarse_grained.protocol import (
    normalize_protocol,
    write_index,
    write_mdp_files,
    write_run_script,
)
from gmxbuilder.pipeline.base import BaseModule, ModuleResult


class CGExportModule(BaseModule):
    name = "cg_export"
    description = "Export a simulation-ready Martini 3 GROMACS package"

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config, {"output_dir", "system_name", "write_mdp", "seed", "_task_dir", "_step_dir"}
        )
        if not config.get("output_dir"):
            raise ModuleConfigError("CG export output directory is missing")
        return True

    def run(self, system, config: dict) -> ModuleResult:
        if not system.metadata.get("system_confirmed"):
            raise ModuleConfigError("The final CG system has not been confirmed")
        output_dir = Path(str(config["output_dir"])).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        system_name = str(config.get("system_name") or "martini3_system")
        GROWriter.write(system.structure, output_dir / "input.gro", "GMXBUILDER Martini 3")
        PDBWriter.write(
            system.structure,
            output_dir / "input.pdb",
            "GMXBUILDER Martini 3",
            wrap_ids_for_viewer=True,
        )
        topology = str(system.metadata.get("cg_master_topology", ""))
        if not topology:
            raise ModuleConfigError("Final CG topology text is missing")
        (output_dir / "topol.top").write_text(topology, encoding="utf-8")
        toppar = output_dir / "toppar"
        materialize_assets(toppar)
        protein_texts = {
            name: text
            for name, text in dict(system.metadata.get("cg_topology_texts") or {}).items()
            if name.endswith(".itp") and name != "martini_v3.0.0.itp"
        }
        write_topology_texts(protein_texts, toppar)
        write_index(system, output_dir / "index.ndx")

        write_mdp = config.get("write_mdp", True) is not False
        sim = normalize_protocol(
            system.metadata.get("simparams"),
            has_membrane=system.metadata.get("cg_environment") == "bilayer",
        )
        stages: list[tuple[str, str]] = []
        if write_mdp:
            stages = write_mdp_files(output_dir / "mdp", sim)
            write_run_script(output_dir / "run_md.sh", stages, sim)
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
            "",
            *(
                ["Run: ./run_md.sh", "Override GROMACS command: GMX=/path/to/gmx ./run_md.sh"]
                if write_mdp
                else ["Dry bilayer geometry package: no solvated simulation protocol is included."]
            ),
            "",
            "Scientific boundary: this package is coarse-grained and must not be mixed with atomistic parameters.",
            "Review equilibration and production length for the scientific question before production use.",
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
        (output_dir / "manifest.json").write_text(
            json.dumps(package_manifest, indent=2) + "\n", encoding="utf-8"
        )
        archive_path = output_dir / f"{sim['system_name']}.zip"
        if archive_path.exists():
            archive_path.unlink()
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in self._archive_members(output_dir, write_mdp=write_mdp):
                archive.write(path, path.relative_to(output_dir).as_posix())
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
            output_dir / name
            for name in (
                "input.gro",
                "input.pdb",
                "topol.top",
                "index.ndx",
                "README.txt",
                "CITATIONS.json",
                "manifest.json",
            )
        }
        top = output_dir / "topol.top"
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
                included = (output_dir / match.group(1)).resolve()
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
                    "input.gro",
                    "-p",
                    "topol.top",
                    "-n",
                    "index.ndx",
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
