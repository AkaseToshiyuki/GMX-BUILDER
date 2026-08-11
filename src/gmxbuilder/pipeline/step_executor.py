"""Step executor for incremental checkpoint-based pipeline execution.

Each interactive pipeline step reads the previous step's checkpoint, runs one
module, and writes its own checkpoint.  Finalization reads the last confirmed
coordinate checkpoint and only assigns topology / writes the downloadable
package; coordinate-building modules are never re-run.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

from gmxbuilder.core.system import System
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.pipeline.base import BaseModule, ModuleResult

logger = logging.getLogger(__name__)


def _serialized_task_operation(method):
    """Keep every checkpoint-changing operation for one task strictly serial."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._operation_lock:
            return method(self, *args, **kwargs)
    return wrapped

# ---------------------------------------------------------------------------
# Step definitions for each pipeline
# ---------------------------------------------------------------------------

# Membrane protein pipeline steps (order matters!)
# Scientifically constrained order: read → parameterize → modify → orient → membrane
MEMBRANE_STEPS = [
    "input",
    "forcefield",
    "structure",
    "orient",
    "membrane",
    "solvation",
    "ions",
    "topology",
    "export",
]

# Solution-phase pipeline steps (no membrane/orientation)
SOLUTION_STEPS = [
    "input",
    "forcefield",
    "structure",
    "solvation",
    "ions",
    "topology",
    "export",
]

# Protein-free pure bilayer. Solvation and ions are optional in the final
# pipeline configuration, but remain ordered here when the user enables them.
PURE_MEMBRANE_STEPS = [
    "forcefield",
    "membrane",
    "solvation",
    "ions",
    "topology",
    "export",
]

# Liquid-only pipeline steps (no protein)
LIQUID_STEPS = [
    "solvation",
    "ions",
    "topology",
    "export",
]

# Independent Martini 3 coarse-grained workflow.  These names intentionally do
# not overlap the atomistic scientific modules, so each implementation remains
# separately maintainable and no atomistic assumptions leak into CG systems.
COARSE_GRAINED_STEPS = [
    "input",
    "cg_model",
    "cg_mapping",
    "cg_environment",
    "cg_solvation",
    "cg_system",
    "topology",
    "export",
]

# ---------------------------------------------------------------------------
# Module factory — returns the BaseModule instance for a given step name
# ---------------------------------------------------------------------------

_MODULE_CACHE: dict[tuple[str, str], BaseModule] = {}


def _get_module(step_name: str, pipeline_type: str) -> BaseModule:
    """Return the task-specific module for *step_name* and pipeline type."""
    cache_key = (pipeline_type, step_name)
    if cache_key not in _MODULE_CACHE:
        if pipeline_type == "coarse-grained":
            from gmxbuilder.modules.coarse_grained import (
                CGEnvironmentModule,
                CGExportModule,
                CGInputModule,
                CGMappingModule,
                CGModelModule,
                CGSolvationModule,
                CGSystemCheckModule,
                CGTopologyModule,
            )
            cg_modules = {
                "input": CGInputModule,
                "cg_model": CGModelModule,
                "cg_mapping": CGMappingModule,
                "cg_environment": CGEnvironmentModule,
                "cg_solvation": CGSolvationModule,
                "cg_system": CGSystemCheckModule,
                "topology": CGTopologyModule,
                "export": CGExportModule,
            }
            module_class = cg_modules.get(step_name)
            if module_class is None:
                raise KeyError(f"Unknown Martini 3 step: {step_name}")
            _MODULE_CACHE[cache_key] = module_class()
            return _MODULE_CACHE[cache_key]
        if pipeline_type == "solvator":
            from gmxbuilder.modules.solution import (
                SolutionExportModule,
                SolutionForceFieldSelector,
                SolutionInputModule,
                SolutionIonBuilder,
                SolutionSolvationBuilder,
                SolutionStructureProcessor,
                SolutionTopologyAssigner,
            )
            solution_modules = {
                "input": SolutionInputModule,
                "forcefield": SolutionForceFieldSelector,
                "structure": SolutionStructureProcessor,
                "solvation": SolutionSolvationBuilder,
                "ions": SolutionIonBuilder,
                "topology": SolutionTopologyAssigner,
                "export": SolutionExportModule,
            }
            module_class = solution_modules.get(step_name)
            if module_class is None:
                raise KeyError(f"Unknown Solvator step: {step_name}")
            _MODULE_CACHE[cache_key] = module_class()
            return _MODULE_CACHE[cache_key]
        if pipeline_type == "pure-membrane":
            from gmxbuilder.modules.pure_membrane import (
                PureMembraneBuilder,
                PureMembraneExportModule,
                PureMembraneForceFieldSelector,
                PureMembraneIonBuilder,
                PureMembraneSolvationBuilder,
                PureMembraneTopologyAssigner,
            )
            pure_modules = {
                "forcefield": PureMembraneForceFieldSelector,
                "membrane": PureMembraneBuilder,
                "solvation": PureMembraneSolvationBuilder,
                "ions": PureMembraneIonBuilder,
                "topology": PureMembraneTopologyAssigner,
                "export": PureMembraneExportModule,
            }
            module_class = pure_modules.get(step_name)
            if module_class is None:
                raise KeyError(f"Unknown Pure Bilayer step: {step_name}")
            _MODULE_CACHE[cache_key] = module_class()
            return _MODULE_CACHE[cache_key]
        if step_name == "input":
            from gmxbuilder.modules.input.pdb_input import PDBInputModule
            _MODULE_CACHE[cache_key] = PDBInputModule()
        elif step_name == "orient":
            from gmxbuilder.modules.membrane.orient_module import OrientModule
            _MODULE_CACHE[cache_key] = OrientModule()
        elif step_name == "structure":
            from gmxbuilder.modules.modifications.processor import StructureProcessor
            _MODULE_CACHE[cache_key] = StructureProcessor()
        elif step_name == "membrane":
            from gmxbuilder.modules.membrane.builder import MembraneBuilder
            _MODULE_CACHE[cache_key] = MembraneBuilder()
        elif step_name == "solvation":
            from gmxbuilder.modules.solvation.solvate import SolvationBuilder
            _MODULE_CACHE[cache_key] = SolvationBuilder()
        elif step_name == "ions":
            from gmxbuilder.modules.ions.add_ions import IonBuilder
            _MODULE_CACHE[cache_key] = IonBuilder()
        elif step_name == "forcefield":
            from gmxbuilder.modules.forcefield.selector import ForceFieldSelector
            _MODULE_CACHE[cache_key] = ForceFieldSelector()
        elif step_name == "topology":
            from gmxbuilder.modules.forcefield.assign import ForceFieldAssigner
            _MODULE_CACHE[cache_key] = ForceFieldAssigner()
        elif step_name == "export":
            from gmxbuilder.modules.export.exporter import ExportModule
            _MODULE_CACHE[cache_key] = ExportModule()
        else:
            raise KeyError(f"Unknown step: {step_name}")
    return _MODULE_CACHE[cache_key]


def get_pipeline_steps(pipeline_type: str) -> list[str]:
    """Return the ordered step list for a pipeline type."""
    if pipeline_type == "membrane-bilayer":
        return list(MEMBRANE_STEPS)
    elif pipeline_type == "pure-membrane":
        return list(PURE_MEMBRANE_STEPS)
    elif pipeline_type == "solvator":
        return list(SOLUTION_STEPS)
    elif pipeline_type == "liquid-builder":
        return list(LIQUID_STEPS)
    elif pipeline_type == "coarse-grained":
        return list(COARSE_GRAINED_STEPS)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

class StepRunner:
    """Execute one pipeline step: load checkpoint → run module → save checkpoint."""

    def __init__(self, task_dir: Path, pipeline_type: str = "membrane-bilayer"):
        self.task_dir = Path(task_dir)
        self.pipeline_type = pipeline_type
        self.steps_dir = self.task_dir / "steps"
        # A browser can issue overlapping requests (double click, refresh,
        # multiple tabs). Serializing at the runner boundary prevents a later
        # step from reading a checkpoint while its predecessor is still being
        # written. RLock allows run_export() to call run_step().
        self._operation_lock = threading.RLock()

    def step_dir(self, step_name: str) -> Path:
        return self.steps_dir / step_name

    def prev_step_dir(self, step_name: str) -> Path | None:
        """Find the directory of the step immediately before *step_name*."""
        steps = get_pipeline_steps(self.pipeline_type)
        try:
            idx = steps.index(step_name)
        except ValueError:
            return None
        if idx == 0:
            return None  # No previous step
        prev_name = steps[idx - 1]
        prev_dir = self.step_dir(prev_name)
        if prev_dir.exists():
            return prev_dir
        return None

    def has_checkpoint(self, step_name: str) -> bool:
        """Check if a checkpoint exists for this step (both npz and json)."""
        d = self.step_dir(step_name)
        return d.exists() and (d / "system.npz").exists() and (d / "system.json").exists()

    def load_system(self, step_name: str) -> System | None:
        """Load the system from *step_name* checkpoint, or None."""
        d = self.step_dir(step_name)
        if not d.exists() or not (d / "system.npz").exists():
            return None
        return System.load_checkpoint(d)

    def invalidate_downstream(self, step_name: str) -> list[str]:
        """Remove checkpoints that were derived from an older upstream state."""
        steps = get_pipeline_steps(self.pipeline_type)
        try:
            start = steps.index(step_name) + 1
        except ValueError:
            return []

        root = self.steps_dir.resolve()
        invalidated: list[str] = []
        for later_step in steps[start:]:
            checkpoint = self.step_dir(later_step)
            if not checkpoint.exists():
                continue
            resolved = checkpoint.resolve()
            if resolved == root or root not in resolved.parents:
                raise OSError(
                    f"Refusing to remove downstream checkpoint outside task steps: {resolved}"
                )
            shutil.rmtree(resolved)
            invalidated.append(later_step)
        return invalidated

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @_serialized_task_operation
    def run_step(
        self,
        step_name: str,
        config: dict[str, Any],
        *,
        initial_system: System | None = None,
        pdb_path: str | None = None,
    ) -> dict:
        """Execute a single pipeline step.

        Parameters
        ----------
        step_name : str
            The pipeline step to execute (e.g. "membrane", "solvation").
        config : dict
            Module configuration dict.
        initial_system : System or None
            Only used for the FIRST step ("input") — provides the empty seed system.
        pdb_path : str or None
            Only used for the "input" step — path to the input PDB file.

        Returns
        -------
        dict with keys:
            status : "ok" | "error"
            step : str
            log : list[str]
            metrics : dict  — key system metrics for the frontend viewer
            viewer_pdb_path : str  — path to the viewer PDB file
            error : str (if status=="error")
        """
        t0 = time.time()

        # ---- 1. Load previous system ----
        # Steps MUST run in order.  Each step reads exactly the previous
        # step's checkpoint — no fallbacks.  If the previous checkpoint is
        # missing the pipeline is in an invalid state and must be re-run
        # from the beginning.
        first_step = get_pipeline_steps(self.pipeline_type)[0]
        if step_name == first_step:
            # First step — no previous checkpoint required
            system = initial_system or System(
                structure=Structure(
                    coordinates=np.empty((0, 3)),
                    box_vectors=np.eye(3) * 10.0,
                ),
                metadata={},
            )
        else:
            prev_dir = self.prev_step_dir(step_name)
            if prev_dir is None or not prev_dir.exists():
                prev_name = get_pipeline_steps(self.pipeline_type)[
                    get_pipeline_steps(self.pipeline_type).index(step_name) - 1
                ] if step_name in get_pipeline_steps(self.pipeline_type) else "previous"
                return {
                    "status": "error",
                    "step": step_name,
                    "error": (
                        f"Previous step '{prev_name}' has not been run. "
                        f"Steps must be executed in order — run '{prev_name}' first."
                    ),
                }
            try:
                system = System.load_checkpoint(prev_dir)
            except (OSError, KeyError, ValueError) as exc:
                logger.error("Failed to load checkpoint from %s: %s", prev_dir, exc)
                return {
                    "status": "error", "step": step_name,
                    "error": f"Failed to load checkpoint: {exc}",
            }

        # ---- 2. Inject config needed by the module ----
        # Always shallow-copy to avoid mutating the caller's config dict
        config = dict(config)
        # The independent CG modules need task-private tool working paths.
        # Atomistic module schemas predate these keys and must remain unchanged.
        # Overwrite rather than setdefault so API clients cannot redirect files.
        if self.pipeline_type == "coarse-grained":
            config["_task_dir"] = str(self.task_dir.resolve())
            config["_step_dir"] = str(self.step_dir(step_name).resolve())
        if step_name == "input" and pdb_path:
            config["pdb"] = pdb_path
        if step_name == "export":
            config.setdefault("output_dir", str(self.step_dir("export")))
            config.setdefault("system_name", system.metadata.get("system_name", "system"))
        # Forward seed from system metadata into config
        if "seed" not in config:
            config["seed"] = system.metadata.get("seed", 42)

        # ---- 3. Run module ----
        module = _get_module(step_name, self.pipeline_type)
        try:
            module.validate_config(config)
        except Exception as exc:
            return {"status": "error", "step": step_name,
                    "error": f"Config validation failed: {exc}"}

        try:
            result: ModuleResult = module.execute(system, config)
        except ModuleConfigError as exc:
            logger.warning("Step %s rejected invalid configuration: %s", step_name, exc)
            return {
                "status": "error", "step": step_name,
                "error": str(exc),
            }
        except Exception as exc:
            logger.exception("Unhandled failure while running step %s", step_name)
            return {
                "status": "error", "step": step_name,
                "error": f"Step execution failed: {exc}",
            }
        if not result.success:
            return {
                "status": "error",
                "step": step_name,
                "error": "Module reported failure",
                "log": result.log,
            }

        system = result.system

        # A successful rerun changes the state consumed by every later step.
        # Remove their checkpoints before saving this one so a refreshed UI
        # cannot silently reuse results derived from stale coordinates/config.
        try:
            invalidated_steps = self.invalidate_downstream(step_name)
        except OSError as exc:
            logger.error("Failed to invalidate downstream checkpoints: %s", exc)
            return {
                "status": "error", "step": step_name,
                "error": f"Could not invalidate stale downstream checkpoints: {exc}",
            }

        # ---- 4. Save checkpoint ----
        out_dir = self.step_dir(step_name)
        system.save_checkpoint(out_dir)

        # Ion Check is the first point at which the final atom ordering is
        # known. Materialize the canonical index here so Simulation Parameters
        # can refer to groups that already exist on disk. Export regenerates
        # the file from the same helper as a final consistency check.
        index_path = None
        if step_name == "ions":
            index_path = out_dir / "index.ndx"
            try:
                from gmxbuilder.modules.export.exporter import ExportModule

                ExportModule._write_index(system, index_path)
            except Exception as exc:
                logger.exception("Failed to write Ion Check index file")
                shutil.rmtree(out_dir, ignore_errors=True)
                return {
                    "status": "error",
                    "step": step_name,
                    "error": f"Could not write the validated index groups: {exc}",
                }

        # ---- 5. Write viewer PDB ----
        viewer_path = out_dir / "viewer.pdb"
        try:
            system.write_viewer_pdb(viewer_path)
        except Exception:
            pass  # Non-critical — frontend falls back to previous viewer

        # ---- 6. Compute metrics ----
        metrics = _compute_step_metrics(system, step_name)

        return {
            "status": "ok",
            "step": step_name,
            "log": result.log,
            "metrics": metrics,
            "viewer_pdb_path": str(viewer_path) if viewer_path.exists() else None,
            "index_path": str(index_path) if index_path and index_path.exists() else None,
            "invalidated_steps": invalidated_steps,
            "elapsed_s": round(time.time() - t0, 1),
        }

    # ------------------------------------------------------------------
    # Final export (last step)
    # ------------------------------------------------------------------

    @_serialized_task_operation
    def run_export(
        self,
        config: dict[str, Any],
    ) -> dict:
        """Run the final export step and return download info."""
        result = self.run_step("export", config)
        if result["status"] != "ok":
            return result

        out_dir = self.step_dir("export")
        # Find the ZIP file
        zip_files = list(out_dir.glob("*.zip"))
        zip_path = None
        if not zip_files:
            # Also check the output subdirectory set by export module
            for d in out_dir.rglob("*.zip"):
                zip_files.append(d)
        if zip_files:
            zip_path = str(max(zip_files, key=lambda p: p.stat().st_size))

        return {
            **result,
            "download_url": f"/api/step/{self.task_dir.name}/export/download" if zip_path else None,
            "zip_path": zip_path,
        }

    @_serialized_task_operation
    def finalize_from_checkpoint(
        self,
        source_step: str,
        *,
        topology_config: dict[str, Any] | None = None,
        export_config: dict[str, Any] | None = None,
        simparams: dict[str, Any] | None = None,
    ) -> dict:
        """Create a package from an already-confirmed coordinate checkpoint.

        This is deliberately separate from :meth:`run_step`: finalization must
        not call membrane, solvation, or ion placement again.  The exported GRO
        is checked against the authoritative checkpoint at GRO precision.
        """
        import numpy as np

        if source_step not in {"membrane", "ions", "cg_system"}:
            return {
                "status": "error",
                "step": "export",
                "error": f"Unsupported final coordinate checkpoint: {source_step}",
            }
        if not self.has_checkpoint(source_step):
            return {
                "status": "error",
                "step": "export",
                "error": (
                    f"Required '{source_step}' Check checkpoint is missing. "
                    "Return to that step, click Check, and confirm the system first."
                ),
            }

        source = System.load_checkpoint(self.step_dir(source_step))
        source_coordinates = np.array(source.structure.coordinates, copy=True)
        source_box = np.array(source.structure.box_vectors, copy=True)
        source_atom_names = tuple(source.structure.atom_names)
        source_resnames = tuple(source.structure.resnames)
        source_resids = tuple(int(value) for value in source.structure.resids)

        system = source
        if simparams is not None:
            system.metadata["simparams"] = dict(simparams)

        topology_module = _get_module("topology", self.pipeline_type)
        topology_cfg = dict(topology_config or {})
        topology_module.validate_config(topology_cfg)
        topology_result = topology_module.execute(system, topology_cfg)
        if not topology_result.success:
            return {
                "status": "error",
                "step": "topology",
                "error": "Topology assignment failed",
                "log": topology_result.log,
            }
        system = topology_result.system
        if (
            not np.array_equal(system.structure.coordinates, source_coordinates)
            or not np.array_equal(system.structure.box_vectors, source_box)
            or tuple(system.structure.atom_names) != source_atom_names
            or tuple(system.structure.resnames) != source_resnames
            or tuple(int(value) for value in system.structure.resids) != source_resids
        ):
            raise RuntimeError(
                "Topology assignment changed confirmed coordinates or atom ordering"
            )
        if self.pipeline_type == "coarse-grained":
            from gmxbuilder.modules.coarse_grained.assets import validate_toolchain
            gromacs_compatibility = {
                "compatible": True,
                "model": "Martini 3",
                "toolchain": validate_toolchain(),
            }
        else:
            from gmxbuilder.modules.forcefield.catalog import validate_local_gromacs
            gromacs_compatibility = validate_local_gromacs(
                str(system.metadata.get("force_field", "amber14sb"))
            )

        topology_dir = self.step_dir("topology")
        shutil.rmtree(topology_dir, ignore_errors=True)
        system.save_checkpoint(topology_dir)

        export_dir = self.step_dir("export")
        shutil.rmtree(export_dir, ignore_errors=True)
        export_cfg = dict(export_config or {})
        export_cfg.setdefault("output_dir", str(export_dir))
        export_cfg.setdefault(
            "system_name", system.metadata.get("system_name", "system")
        )
        export_module = _get_module("export", self.pipeline_type)
        export_module.validate_config(export_cfg)
        export_result = export_module.execute(system, export_cfg)
        if not export_result.success:
            return {
                "status": "error",
                "step": "export",
                "error": "Package export failed",
                "log": topology_result.log + export_result.log,
            }

        from gmxbuilder.io.gro import GROReader

        exported = GROReader().read(export_dir / "input.gro")
        if exported.num_atoms != source.num_atoms:
            raise RuntimeError(
                "Export integrity check failed: GRO atom count differs from the "
                "confirmed checkpoint"
            )
        if not np.allclose(
            exported.coordinates, source_coordinates, rtol=0.0, atol=5.1e-4
        ):
            raise RuntimeError(
                "Export integrity check failed: GRO coordinates differ from the "
                "confirmed checkpoint beyond GRO rounding precision"
            )
        if not np.allclose(exported.box_vectors, source_box, rtol=0.0, atol=5.1e-6):
            raise RuntimeError(
                "Export integrity check failed: GRO box differs from the confirmed "
                "checkpoint beyond GRO rounding precision"
            )

        zip_files = list(export_dir.glob("*.zip"))
        zip_path = max(zip_files, key=lambda path: path.stat().st_mtime_ns) if zip_files else None
        if zip_path is None:
            raise RuntimeError("Export completed without producing a ZIP archive")

        import zipfile

        with zipfile.ZipFile(zip_path) as archive:
            members = set(archive.namelist())
            common_required = {"input.gro", "topol.top", "index.ndx", "README.txt"}
            missing_common = sorted(common_required - members)
            if missing_common:
                raise RuntimeError(
                    "Export package contract failed; missing required files: "
                    + ", ".join(missing_common)
                )

            mdp_files = sorted(
                name for name in members
                if name.startswith("mdp/") and name.endswith(".mdp")
            )
            write_mdp = export_cfg.get("write_mdp", True) is not False
            has_run_script = "run_md.sh" in members
            if write_mdp:
                missing_simulation = []
                if not has_run_script:
                    missing_simulation.append("run_md.sh")
                if "mdp/mini.mdp" not in members:
                    missing_simulation.append("mdp/mini.mdp")
                if not any(
                    name.startswith("mdp/production") for name in mdp_files
                ):
                    missing_simulation.append("mdp/production*.mdp")
                if missing_simulation:
                    raise RuntimeError(
                        "Simulation-ready package contract failed; missing: "
                        + ", ".join(missing_simulation)
                    )
                run_info = archive.getinfo("run_md.sh")
                if not ((run_info.external_attr >> 16) & 0o111):
                    raise RuntimeError(
                        "Simulation-ready package contract failed; run_md.sh "
                        "is not executable"
                    )
            elif has_run_script or mdp_files:
                raise RuntimeError(
                    "Dry package contract failed; disabled MDP generation must "
                    "not leave an unusable launcher or MDP files"
                )

        package_contents = {
            "simulation_ready": write_mdp,
            "run_script": "run_md.sh" if has_run_script else None,
            "mdp_files": mdp_files,
            "dry_export": not write_mdp,
        }
        package_message = (
            f"Package manifest verified: run_md.sh + {len(mdp_files)} MDP files"
            if write_mdp
            else "Dry package manifest verified: coordinates and topology only; "
                 "solvation was disabled, so no launcher or MDP files were included"
        )

        return {
            "status": "ok",
            "step": "export",
            "system": system,
            "source_step": source_step,
            "zip_path": str(zip_path),
            "package_contents": package_contents,
            "log": (
                [
                    f"Loaded exact coordinates from confirmed {source_step} checkpoint",
                    "Assigned topology without changing coordinates or atom order",
                ]
                + topology_result.log
                + [gromacs_compatibility]
                + export_result.log
                + [
                    package_message,
                    "Export integrity check passed: atom count, coordinates, and box "
                    "match the confirmed checkpoint"
                ]
            ),
        }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_step_metrics(system: System, step_name: str) -> dict:
    """Compute frontend-relevant metrics for this step."""
    metrics: dict[str, Any] = {
        "num_atoms": system.num_atoms,
        "box_dimensions_nm": [round(v, 3) for v in system.structure.dimensions().tolist()],
        "components": [],
    }

    for comp in system.components:
        info = {
            "name": comp.name,
            "kind": comp.kind.name,
            "n_atoms": len(comp.atom_indices),
        }
        n_mol = comp.metadata.get("n_molecules")
        if n_mol is not None:
            info["n_molecules"] = n_mol
        for key in ("water_model", "volume_nm3"):
            if key in comp.metadata:
                info[key] = comp.metadata[key]
        metrics["components"].append(info)

    if step_name == "solvation":
        metrics["water_model"] = system.metadata.get("water_model")
        metrics["solvation"] = system.metadata.get("solvation", {})

    if step_name == "ions":
        metrics["ions"] = system.metadata.get("ions", {})

    if step_name == "cg_mapping":
        metrics["cg_mapping"] = system.metadata.get("cg_mapping", {})

    if step_name == "input":
        metrics["input_repair"] = system.metadata.get("input_repair", {
            "status": "not_needed",
            "residues_repaired": 0,
            "atoms_added": 0,
            "residues": [],
            "validation": "No missing standard protein heavy atoms detected.",
        })
        metrics["input_modifications"] = system.metadata.get(
            "input_modifications",
            {"detected": 0, "recognized": 0, "records": [], "warnings": []},
        )
        protein_atoms = {
            int(index)
            for component in system.component_by_kind(ComponentKind.PROTEIN)
            for index in component.atom_indices
        }
        chains: dict[str, list[dict]] = {}
        seen_residues: set[tuple[str, int]] = set()
        for index in range(system.structure.num_atoms):
            if index not in protein_atoms:
                continue
            chain = str(system.structure.chain_ids[index]).strip() or "A"
            resid = int(system.structure.resids[index])
            key = (chain, resid)
            if key in seen_residues:
                continue
            seen_residues.add(key)
            chains.setdefault(chain, []).append({
                "resname": str(system.structure.resnames[index]).strip().upper(),
                "resid": resid,
                "is_protein": True,
            })
        metrics["input_sequences"] = [
            {"chain_id": chain, "length": len(residues), "residues": residues}
            for chain, residues in chains.items()
        ]
        metrics["input_nucleic_acids"] = [
            {
                "name": component.name,
                "chain_id": component.metadata.get("chain_id", ""),
                "polymer_type": component.metadata.get("polymer_type", "unknown"),
                "n_residues": component.metadata.get("n_residues", 0),
                "unsupported_residues": component.metadata.get(
                    "unsupported_residues", []
                ),
            }
            for component in system.component_by_kind(ComponentKind.NUCLEIC_ACID)
        ]

    if step_name == "forcefield":
        metrics["forcefield_resolution"] = {
            "requested_protein_ff": system.metadata.get("requested_force_field"),
            "effective_protein_ff": system.metadata.get("force_field"),
            "effective_lipid_ff": system.metadata.get("lipid_ff"),
            "effective_ligand_ff": system.metadata.get("ligand_ff"),
            "water_model": system.metadata.get("water_model"),
            "gaff_lipids": system.metadata.get("gaff_lipids", []),
            "ligand_parameters": system.metadata.get("ligand_parameters", {}),
            "nucleic_acid_backend": (
                "gromacs-pdb2gmx-charmm36"
                if system.component_by_kind(ComponentKind.NUCLEIC_ACID)
                else "none"
            ),
        }

    if step_name == "structure":
        metrics["modification_geometry"] = system.metadata.get(
            "modification_geometry", []
        )
        metrics["crosslinks"] = system.metadata.get("crosslinks", [])
        metrics["nucleic_acids"] = [
            {
                key: record.get(key)
                for key in (
                    "molecule_type", "polymer_type", "chain_id", "net_charge",
                    "atom_count", "residue_count", "backend",
                )
            }
            for record in system.metadata.get("native_nucleic_topologies", [])
        ]

    if step_name == "orient":
        metrics["orientation"] = system.metadata.get("_orient_params", {})
        metrics["orientation_method"] = system.metadata.get("_orientation_method")
        metrics["orientation_quality"] = system.metadata.get(
            "_orientation_quality", {}
        )

    return metrics
