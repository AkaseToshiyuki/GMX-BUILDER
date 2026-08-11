"""CLI entry point for GMXBUILDER."""

from __future__ import annotations

import json
import sys
import shutil
import tempfile
from pathlib import Path

import click

from gmxbuilder import __version__


def _prepare_cli_build_config(cfg, output: str | None = None):
    """Bind top-level CLI options to the modules that consume them."""
    output_dir = Path(output) if output else cfg.output_dir

    modules = {
        name: dict(values)
        for name, values in cfg.modules.items()
    }
    for name in {
        "input", "forcefield", "structure", "orient",
        "membrane", "solvation", "ions",
    }:
        if name in modules:
            modules[name].setdefault("seed", cfg.seed)

    if "export" in modules:
        modules["export"]["output_dir"] = str(output_dir)
        modules["export"].setdefault("system_name", cfg.system_name)

    return cfg.model_copy(update={"modules": modules, "output_dir": output_dir})


@click.group()
@click.version_option(version=__version__, prog_name="gmxbuilder")
def main():
    """GMXBUILDER — Build GROMACS molecular dynamics simulation systems.

    A tool for constructing phospholipid bilayers,
    embedding membrane proteins, solvating systems, and generating
    production-ready simulation packages.
    """
    pass


@main.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML configuration file",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(),
    help="Override output directory",
)
def build(config: str, output: str | None):
    """Build a simulation system from a YAML configuration file.

    Example:

        gmxbuilder build -c build.yaml -o ./my_system
    """
    from gmxbuilder.pipeline.config import PipelineConfig
    from gmxbuilder.pipeline.pipeline import Pipeline

    click.echo(f"GMXBUILDER v{__version__}")
    click.echo(f"Loading config: {config}")

    # Load configuration
    cfg = PipelineConfig.from_yaml(config)

    cfg = _prepare_cli_build_config(cfg, output)

    click.echo(f"Output directory: {cfg.output_dir}")
    click.echo(f"System name: {cfg.system_name}")

    # Build pipeline
    pipeline = Pipeline.create_default()

    # Run
    from gmxbuilder.core.system import System
    from gmxbuilder.core.structure import Structure
    import numpy as np

    # Start with an empty system; forward simparams to metadata
    metadata = {"seed": cfg.seed, "system_name": cfg.system_name}
    if "simparams" in cfg.modules:
        metadata["simparams"] = cfg.modules["simparams"]
    initial = System(
        structure=Structure(
            coordinates=np.empty((0, 3)),
            box_vectors=np.eye(3) * 10.0,
        ),
        metadata=metadata,
    )

    try:
        result = pipeline.run(initial, cfg)
        click.echo(f"\nBuild complete — {result.system.num_atoms} atoms total")
        click.echo(f"Components: {[c.name for c in result.system.components]}")
        for entry in result.log:
            click.echo(f"  {entry}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


def _parse_cg_lipids(values: tuple[str, ...], label: str) -> list[dict]:
    """Parse repeatable CLI NAME:RATIO values without accepting silent defaults."""
    parsed: list[dict] = []
    for raw in values:
        name, separator, ratio_text = raw.partition(":")
        name = name.strip().upper()
        if not separator or not name:
            raise click.BadParameter(f"{label} entries must use NAME:RATIO")
        try:
            ratio = float(ratio_text)
        except ValueError as exc:
            raise click.BadParameter(f"{label} ratio must be numeric: {raw}") from exc
        if ratio <= 0:
            raise click.BadParameter(f"{label} ratio must be positive: {raw}")
        parsed.append({"name": name, "ratio": ratio})
    if not parsed:
        raise click.BadParameter(f"At least one {label} lipid is required")
    return parsed


@main.command("coarse-grained")
@click.option("--mode", type=click.Choice(["solution", "bilayer"]), default="bilayer", show_default=True)
@click.option("--pdb", type=click.Path(exists=True, dir_okay=False), help="Standard-protein PDB; omit for a protein-free bilayer")
@click.option("--upper", "upper_lipids", multiple=True, default=("POPC:1",), show_default=True, help="Upper leaflet NAME:RATIO; repeat for mixtures")
@click.option("--lower", "lower_lipids", multiple=True, help="Lower leaflet NAME:RATIO; defaults to upper")
@click.option("--box-xy", default=12.0, type=click.FloatRange(5.0, 40.0), show_default=True)
@click.option("--box-z", default=14.0, type=click.FloatRange(6.0, 50.0), show_default=True)
@click.option("--salt", default=0.15, type=click.FloatRange(0.0, 1.0), show_default=True, help="Target NaCl concentration in mol/L")
@click.option("--dry", is_flag=True, help="Bilayer geometry/topology only; no simulation MDPs")
@click.option("--protein-model", type=click.Choice(["folded", "tm_helix", "disordered"]), default="folded", show_default=True)
@click.option("--secondary-structure", default="auto", show_default=True, help="auto or a manual DSSP string")
@click.option("--elastic/--no-elastic", default=True, show_default=True)
@click.option("--rotate-x", default=0.0, type=click.FloatRange(-180, 180), show_default=True)
@click.option("--rotate-y", default=0.0, type=click.FloatRange(-180, 180), show_default=True)
@click.option("--rotate-z", default=0.0, type=click.FloatRange(-180, 180), show_default=True)
@click.option("--z-offset", default=0.0, type=click.FloatRange(-8, 8), show_default=True)
@click.option("--production-ns", default=1000.0, type=click.FloatRange(1, 100000), show_default=True)
@click.option("--threads", default=8, type=click.IntRange(1, 1024), show_default=True)
@click.option("--mpi-ranks", default=1, type=click.IntRange(1, 1024), show_default=True)
@click.option("--gpu-ids", default="0", show_default=True, help="Comma-separated logical GPU IDs; empty disables GPU")
@click.option("--seed", default=42, type=click.IntRange(0, 2_147_483_647), show_default=True)
@click.option("--system-name", default="martini3_system", show_default=True)
@click.option("--output", "output_dir", required=True, type=click.Path(file_okay=False))
@click.option("--yes", "accept", is_flag=True, help="Accept the scientifically checked exact system non-interactively")
def coarse_grained(
    mode: str,
    pdb: str | None,
    upper_lipids: tuple[str, ...],
    lower_lipids: tuple[str, ...],
    box_xy: float,
    box_z: float,
    salt: float,
    dry: bool,
    protein_model: str,
    secondary_structure: str,
    elastic: bool,
    rotate_x: float,
    rotate_y: float,
    rotate_z: float,
    z_offset: float,
    production_ns: float,
    threads: int,
    mpi_ranks: int,
    gpu_ids: str,
    seed: int,
    system_name: str,
    output_dir: str,
    accept: bool,
):
    """Build a complete Martini 3 solution or flat-bilayer system serially.

    Ligands, modified residues, glycans, nucleic acids, curved membranes,
    custom molecules, and backmapping are intentionally outside this command.
    """
    from gmxbuilder.pipeline.step_executor import StepRunner

    include_protein = pdb is not None
    if mode == "solution" and not include_protein:
        raise click.UsageError("--mode solution requires --pdb")
    if dry and mode != "bilayer":
        raise click.UsageError("--dry is available only for a bilayer")
    if threads % mpi_ranks:
        raise click.BadParameter("--threads must be exactly divisible by --mpi-ranks")
    gpu_ids = gpu_ids.strip()
    use_gpu = bool(gpu_ids)
    upper = _parse_cg_lipids(upper_lipids, "upper") if mode == "bilayer" else []
    lower = _parse_cg_lipids(lower_lipids or upper_lipids, "lower") if mode == "bilayer" else []
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise click.ClickException(f"Output directory is not empty: {output}")

    secondary_mode = "auto" if secondary_structure.strip().lower() == "auto" else "manual"
    mapping_config = {
        "protein_model": protein_model,
        "secondary_structure": secondary_mode,
        "secondary_structure_string": "" if secondary_mode == "auto" else secondary_structure.strip().upper(),
        "elastic": elastic,
    }
    environment_config = {
        "environment": mode,
        "box_xy": box_xy,
        "box_z": box_z,
        "rotate_x": rotate_x,
        "rotate_y": rotate_y,
        "rotate_z": rotate_z,
        "z_offset": z_offset,
        "seed": seed,
    }
    if mode == "bilayer":
        environment_config.update({
            "upper_leaflet": upper,
            "lower_leaflet": lower,
            "asymmetric": upper != lower,
        })

    def checked_step(runner: StepRunner, name: str, config: dict) -> None:
        click.echo(f"[{name}] running")
        result = runner.run_step(name, config, pdb_path=pdb if name == "input" else None)
        if result.get("status") != "ok":
            raise click.ClickException(result.get("error", f"{name} failed"))
        for line in result.get("log", []):
            click.echo(f"  {line}")

    with tempfile.TemporaryDirectory(prefix="gmxbuilder-martini3-") as temporary:
        runner = StepRunner(Path(temporary) / "task", pipeline_type="coarse-grained")
        checked_step(runner, "input", {"include_protein": include_protein, "environment": mode})
        checked_step(runner, "cg_model", {"model": "martini3", "water_model": "W"})
        checked_step(runner, "cg_mapping", mapping_config)
        checked_step(runner, "cg_environment", environment_config)
        checked_step(runner, "cg_solvation", {"include_solvent": not dry, "salt_molarity": salt})
        checked_step(runner, "cg_system", {"salt_molarity": salt, "confirm_system": False})

        final = runner.load_system("cg_system")
        if final is None:
            raise click.ClickException("Final CG checkpoint was not created")
        quality = final.metadata.get("cg_scientific_check") or {}
        click.echo(json.dumps(quality, indent=2, sort_keys=True))
        if not accept and not click.confirm("Accept this exact checked system for export?"):
            raise click.Abort()
        final.metadata["system_confirmed"] = True
        final.save_checkpoint(runner.step_dir("cg_system"))

        simparams = {
            "temperature": 310.0,
            "pressure": 1.0,
            "production_ns": production_ns,
            "output_interval_ps": 100.0,
            "equilibration_1": True,
            "equilibration_2": True,
            "use_gpu": use_gpu,
            "gpu_ids": gpu_ids or "0",
            "threads": threads,
            "mpi_ranks": mpi_ranks,
            "system_name": system_name,
        }
        result = runner.finalize_from_checkpoint(
            "cg_system",
            topology_config={},
            export_config={"write_mdp": not dry, "system_name": system_name},
            simparams=simparams,
        )
        if result.get("status") != "ok":
            raise click.ClickException(result.get("error", "Martini 3 export failed"))
        source = runner.step_dir("export")
        output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, output, dirs_exist_ok=True)
    click.echo(f"Martini 3 package written to {output}")


@main.command()
@click.option(
    "--pdb", "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to PDB file",
)
def info(pdb: str):
    """Print information about a PDB file."""
    from gmxbuilder.io.pdb import PDBParser

    parser = PDBParser()
    structure = parser.parse(pdb)

    click.echo(f"File: {pdb}")
    click.echo(f"Atoms: {structure.num_atoms}")
    click.echo(f"Box: {structure.dimensions()}")

    # Residue summary
    from collections import Counter
    res_counts = Counter(structure.resnames)
    click.echo(f"Residues: {len(res_counts)} unique types")
    for res, count in res_counts.most_common(10):
        click.echo(f"  {res}: {count}")


@main.command()
def list_lipids():
    """List available lipid types."""
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    click.echo("Available lipids:")
    for name in LipidRegistry.list():
        try:
            lt = LipidRegistry.get(name)
            click.echo(
                f"  {lt.name:<6s}  "
                f"{lt.common_name:<60s}  "
                f"APL={lt.area_per_lipid:.3f} nm²  "
                f"charge={lt.charge:+d}"
            )
        except KeyError:
            click.echo(f"  {name:<6s}  [WARNING: registry inconsistency — skipping]", err=True)


@main.command()
def list_water():
    """List available water models."""
    from gmxbuilder.modules.solvation.water_models import WaterRegistry

    click.echo("Available water models:")
    for name in WaterRegistry.list():
        wm = WaterRegistry.get(name)
        click.echo(f"  {wm.name:<8s}  {wm.full_name:<8s}  atoms={wm.n_atoms}  density={wm.default_density}")


@main.command()
def list_ff():
    """List available force fields."""
    from gmxbuilder.modules.forcefield.registry import ForceFieldRegistry

    click.echo("Available force fields:")
    for name in ForceFieldRegistry.list():
        ff = ForceFieldRegistry.get(name)
        click.echo(f"  {ff.name:<12s}  version={ff.version}  water={ff.water_model}")


@main.command()
def list_modules():
    """List all available pipeline modules."""
    from gmxbuilder.modules import discover_modules

    modules = discover_modules()
    click.echo("Available pipeline modules:")
    for name, cls in modules.items():
        click.echo(f"  {name:<20s}  {cls.description}")


@main.group("prebuilt-assets")
def prebuilt_assets():
    """Inspect or install release-bundled lipid assets."""


@prebuilt_assets.command("status")
def prebuilt_assets_status():
    """Report whether validated lipid assets are installed."""
    from gmxbuilder.runtime.prebuilt_assets import prebuilt_asset_status

    status = prebuilt_asset_status()
    click.echo(f"Status: {status['status']}")
    click.echo(f"Asset version: {status['asset_version']}")
    click.echo(f"Archive: {status['archive']} ({status['archive_bytes']} bytes)")
    click.echo(
        "Contents: "
        f"{status['contents']['strict_library_entries']} strict libraries, "
        f"{status['contents']['gaff2_cache_entries']} GAFF2 caches"
    )


@prebuilt_assets.command("install")
def prebuilt_assets_install():
    """Verify and install bundled assets into writable user caches."""
    from gmxbuilder.runtime.prebuilt_assets import install_prebuilt_assets

    try:
        result = install_prebuilt_assets()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"{result['status']}: asset v{result['asset_version']}; "
        f"{result['installed_files']} files installed"
    )
    click.echo(f"Lipid library: {result['lipid_root']}")
    click.echo(f"GAFF2 cache: {result['gaff_root']}")


@main.group("lipid-library")
def lipid_library():
    """Inspect or build the force-field-specific Step 5 conformer library."""


@lipid_library.command("status")
@click.option("--force-field", multiple=True, help="Limit coverage to a force field")
def lipid_library_status(force_field: tuple[str, ...]):
    """Print offline coverage; this command never generates coordinates."""
    from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary

    library = EquilibratedLipidLibrary()
    jobs = library.coverage(list(force_field) or None)
    ready = sum(bool(job["ready"]) for job in jobs)
    click.echo(f"Validated entries: {ready}/{len(jobs)} compatible force-field jobs")
    for job in jobs:
        mark = "READY" if job["ready"] else "MISSING"
        click.echo(
            f"{mark:<7s} {job['parameter_family']:<18s} "
            f"{job['lipid_name']:<8s} ({job['force_field']})"
        )


@lipid_library.command("build")
@click.option("--force-field", default="charmm36m", show_default=True)
@click.option("--lipid", "lipids", multiple=True, help="Build only these lipid names")
@click.option("--npt-ps", default=1000.0, type=float, show_default=True)
@click.option("--test-mode", is_flag=True, help="Short GROMACS smoke run; output is not runtime eligible")
@click.option("--force", is_flag=True, help="Replace an existing validated entry")
def lipid_library_build(
    force_field: str,
    lipids: tuple[str, ...],
    npt_ps: float,
    test_mode: bool,
    force: bool,
):
    """Build compatible entries offline using explicit-solvent NPT."""
    from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
    from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder

    if npt_ps <= 0:
        raise click.BadParameter("--npt-ps must be positive")
    library = EquilibratedLipidLibrary()
    requested = {name.strip().upper() for name in lipids}
    jobs = library.coverage([force_field])
    if requested:
        jobs = [job for job in jobs if job["lipid_name"] in requested]
        missing = requested - {job["lipid_name"] for job in jobs}
        if missing:
            raise click.ClickException(
                "Incompatible or unknown lipid(s): " + ", ".join(sorted(missing))
            )
    if not jobs:
        raise click.ClickException("No compatible lipid-library jobs")
    builder = LipidEquilibrationBuilder(library=library)
    failures = []
    for number, job in enumerate(jobs, 1):
        click.echo(
            f"[{number}/{len(jobs)}] {job['lipid_name']} "
            f"{job['parameter_family']}"
        )
        try:
            output = builder.build(
                job["lipid_name"],
                job["force_field"],
                job["lipid_ff"],
                npt_ps=npt_ps,
                test_mode=test_mode,
                force=force,
            )
            click.echo(f"  -> {output}")
        except Exception as exc:
            failures.append((job["lipid_name"], str(exc)))
            click.echo(f"  FAILED: {exc}", err=True)
    if failures:
        raise click.ClickException(
            f"{len(failures)} of {len(jobs)} library builds failed"
        )


@lipid_library.command("queue")
@click.option(
    "--force-field", "force_fields", multiple=True,
    default=("amber14sb", "charmm36m", "charmm36"), show_default=True,
    help="Parameter releases to continue in order",
)
@click.option("--npt-ps", default=1000.0, type=float, show_default=True)
@click.option("--log-dir", default="output/lipid-library", show_default=True)
def lipid_library_queue(force_fields: tuple[str, ...], npt_ps: float, log_dir: str):
    """Continue missing production entries using two rotating GPUs."""
    from gmxbuilder.modules.membrane.library_queue import run_library_queue

    try:
        results = run_library_queue(force_fields, npt_ps=npt_ps, log_dir=log_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not results:
        click.echo("All requested force-field libraries are already validated")
        return
    failures = []
    for job, success, path in results:
        status = "DONE" if success else "FAILED"
        click.echo(f"{status:<6s} {job['force_field']:<15s} {job['lipid_name']:<8s} {path}")
        if not success:
            failures.append(job)
    if failures:
        raise click.ClickException(f"{len(failures)} of {len(results)} queued builds failed")


@main.command()
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=7788, type=int, help="Port to listen on")
@click.option(
    "--max-builds", "-j", default=None, type=click.IntRange(min=1),
    help="Maximum concurrent compute tasks (default: up to 4 within the CPU budget)",
)
@click.option(
    "--cpu-cores", type=click.IntRange(min=1), default=None,
    help="CPU threads exposed to GMXBUILDER (default: half of available threads)",
)
@click.option(
    "--task-threads", type=click.IntRange(min=1), default=None,
    help=(
        "Maximum CPU threads per task; must divide --cpu-cores exactly "
        "(default: largest exact share for the requested concurrency)"
    ),
)
@click.option(
    "--gpu-count", type=click.IntRange(min=0), default=None,
    help="Number of GPUs exposed to GMXBUILDER (default: GPU 0 only; 0 disables GPU)",
)
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(
    host: str, port: int, reload: bool, max_builds: int | None,
    cpu_cores: int | None, task_threads: int | None, gpu_count: int | None,
):
    """Start the GMXBUILDER web interface.

    Open http://HOST:PORT in your browser to use the interactive
    step-by-step system builder.

    Builds beyond --max-builds are placed in a FIFO queue and start
    automatically when a slot frees up.  This limit also controls
    the worker thread-pool size.
    """
    import os
    from gmxbuilder.runtime.hardware import configure_runtime_resources
    from gmxbuilder.web.security import validate_server_bind

    try:
        security = validate_server_bind(host)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    configured_max_builds = os.environ.get("GMXBUILDER_MAX_BUILDS", "").strip()
    if max_builds is not None:
        requested_slots = max_builds
        slots_are_explicit = True
    elif configured_max_builds:
        try:
            requested_slots = int(configured_max_builds)
        except ValueError as exc:
            raise click.ClickException(
                "GMXBUILDER_MAX_BUILDS must be a positive integer"
            ) from exc
        if requested_slots <= 0:
            raise click.ClickException(
                "GMXBUILDER_MAX_BUILDS must be a positive integer"
            )
        slots_are_explicit = True
    else:
        requested_slots = 4
        slots_are_explicit = False

    try:
        hardware = configure_runtime_resources(
            cpu_cores=cpu_cores,
            gpu_count=gpu_count,
            task_threads=task_threads,
            target_task_slots=requested_slots,
            apply_affinity=True,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if slots_are_explicit and requested_slots > hardware.configured_task_slots:
        raise click.ClickException(
            f"Requested {requested_slots} concurrent tasks, but "
            f"{hardware.configured_cpu_cores} allocated CPU threads with "
            f"{hardware.configured_task_threads} threads per task provide only "
            f"{hardware.configured_task_slots} task slots"
        )
    effective_slots = min(requested_slots, hardware.configured_task_slots)
    os.environ["GMXBUILDER_MAX_BUILDS"] = str(effective_slots)

    import uvicorn
    from gmxbuilder.web.server import app as web_app

    click.echo(f"\n  GMXBUILDER Web Server v{__version__}")
    click.echo(f"  Listening on http://{host}:{port}")
    click.echo(
        f"  Deployment security: {security.mode}; "
        f"authentication={'enabled' if security.authentication_enabled else 'disabled'}"
    )
    click.echo(
        f"  Concurrent task slots: {effective_slots}  "
        "(env GMXBUILDER_MAX_BUILDS)"
    )
    click.echo(
        f"  CPU budget: {hardware.configured_cpu_cores}/"
        f"{hardware.detected_cpu_threads} available threads"
    )
    click.echo(
        f"  Per-task CPU limit: {hardware.configured_task_threads} threads "
        f"({hardware.configured_task_slots} exact CPU shares)"
    )
    if hardware.gmx_installed:
        click.echo(
            f"  GROMACS: {hardware.gmx_version or 'unknown version'} "
            f"({hardware.gmx_path})"
        )
    else:
        click.echo("  GROMACS: not found (GROMACS-dependent features disabled)")
    if hardware.configured_gpu_count:
        click.echo(
            f"  GPUs exposed: {hardware.configured_gpu_count} "
            f"({', '.join(hardware.configured_gpu_devices)})"
        )
    else:
        click.echo("  GPUs exposed: 0 (CPU execution)")
    for warning in hardware.warnings:
        click.echo(f"  Warning: {warning}", err=True)
    click.echo("  Press Ctrl+C to stop.\n")

    uvicorn.run(web_app, host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    main()
