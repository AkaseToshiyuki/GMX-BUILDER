"""Inspection and research reports without starting a molecular dynamics run."""

from __future__ import annotations

import json

import click


@click.group("charmm-compat")
def charmm_compat_cli():
    """Inspect the independent local CHARMM-compatible backend."""


@charmm_compat_cli.command()
@click.option("--ff", type=click.Choice(["charmm36", "charmm36m"]), default="charmm36m")
def doctor(ff):
    """Report dependencies, reference identities and source hashes."""
    from gmxbuilder.modules.forcefield.charmm_compat import TEMPLATES, availability, source_manifest

    enabled, reason = availability(ff)
    result = {
        "available": enabled,
        "reason": reason,
        "templates": [
            {"residue": t.residue, "smiles": t.smiles, "name": t.label} for t in TEMPLATES
        ],
        "physical_validation": "not_evaluated",
        "official_cgenff": False,
    }
    if enabled:
        result["source_manifest"] = source_manifest(ff)
    click.echo(json.dumps(result, indent=2))


@charmm_compat_cli.command("inspect")
@click.option("--smiles", required=True)
@click.option("--ff", type=click.Choice(["charmm36", "charmm36m"]), default="charmm36m")
def inspect_chemistry(smiles, ff):
    """Explain assignment coverage and the experimental model; write no simulation files."""
    from gmxbuilder.modules.forcefield.charmm_compat import (
        CharmmCompatError,
        inspect_smiles,
        source_manifest,
        template_database,
    )
    from gmxbuilder.modules.forcefield.charmm_research import assign_research
    from gmxbuilder.modules.forcefield.rtp_parser import RTPParser

    try:
        molecule, template = inspect_smiles(smiles, allow_unknown=True)
        result = {"source_manifest": source_manifest(ff), "input_smiles": smiles}
        if template is not None:
            result.update(method="complete_template", template=template.residue)
        else:
            _, _, _, assignment = assign_research(molecule, RTPParser(template_database(ff)), ff)
            result.update(method="experimental_environment_bci", **assignment)
        result.update(physical_validation="not_evaluated", export_eligibility="not_checked")
    except CharmmCompatError as error:
        click.echo(
            json.dumps(
                {"error_code": error.code, "message": str(error), "export_eligibility": "blocked"}
            )
        )
        raise click.exceptions.Exit(1) from error
    click.echo(json.dumps(result, indent=2))
