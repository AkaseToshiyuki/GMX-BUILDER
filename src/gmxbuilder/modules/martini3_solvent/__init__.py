"""Martini 3 solution workflow.

The construction code lives in :mod:`gmxbuilder.modules.coarse_grained`; this
package supplies only the admission rules that make it the solution builder. The two
Martini workflows were once separate near-verbatim copies, which drifted far
enough that a fix applied to one was absent from the other.
"""

from __future__ import annotations

from gmxbuilder.modules.coarse_grained.environment import (
    CGEnvironmentModule as _CGEnvironmentModule,
)
from gmxbuilder.modules.coarse_grained.export import CGExportModule as _CGExportModule
from gmxbuilder.modules.coarse_grained.input import CGInputModule as _CGInputModule
from gmxbuilder.modules.coarse_grained.mapping import CGMappingModule as _CGMappingModule
from gmxbuilder.modules.coarse_grained.model import CGModelModule as _CGModelModule
from gmxbuilder.modules.coarse_grained.solvation import (
    CGSolvationModule as _CGSolvationModule,
)
from gmxbuilder.modules.coarse_grained.system_check import (
    CGSystemCheckModule as _CGSystemCheckModule,
)
from gmxbuilder.modules.coarse_grained.topology import (
    CGTopologyModule as _CGTopologyModule,
)

_LABEL = "Martini 3 Solvent Builder"


class _Workflow:
    """Admission rules shared by every module in this workflow."""

    cg_environment = "solution"
    workflow_label = _LABEL


class CGInputModule(_Workflow, _CGInputModule):
    pinned_config = {"environment": "solution", "include_protein": True}  # noqa: RUF012
    pin_refusals = {
        "environment": f"{_LABEL} cannot switch to bilayer mode",
        "include_protein": f"{_LABEL} requires an uploaded protein",
    }  # noqa: RUF012


class CGModelModule(_Workflow, _CGModelModule):
    pass


class CGMappingModule(_Workflow, _CGMappingModule):
    pass


class CGEnvironmentModule(_Workflow, _CGEnvironmentModule):
    checks_task_environment = True
    forbidden_config_keys = frozenset(
        {"upper_leaflet", "lower_leaflet", "asymmetric", "n_lipids_per_leaflet"}
    )  # noqa: RUF012
    forbidden_config_subject = "bilayer setting(s)"
    pinned_config = {"environment": "solution"}  # noqa: RUF012
    pin_refusals = {  # noqa: RUF012
        "environment": f"{_LABEL} cannot switch to bilayer mode"
    }


class CGSolvationModule(_Workflow, _CGSolvationModule):
    checks_task_environment = True
    pinned_config = {"include_solvent": True}  # noqa: RUF012
    pin_refusals = {"include_solvent": f"{_LABEL} requires Martini water"}  # noqa: RUF012


class CGSystemCheckModule(_Workflow, _CGSystemCheckModule):
    checks_task_environment = True


class CGTopologyModule(_Workflow, _CGTopologyModule):
    pass


class CGExportModule(_Workflow, _CGExportModule):
    pass


MODULES = {
    "input": CGInputModule,
    "cg_model": CGModelModule,
    "cg_mapping": CGMappingModule,
    "cg_environment": CGEnvironmentModule,
    "cg_solvation": CGSolvationModule,
    "cg_system": CGSystemCheckModule,
    "topology": CGTopologyModule,
    "export": CGExportModule,
}

__all__ = [
    "MODULES",
    "CGEnvironmentModule",
    "CGExportModule",
    "CGInputModule",
    "CGMappingModule",
    "CGModelModule",
    "CGSolvationModule",
    "CGSystemCheckModule",
    "CGTopologyModule",
]
