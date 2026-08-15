"""Independent Martini 3 solvent workflow modules."""

from gmxbuilder.modules.martini3_solvent.environment import CGEnvironmentModule
from gmxbuilder.modules.martini3_solvent.export import CGExportModule
from gmxbuilder.modules.martini3_solvent.input import CGInputModule
from gmxbuilder.modules.martini3_solvent.mapping import CGMappingModule
from gmxbuilder.modules.martini3_solvent.model import CGModelModule
from gmxbuilder.modules.martini3_solvent.solvation import CGSolvationModule
from gmxbuilder.modules.martini3_solvent.system_check import CGSystemCheckModule
from gmxbuilder.modules.martini3_solvent.topology import CGTopologyModule

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
