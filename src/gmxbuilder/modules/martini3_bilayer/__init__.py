"""Independent Martini 3 bilayer workflow modules."""

from gmxbuilder.modules.martini3_bilayer.environment import CGEnvironmentModule
from gmxbuilder.modules.martini3_bilayer.export import CGExportModule
from gmxbuilder.modules.martini3_bilayer.input import CGInputModule
from gmxbuilder.modules.martini3_bilayer.mapping import CGMappingModule
from gmxbuilder.modules.martini3_bilayer.model import CGModelModule
from gmxbuilder.modules.martini3_bilayer.orientation import CGOrientationModule
from gmxbuilder.modules.martini3_bilayer.solvation import CGSolvationModule
from gmxbuilder.modules.martini3_bilayer.system_check import CGSystemCheckModule
from gmxbuilder.modules.martini3_bilayer.topology import CGTopologyModule

MODULES = {
    "input": CGInputModule,
    "cg_model": CGModelModule,
    "cg_mapping": CGMappingModule,
    "cg_orientation": CGOrientationModule,
    "cg_environment": CGEnvironmentModule,
    "cg_solvation": CGSolvationModule,
    "cg_system": CGSystemCheckModule,
    "topology": CGTopologyModule,
    "export": CGExportModule,
}
