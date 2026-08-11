"""Verification for protein-free bilayers."""

from gmxbuilder.modules.verify.system_verify import SystemVerificationModule


class PureMembraneVerificationModule(SystemVerificationModule):
    """Verify bilayer geometry and optional solvent components."""

    description = "Verify pure bilayer coordinates and components"
