"""Verification dedicated to solution-phase systems."""

from gmxbuilder.modules.verify.system_verify import SystemVerificationModule


class SolutionVerificationModule(SystemVerificationModule):
    """Verify the final solution-phase system."""

    description = "Verify solution-phase coordinates and components"
