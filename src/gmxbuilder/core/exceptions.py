"""Custom exception hierarchy for GMXBUILDER."""


class GMXBuilderError(Exception):
    """Base exception for all GMXBUILDER errors."""

    pass


class ParseError(GMXBuilderError):
    """Failure to parse an input file."""

    pass


class ValidationError(GMXBuilderError):
    """Input validation failure."""

    pass


class ModuleError(GMXBuilderError):
    """Module execution failure."""

    pass


class ModuleConfigError(ValidationError):
    """Invalid module configuration."""

    pass


class TopologyError(GMXBuilderError):
    """Topology building failure."""

    pass


class GeometryError(GMXBuilderError):
    """Geometric operation failure."""

    pass


class ForceFieldError(GMXBuilderError):
    """Force field parameter not found or invalid."""

    pass


class OverlapError(GMXBuilderError):
    """Irresolvable atomic overlap."""

    pass


class PipelineError(GMXBuilderError):
    """Aggregates errors from multiple pipeline stages."""

    def __init__(self, stage_errors: dict[str, list[str]]):
        self.stage_errors = stage_errors
        msg_parts = []
        for stage, errors in stage_errors.items():
            msg_parts.append(f"  [{stage}]: {'; '.join(errors)}")
        super().__init__("Pipeline errors:\n" + "\n".join(msg_parts))
