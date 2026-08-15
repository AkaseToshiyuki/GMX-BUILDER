"""Pipeline configuration via pydantic models and YAML/JSON parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration.

    Can be constructed from a YAML file or dictionary.
    """

    modules: dict[str, dict[str, Any]] = Field(default_factory=dict)
    global_params: dict[str, Any] = Field(default_factory=dict)
    output_dir: Path = Path("./output")
    system_name: str = "system"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        with open(path) as fh:
            data = yaml.safe_load(fh)
        if data is None:
            data = {}
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from a JSON file."""
        import json

        path = Path(path)
        with open(path) as fh:
            data = json.load(fh)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False)

    def module_config(self, name: str) -> dict[str, Any] | None:
        """Return the config dict for a specific module, or None."""
        return self.modules.get(name)
