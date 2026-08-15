"""Select and verify the immutable Martini 3 model bundle."""

from __future__ import annotations

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.coarse_grained.assets import load_manifest, validate_toolchain
from gmxbuilder.pipeline.base import BaseModule, ModuleResult


class CGModelModule(BaseModule):
    name = "cg_model"
    description = "Verify Martini 3 parameters and coarse-graining tools"

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config,
            {
                "model",
                "water_model",
                "seed",
                "_task_dir",
                "_step_dir",
            },
        )
        if str(config.get("model", "martini3")).lower() != "martini3":
            raise ModuleConfigError("Only the validated Martini 3 model is available")
        if str(config.get("water_model", "W")).upper() != "W":
            raise ModuleConfigError("Initial release supports regular Martini water (W) only")
        return True

    def run(self, system, config: dict) -> ModuleResult:
        tools = validate_toolchain()
        manifest = load_manifest()
        output = system.copy()
        output.metadata.update(
            {
                "force_field": "martini3",
                "cg_bundle_id": manifest["bundle_id"],
                "cg_tool_versions": tools,
                "water_model": "W",
                "cg_citations": list(manifest["citations"]),
            }
        )
        return ModuleResult(
            True,
            output,
            [
                f"Verified {manifest['force_field']} bundle {manifest['bundle_id']}",
                "Verified force-field SHA-256 manifest",
                ", ".join(f"{name} {version}" for name, version in sorted(tools.items())),
            ],
        )
