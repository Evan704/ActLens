"""Hatch build hook: bundle the built frontend (frontend/dist) into wheels and sdists.

Editable installs skip it; a source checkout serves frontend/dist directly (see actlens.app.FRONTEND_DIST).
"""
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class FrontendBundleHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if version == "editable":
            return
        dist = Path(self.root) / "frontend" / "dist"
        if not (dist / "index.html").is_file():
            raise RuntimeError("frontend/dist is missing: run `npm install && npm run build` in frontend/ first")
        target = "actlens/static" if self.target_name == "wheel" else "frontend/dist"
        build_data["force_include"][str(dist)] = target
