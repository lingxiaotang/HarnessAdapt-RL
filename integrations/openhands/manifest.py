"""Record the actual official distributions and dependency versions at build time."""
from importlib.metadata import distributions, version
import json
from pathlib import Path
import platform

root = Path("/opt/harnessadapt-openhands")
packages = {name: version(name) for name in ("openhands-sdk", "openhands-tools")}
if set(packages.values()) != {"1.54.0"}:
    raise RuntimeError(f"Unexpected OpenHands packages: {packages}")
manifest = {
    "backend": "openhands_sdk",
    "upstream": "https://github.com/OpenHands/software-agent-sdk/tree/v1.54.0",
    "packages": packages,
    "python": platform.python_version(),
    "architecture": platform.machine(),
    "dependencies": dict(sorted((dist.metadata["Name"], dist.version) for dist in distributions())),
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
