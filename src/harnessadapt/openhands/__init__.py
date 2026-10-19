"""Integration with the upstream OpenHands SDK, not an agent implementation."""

BACKEND = "openhands_sdk"
VERSION = "1.54.0"
UPSTREAM = "https://github.com/OpenHands/software-agent-sdk/tree/v" + VERSION
RUNTIME_TARGET = "/opt/harnessadapt-openhands"


def provenance() -> dict:
    return {"backend": BACKEND, "upstream": UPSTREAM,
            "packages": {"openhands-sdk": VERSION, "openhands-tools": VERSION}}
