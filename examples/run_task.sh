#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PACKAGE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MSWEA_SILENT_STARTUP=1
export MSWEA_COST_TRACKING=ignore_errors
exec "${HARNESSADAPT_PYTHON:-python}" -m harnessadapt.run_task "$@"
