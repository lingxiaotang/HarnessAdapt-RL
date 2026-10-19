#!/usr/bin/env bash
# Explicit opt-in build/export; this script is never invoked by training.
# After exporting, in the normal replication-package environment:
#   export HARNESSADAPT_OPENHANDS_RUNTIME=/absolute/path/to/new/runtime
#   export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
#   python -m harnessadapt.training prepare --recipe openhands_grpo --run-dir /path/to/new/run
#   python -m harnessadapt.training train --run-dir /path/to/new/run
# Single-task inference uses the same official SDK:
#   bash examples/run_task.sh --harness openhands --task-image IMAGE \
#     --issue-file ISSUE --output-dir NEW_OUTPUT --base-url http://127.0.0.1:31980/v1
# Keep/export the runtime manifest with experiment results.
# No browser, teacher or condenser is configured.
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: bash integrations/openhands/build.sh /absolute/path/to/new/runtime" >&2
  exit 2
fi
artifact_dir="$1"
if [[ "$artifact_dir" != /* || -e "$artifact_dir" ]]; then
  echo "Choose a new absolute output directory; existing artifacts are never overwritten." >&2
  exit 2
fi
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
docker build --platform linux/amd64 --output "type=local,dest=${artifact_dir}" "$source_dir"
echo "Official SDK runtime exported to ${artifact_dir}"
echo "Set HARNESSADAPT_OPENHANDS_RUNTIME to this directory before training or inference."
