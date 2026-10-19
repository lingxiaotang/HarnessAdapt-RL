# Training environment setup

Run these commands from the replication package directory on Linux x86-64.
The host needs Python 3.11, Git, Docker with a running daemon, and NVIDIA
Container Toolkit. Training uses 8 visible GPUs; checkpoint conversion below
uses one. The paper does not specify a minimum GPU memory capacity or total
disk requirement, so those have not been inferred here. Allocate space for the
base model, converted reference, training checkpoints, trajectories, and the
benchmark Docker images in addition to the packaged deployment model.

The source pins were recovered from the local repositories; the original HF
base-model revision was recovered from its download metadata. They are recorded
in `integrations/training/sources.json`. The Dockerfile retains the original
Slime image digest and added runtime dependency versions. The bundled Slime
agent patch preserves the local format-error recovery changes; the launcher
also mounts the bundled Slime rollout and SGLang weight-checker replacements.
The training driver is provided by this package, not Slime's `train.py`.

## 1. Host dependencies and pinned sources

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-data.txt

export HARNESSADAPT_PACKAGE="$PWD"
export HARNESSADAPT_WORKSPACE="$PWD/outputs/workspace"
export HARNESSADAPT_BASE_MODEL="$HARNESSADAPT_WORKSPACE/models/Qwen3.5-9B"
export HARNESSADAPT_REF_MODEL="$HARNESSADAPT_WORKSPACE/models/Qwen3.5-9B_torch_dist"
export HARNESSADAPT_TRAIN_IMAGE="harnessadapt-training:20261019"
python examples/prepare_training_sources.py --workspace "$HARNESSADAPT_WORKSPACE"
```

The script checks out Slime `20c6dbe0746d76d99dbc754afcf389cfdc8ebe71`
and R2E-Gym `207c0914139b29f10e391405515d7d461ff93d57` and applies the
recorded Slime agent patch. It refuses a different existing revision; choose a
new workspace instead of resetting an existing research checkout.

Keep these exports in every shell used to prepare or launch this run. `prepare`
freezes the model paths and image name into `config.json`; changing exports
afterwards does not rewrite that configuration.

## 2. Training image and teacher CLIs

For recipes without teachers, such as `grpo`:

```bash
bash integrations/training/build.sh runtime
```

For `harnessadapt_rl` and other teacher-assisted recipes, build the `teachers`
target. It adds Node.js and the Codex/OpenCode CLIs to the same GPU stack:

```bash
# Set explicit package versions approved for your reproduction environment.
# Historical CLI package versions were not preserved in the original image.
read -r -p 'Codex CLI package version: ' CODEX_VERSION
read -r -p 'OpenCode CLI package version: ' OPENCODE_VERSION
export CODEX_VERSION OPENCODE_VERSION
bash integrations/training/build.sh teachers
```

The build requires both versions and prints the resulting Docker image ID.
`preflight --check-environment` records that ID and the installed CLI versions.
These CLI versions should be reported with reproduction results; the recovered
source/image pins do not establish historical teacher CLI versions.
Use `HARNESSADAPT_CODEX_BIN` and `HARNESSADAPT_OPENCODE_BIN` only to override
executable paths **inside the training image**. A host-only installation is
not sufficient.

For the `openhands_grpo` baseline, additionally build its separate SDK runtime:

```bash
bash integrations/openhands/build.sh "$HARNESSADAPT_WORKSPACE/openhands-runtime"
export HARNESSADAPT_OPENHANDS_RUNTIME="$HARNESSADAPT_WORKSPACE/openhands-runtime"
```

## 3. Original base model and reference conversion

Download the original base model, not `models/update45`, which is the trained
deployment checkpoint. The pinned model revision is
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`:

```bash
python - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
source = json.loads(Path('integrations/training/sources.json').read_text())['base_model']
snapshot_download(repo_id=source['repo_id'], revision=source['revision'],
                  local_dir=os.environ['HARNESSADAPT_BASE_MODEL'])
PY

docker run --rm --gpus 'device=0' --ipc host \
  -v "$HARNESSADAPT_WORKSPACE:$HARNESSADAPT_WORKSPACE" \
  -v "$HARNESSADAPT_PACKAGE:$HARNESSADAPT_PACKAGE:ro" \
  -e WORKSPACE="$HARNESSADAPT_WORKSPACE" \
  -e MODEL_DIR="$HARNESSADAPT_BASE_MODEL" \
  -e REF_MODEL_PATH="$HARNESSADAPT_REF_MODEL" \
  --entrypoint bash "$HARNESSADAPT_TRAIN_IMAGE" \
  "$HARNESSADAPT_PACKAGE/examples/convert_reference.sh"
```

The conversion refuses to overwrite an existing output directory and requires
the upstream conversion to finish with a `release` checkpoint marker. Keep the
reference fixed for the entire training run. If overriding model paths outside
the workspace, also mount those directories into this conversion container.

## 4. Teacher endpoints and credentials

The current skill teacher runs OpenCode with tools disabled, using the supplied
issue and preparation trajectories. Its configured provider must serve
`glm-5.3-flash` through an OpenAI-compatible chat-completions endpoint. Set the
base URL including `/v1` when required by your service:

```bash
read -r -p 'Skill teacher base URL: ' HARNESSADAPT_SKILL_TEACHER_URL
read -r -s -p 'Skill teacher API key: ' HARNESSADAPT_SKILL_TEACHER_KEY
export HARNESSADAPT_SKILL_TEACHER_URL HARNESSADAPT_SKILL_TEACHER_KEY
```

The harness teacher runs Codex with `gpt-5.6-sol` in the candidate workspace.
For a custom provider, `HARNESSADAPT_HARNESS_TEACHER_URL` must be a base URL
supporting the Responses protocol used by the CLI:

```bash
read -r -p 'Harness teacher base URL (empty for CLI default provider): ' HARNESSADAPT_HARNESS_TEACHER_URL
read -r -s -p 'Harness teacher API key: ' HARNESSADAPT_HARNESS_TEACHER_KEY
export HARNESSADAPT_HARNESS_TEACHER_URL HARNESSADAPT_HARNESS_TEACHER_KEY
```

The launcher forwards credential variable names to Docker and passes the values
through its process environment. Do not put credentials in `config.json` or
Docker build arguments. Teacher endpoint access and availability of the paper's
model IDs depend on your provider. Environment preflight checks installed CLIs
without making teacher requests; it does not validate provider access. Inference
and evaluation do not require teacher credentials.

## 5. Data and task images

`examples/train.sh prepare` materializes the published 1,058 task IDs from
`R2E-Gym/R2E-Gym-Lite`, using a local Arrow cache when present or downloading
through `datasets`. It writes a completion manifest. You may set
`HARNESSADAPT_TRAIN_DATASET_REVISION` for a download or pass `--source` with a
prepared JSONL. The packaged task IDs/order are fixed; the original upstream
training dataset revision was not recorded. A requested revision does not
replace an already present local Arrow cache.

Verified and Multilingual inputs are prepared automatically when evaluated.
Multilingual uses the revision recorded in `training/verified.py`. SWE-Pro
requires the upstream V2 task directory (instructions, `task.toml`, and tests):

```bash
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git "$HARNESSADAPT_WORKSPACE/swebench-pro"
git -C "$HARNESSADAPT_WORKSPACE/swebench-pro" checkout --detach 66f92766bba642462d4bbe5479e83f91f9211862
export HARNESSADAPT_SWEPRO_TASKS="$HARNESSADAPT_WORKSPACE/swebench-pro/v2/tasks"
```

The host Docker daemon needs access to task images named in the training
dataset and benchmark metadata. Authenticate with their registries if needed,
and allow sufficient time and disk for initial pulls. Docker image tags from
those datasets are not all immutable digests. To prepare a benchmark without
starting model inference:

```bash
bash examples/train.sh prepare-verified --run-dir outputs/runs/harnessadapt_rl --benchmark verified
bash examples/train.sh prepare-verified --run-dir outputs/runs/harnessadapt_rl --benchmark multilingual
bash examples/train.sh prepare-verified --run-dir outputs/runs/harnessadapt_rl --benchmark swepro
```

Inputs are separate under `RUN_DIR/evaluation/BENCHMARK/input/`. The task files
retain the internal filename `verified.jsonl` for compatibility with the
rollout adapter; each benchmark has its own file and manifest.

For Multilingual images without Python 3, a portable runtime is optional:
set `HARNESSADAPT_PORTABLE_PYTHON` or pass `--portable-python`. Supply a
self-contained Linux runtime with executable `bin/python3` and its standard
library under `lib/python*/`, not a virtualenv referring to the host's Python.
The launcher mounts it read-only in the training container and task containers;
tool helpers use `/opt/v5python`. If unset, image-native Python is used, with
no automatic lookup in previous experiment directories. Test compatibility
with the target task image before a full evaluation.

## 6. Prepare, check, and train

```bash
bash examples/train.sh prepare --recipe harnessadapt_rl --run-dir outputs/runs/harnessadapt_rl
bash examples/train.sh preflight --run-dir outputs/runs/harnessadapt_rl --check-environment
bash examples/train.sh train --run-dir outputs/runs/harnessadapt_rl --stop-after 46
```

The ordinary preflight checks the frozen data/schedule. With
`--check-environment`, it also checks the source revisions/agent patch, base
model shards, converted reference files, local Docker image, GPU-stack imports,
and recipe-required teacher CLIs. It writes `environment_preflight.json` without
credentials. This is not a GPU execution, teacher API, or benchmark-image test.
The preparation scripts and Dockerfile must still be exercised on the target
GPU host; they were not rebuilt or run end to end in the CPU review environment.

Evaluation output paths are
`RUN_DIR/evaluation/BENCHMARK/step_NNNN/run_RR/`. Use `--repetition` to choose
an independent attempt, and retain its `run.json` with the input hash and seed.
Repeated use of a nonempty attempt directory fails rather than adding duplicate
trajectories. Periodic training evaluations use Verified `run_01`; choose
`--repetition 2` for another attempt at that checkpoint. Legacy `verified/`
directories are left untouched and are not automatically reused.
