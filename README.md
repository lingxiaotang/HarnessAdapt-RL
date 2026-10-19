# HarnessAdapt-RL

Anonymous replication package for *Training the Whole Agent, Not Just the Model*.

**Model:** [Hugging Face](https://huggingface.co/anon-review-7f3a/HarnessAdapt-RL)

## Overview

[![HarnessAdapt-RL overview: rollouts, skill and harness proposals, harness selection, GRPO training, and deployment.](assets/figures/harnessadapt-overview.png)](assets/figures/harnessadapt-overview.pdf)

HarnessAdapt-RL trains a coding agent's policy while adapting its tools and runtime
hooks. It uses repair trajectories to generate task-specific skills and improve
the harness as the policy learns.

- **Skill guidance:** a teacher derives repair skills from successful and failed attempts.
- **Harness adaptation:** tools and hooks evolve to address recurring interaction problems.
- **Deployment:** the trained policy runs with the evolved harness, without task-specific skills or teacher calls.

The package includes the training and harness-evolution code, teacher prompts,
training task IDs and sampling schedule, and evolved harnesses **H1** and **H2**.

## Results

The paper reports mean pass@1 (%) across three evaluation runs. Improvements
over the mini-swe-agent + GRPO baseline are shown in percentage points (pp).

| Method | SWE-bench Verified | SWE-bench Multilingual | SWE-bench Pro | Average |
| --- | --- | --- | --- | --- |
| mini-swe-agent + GRPO | 47.8 | 24.7 | 33.3 | 35.3 |
| HarnessAdapt-RL | **57.6** | **37.3** | **39.7** | **44.9** |
| Improvement (pp) | **+9.8** | **+12.6** | **+6.4** | **+9.6** |

## Repository Structure

| Directory | Contents |
| --- | --- |
| [src/harnessadapt/](src/harnessadapt/) | Training, harness evolution, inference, and benchmark evaluation |
| [harnesses/](harnesses/) | Evolved H1 and H2 harnesses |
| [data/](data/) | Training task IDs and sampling schedule |
| [prompts/](prompts/) | Teacher prompt templates |
| [examples/](examples/) | Setup, training, and inference scripts |

## Quickstart

Use Python 3.11 on Linux. Serving requires Docker and NVIDIA GPUs; repair tasks
run in prepared task images.

```bash
pip install -r requirements.txt
bash examples/serve.sh
```

In another terminal, run the policy with H2 on a repair task:

```bash
bash examples/run_task.sh \
  --harness h2 \
  --task-image YOUR_PREPARED_TASK_IMAGE \
  --issue-file issue.txt \
  --output-dir outputs/example
```

For training and evaluation, see the [setup guide](docs/TRAINING_SETUP.md).
See [artifact notes](docs/ARTIFACT_NOTES.md) for the correspondence between the
package and the paper.
