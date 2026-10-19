# Artifact Specifications and Paper Alignment

## Included Artifacts and Code Modules

### 1. Training Pipeline (`src/harnessadapt/training/`)
Implements the full training loop and all experimental baselines described in
Section 3, Section 4.1, Appendix A, and Appendix B:
- **Training pool and schedule (`data/training_pool_1058.json`, `data/sampling_plan.json`, `sampling.py`, `task_pool.py`)**:
  - 1,058 tasks from `R2E-Gym-Lite` matching the exact repository distribution in Table 3 (`pandas: 399`, `numpy: 163`, `orange3: 124`, `pillow: 122`, `aiohttp: 76`, `tornado: 62`, `scrapy: 51`, `datalad: 31`, `coveragepy: 24`, `pyramid: 6`).
  - Two epochs of 23 updates (`46` tasks per update, `46` updates in total, `2,116` task visits).
  - Epoch 1 (updates 1–23) assists all `1,058` tasks (`4,232` preparation rollouts); Epoch 2 (updates 24–46) assists the fixed `512`-task subset (`2,048` preparation rollouts) with regenerated skills.
- **Skill generation (`teacher.py`, `prompts/skill_*.md`)**:
  - Compares two of the four preparation rollouts per assisted task (preferring one success and one failure; two successes if all pass; two failures if all fail; or failure-gated in `Skill-Gated` when all four fail).
  - Follows Table 4 and Appendix A.2: the frozen `GLM-5.3-Flash` teacher sees only the issue and the two trajectories (with their observed test commands and outputs), without direct repository access, reference patches, or hidden tests.
  - Supports `Skill-Mechanism` (`prompts/skill_mechanism.md`, Figures 10–14, with leakage screening and up to 3 regeneration retries), `Skill-Hint` (`prompts/skill_hint.md`, Figure 9), `Skill-Bank` (`prompts/skill_bank.md`, Figure 15), and `Skill-Gated`.
- **Configurations (`recipes.py`)**:
  - Provides explicit recipes for `harnessadapt_rl`, `grpo` (8 rollouts/task, 16,928 total rollouts), `openhands_grpo`, `skill_hint`, `skill_mechanism`, `skill_bank`, `skill_gated`, `pre_evolved`, `h1_mechanism`, `frozen_h1`, `frozen_h1_mechanism`, and `frozen_h2_mechanism` (4 training rollouts/task, 8,464 training rollouts).
- **GRPO optimization (`runtime/train.sh`, `runtime/reward_processing.py`, `driver.py`, `actor.py`)**:
  - Trains `Qwen3.5-9B` with Adam (`lr = 1e-6` constant, $\beta = (0.9, 0.98)$, $\epsilon = 10^{-8}$, weight decay `0.1`), group-mean centered advantages without std normalization, PPO clip `[0.2, 0.28]`, zero KL/entropy coefficients, 64K context (`65,536` tokens), at most `60` turns, temperature `1.0`, and loss computed only on policy-generated tokens (masking skill text, hook messages, and tool outputs).
- **Benchmark preparation, runtime adaptation, and official grading (`verified.py`, `benchmark_runtime.py`, `multilingual_scorer.py`, `swepro_scorer.py`, `vendor/swebench/`)**:
  - Supports `SWE-bench Verified` (`500` tasks), `SWE-bench Multilingual` (`300` tasks, pinned revision `846e647b9f33c0b51b739d005d13d85493c9af09`, 19 multi-language log parsers in `vendor/swebench/harness/log_parsers/`, non-login `bash --noprofile --norc -c`, and in-memory `/opt/v5python` tool adaptation), and `SWE-bench Pro V2` (`642` tasks, pinned dataset revision `2d52cb3df914a3fcf80c7f66738b3a88ae37fc50` and verifier revision `66f92766bba642462d4bbe5479e83f91f9211862`, `/app -> /testbed` symlink, `--network none`, and clean-container `/tests/test.sh` replay in `swepro_scorer.py`).

### 2. Tool and Hook Evolution (`src/harnessadapt/evolution/` & `prompts/harness_proposal.md`)
Implements Section 3.2, Section 3.3, Appendix B.2, and Appendix C.8:
- **Teacher proposal and implementation (`prompts/harness_proposal.md`, Figures 17–21)**:
  - Uses Codex with `GPT-5.6-Sol` over the round's preparation trajectories, current harness, and previous proposal outcomes to diagnose recurring interaction problems, design tool/hook changes, and implement a candidate harness bundle (or return `no_change`) in up to five rounds at the start of each epoch.
  - Checks candidate format, tool schemas, module loading, byte-identity of unchanged components, and software smoke tests before rollout comparison.
- **Training-aware Pareto selection (`common.py`, `evaluate.py`, `controller.py`)**:
  - Evaluates current and candidate harnesses on `184` fresh rollouts each (`46` tasks $\times$ `4` rollouts) under the same policy, tasks, skills, and budget.
  - Accepts a candidate iff neither `pass_rate` nor `trainable_rate` (non-truncated rollouts in mixed-reward groups divided by all `184` sampled rollouts) decreases and at least one strictly increases.
  - Includes the rejected Epoch 1 Round 3 candidate components (`proposal_sources/inspect_windows.py` and `proposal_sources/inspection_checkpoint.py`) documented in Section 4.4 and Table 15.

### 3. First-Epoch Evolved Harness `H1` (`harnesses/h1/`)
Corresponds to $H_1$ in Section 3.3, Section 4.4, and Table 12 of the paper:
- **Seven added tools** (in addition to native Bash): `read_file`, `grep`, `edit_file`, `write_file`, `view_diff`, `run_check`, and `prepare_patch`.
- **Four runtime hooks**: `budget_submission`, `progress_guard`, `validation_feedback`, and `semantic_review`.

### 4. Second-Epoch Evolved Harness `H2` (`harnesses/h2/`)
Corresponds to $H_2$ ($H_T$) in Section 3.3, Section 4.4, and Table 14 of the paper:
- **Eight added tools**: retains the six unchanged tools from $H_1$, upgrades `prepare_patch` (to rank and run up to two related Python test files within 25s and flag large deletions), and adds `restore_file` (to restore one tracked file from the unchanged Git index).
- **Five runtime hooks**: `delivery_evidence` (replaces `semantic_review`), `recovery_hint` (added), `validation_feedback` (modified), `budget_submission` (modified to preserve focused testing), and `progress_guard` (unchanged).

### 5. Final Trained Policy Checkpoint `update45` (`models/update45/`)
Corresponds to the final `HarnessAdapt-RL` policy $\theta_T$ in the paper (Section 4.1, Appendix B.1, and Appendix B.4):
- Saved at update 45 (the final checkpoint on the 5-update saving schedule before training finishes at update 46).
- Exported in Hugging Face `safetensors` format (`4` shards, `~19.3 GB`) with tokenizer, configuration, and chat template.
