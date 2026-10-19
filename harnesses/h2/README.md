# H2 Harness (`harnesses/h2/`)

Second-epoch evolved harness $H_2$ ($H_T$, Section 3.3, Section 4.4, and
Table 14 of the paper), containing eight added tools and five runtime hooks in
addition to native Bash. `H2` is the final deployment harness paired with the
trained policy checkpoint $\theta_T$ (`models/update45/`).

- **Tools**: `read_file`, `grep`, `edit_file`, `write_file`, `view_diff`,
  `run_check`, `prepare_patch` (upgraded with automatic related-test execution
  and large-deletion detection), and `restore_file` (single-file restoration
  from the unchanged Git index).
- **Runtime hooks**: `budget_submission` (revised to preserve focused testing),
  `progress_guard`, `validation_feedback`, `delivery_evidence` (replaces
  `semantic_review`), and `recovery_hint`.

Relative to `H1`:
1. `prepare_patch` ranks tracked Python test files referencing the changed paths
   and symbols, runs up to two selected test files or modules within 25 seconds
   (`related_test_run`), and flags large deletions.
2. `delivery_evidence` summarizes observed checks since the latest tracked edit,
   related-test results, large deletions, and remaining budget after successful
   patch preparation.
3. `restore_file` and `recovery_hint` allow the policy to safely discard broken
   edits to a single tracked file when Git revert commands are blocked.

Run with this harness using `examples/run_task.sh --harness h2` (the default)
and the task arguments documented in the root `README.md`.
