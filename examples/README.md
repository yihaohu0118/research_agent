# BFCL Experiment Configs

This folder keeps the active BFCL configs focused on module attribution over
the original BFCL tasks. The active ladder is intentionally small: prove
whether T-Patch and A-Patch help on top of GRPO before adding other modules.

## Strict GRPO + Module Ladder

| Config | Role |
| --- | --- |
| `bfcl_grpo.yaml` | Launchable sparse GRPO baseline. |
| `bfcl_grpo_base.yaml` | Shared sparse GRPO training base. |
| `bfcl_tocf_taes_base.yaml` | Diagnostic control: GRPO reward stays binary, BFCL failure tags are collected, T/A/E/S all off. |
| `bfcl_grpo_tpatch.yaml` | GRPO + T-Patch only: adaptive task exposure. |
| `bfcl_grpo_apatch.yaml` | GRPO + A-Patch only: tag-aware advantage scaling. |
| `bfcl_grpo_ta.yaml` | GRPO + T-Patch + A-Patch together. |
| `bfcl_grpo_dense_t3rl.yaml` | GRPO + dense per-turn T3RL/EnvTuning-style reward only. |
| `bfcl_grpo_mfpatch.yaml` | GRPO + MF-Patch: EnvTuning-style missing-function transition protocol. |
| `bfcl_grpo_apatch_rerun.yaml` | A-Patch reproducibility rerun. |
| `bfcl_grpo_apatch_posonly.yaml` | A-Patch, positive advantages only. |
| `bfcl_grpo_apatch_posonly_mild.yaml` | A-Patch positive-only with softer tag weights. |
| `bfcl_grpo_ta_mild_t.yaml` | Mild T-Patch + default A-Patch. |
| `bfcl_grpo_ta_mild_t_posonly.yaml` | Mild T-Patch + positive-only A-Patch. |
| `bfcl_grpo_a_mfpatch_posonly.yaml` | MF-Patch + conservative positive-only A-Patch. |
| `bfcl_grpo_tpatch_rerun.yaml` | T-Patch reproducibility rerun. |
| `bfcl_grpo_ta_category_budget.yaml` | Simultaneous category-only T-Patch + budgeted residual A-Patch. |
| `bfcl_grpo_ta_category_budget_mild.yaml` | Mild simultaneous category-only T-Patch + budgeted residual A-Patch. |

The diagnostic control uses `bfcl-dense-env` with `mode: capped` and
`partial_credit_cap: 0.0`, so failed trajectories still receive reward `0.0`.
That keeps the reward identical to sparse GRPO while exposing failure tags for
modules that need them.

## Scope

- E-Patch, S-Patch, full TAES, and synthetic task generation are out of scope
  for this active ladder.
- `task_manager.n=0` and `synthetic_data_ratio=0.0` keep generation off for the
  main configs.

## Pair Launch Scripts

The tuning scripts run two experiments sequentially on one 4-GPU slice and
print directly to the current terminal or tmux pane:

```bash
bash scripts/launch_bfcl_ablation_pair1.sh  # GPUs 0,1,2,3
bash scripts/launch_bfcl_ablation_pair2.sh  # GPUs 4,5,6,7
bash scripts/launch_bfcl_ablation_pair3.sh  # GPUs 8,9,10,11
```

Override the GPU slice when launching from another machine:

```bash
GPU_SET=0,1,2,3 bash scripts/launch_bfcl_ablation_pair3.sh
```

For the current coupled T+A follow-up, skip the already-finished A-Patch
rerun and launch experiments 2/3/4 sequentially on one 4-GPU slice:

```bash
bash scripts/launch_bfcl_coupled_234.sh
```
