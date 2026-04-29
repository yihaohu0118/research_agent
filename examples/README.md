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

The diagnostic control uses `bfcl-dense-env` with `mode: capped` and
`partial_credit_cap: 0.0`, so failed trajectories still receive reward `0.0`.
That keeps the reward identical to sparse GRPO while exposing failure tags for
modules that need them.

## Scope

- E-Patch, S-Patch, full TAES, and synthetic task generation are out of scope
  for this active ladder.
- `task_manager.n=0` and `synthetic_data_ratio=0.0` keep generation off for the
  main configs.
