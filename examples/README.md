# BFCL Experiment Configs

This folder keeps only the active BFCL configs for the current paper path:
strict GRPO controls, verifier-diagnostic plumbing, the working advantage
reweighting module, and the working observation-interface module.

## Active Configs

| Config | Role |
| --- | --- |
| `bfcl_grpo.yaml` | Launchable sparse GRPO baseline. |
| `bfcl_grpo_base.yaml` | Shared sparse GRPO training base. |
| `bfcl_tocf_taes_base.yaml` | Diagnostic control: GRPO reward stays binary while BFCL failure tags are collected. |
| `bfcl_tocf_taes_shared.yaml` | Shared diagnostic/A-Patch config body. |
| `bfcl_grpo_apatch.yaml` | A-Patch only: diagnosis-guided advantage reweighting. |
| `bfcl_grpo_apatch_rerun.yaml` | A-Patch reproducibility rerun; current strict best baseline. |
| `bfcl_grpo_apatch_observation_required.yaml` | A-Patch + train-only required/enum observation annotations; current best environment-interface setting. |
| `bfcl_grpo_apatch_observation_required_rerun.yaml` | Reproducibility rerun for the observation-interface setting. |
| `bfcl_grpo_apatch_observation_required_nolc.yaml` | Observation-interface setting applied only outside long-context tasks. |
| `bfcl_grpo_apatch_static_fission.yaml` | A-Patch + train-only static recovery continuations after parse/tool errors. |

## Scope

- Synthetic task generation, dense T3RL reward, T-Patch, MF-Patch,
  recovery-lite variants, typed observation hints, and failed composition
  sweeps have been removed from the active config set.
- The strict setting keeps validation unchanged unless a config explicitly says
  otherwise. Current active configs are train-time only for observation changes.

## Launch

Run the observation follow-up pair sequentially on one 4-GPU slice:

```bash
bash scripts/launch_bfcl_observation_followup.sh
```

Override the GPU slice when launching from another machine:

```bash
GPU_SET=0,1,2,3 bash scripts/launch_bfcl_observation_followup.sh
```
