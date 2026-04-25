# BFCL Co-Evolution Experiment Configs

This folder is intentionally kept small. The configs left at the top level are
the minimal experiment matrix for testing environment/model co-evolution.

## Primary Matrix

| Config | Role | What It Tests |
| --- | --- | --- |
| `bfcl_grpo.yaml` | Baseline | Vanilla BFCL GRPO, no TOCF adaptation, no co-evolution. |
| `bfcl_tocf_taes.yaml` | TOCF without coevo | Task/advantage/experience/strategy adaptation from environment feedback. |
| `bfcl_tocf_taes_coevo.yaml` | Full idea | TOCF + coevolution, with synthetic BFCL cases materialized and GT replay-verified. |

## Base Configs

These are imported by the primary configs and should not usually be launched
directly:

- `bfcl_grpo_base.yaml`
- `bfcl_tocf_taes_base.yaml`

## Recommended Comparisons

1. `bfcl_grpo.yaml` vs `bfcl_tocf_taes.yaml`
   Measures whether feedback-driven model/environment adaptation helps.

2. `bfcl_tocf_taes.yaml` vs `bfcl_tocf_taes_coevo.yaml`
   Measures whether co-evolved, executable synthetic BFCL cases add value.

3. Inspect `experiments/bfcl_tocf_taes_coevo/bfcl_synthetic_cases`
   Confirms that coevo produced materialized synthetic BFCL cases and
   possible-answer entries rather than only runtime prompt overlays.

## Archive

Older or secondary configs are under `examples/archive/`:

- `archive/bfcl_patch_ablations/`: isolated T/A/E/S patch ablations.
- `archive/legacy/`: generic starter configs and scripts not used for BFCL coevo.
- `archive/game/`: non-BFCL game experiments.
