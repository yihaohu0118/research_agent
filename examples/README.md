# BFCL Experiment Configs

This folder keeps the active BFCL configs small and hard to misuse. Current
focus: no-coevo experiments that build on the already-tested conservative
T/A baseline and target the weak `multi_turn_miss_func` category.

## Anchors

| Config | Role | What It Tests |
| --- | --- | --- |
| `bfcl_grpo.yaml` | Baseline | Vanilla BFCL GRPO, no TOCF adaptation, no co-evolution. |
| `bfcl_tocf_ta_conservative.yaml` | Proven T/A baseline | Keeps task/advantage adaptation, disables self-evolution memory and strategy injection. |

## Current 16-GPU Run Matrix

| Config | Role | What It Tests |
| --- | --- | --- |
| `bfcl_tocf_ta_hardcats.yaml` | HardCat-T | Raises prior pressure on `multi_turn_miss_func` and `multi_turn_long_context`. |
| `bfcl_tocf_ta_a3.yaml` | A3-Patch | Strengthens abstention-aware advantage weights for `spurious_tool_call` and `correct_abstention`. |
| `bfcl_tocf_ta_dpatch.yaml` | D-Patch | Adds a short train-time tool-affordance discipline rule; validation remains unpatched. |
| `bfcl_tocf_ta_combo.yaml` | Combined stack | Combines HardCat-T, A3-Patch, and D-Patch. |

## Base Configs

These are imported by the primary configs and should not usually be launched
directly:

- `bfcl_grpo_base.yaml`
- `bfcl_tocf_taes_base.yaml`

## Recommended Comparisons

1. `bfcl_grpo.yaml` vs `bfcl_tocf_ta_conservative.yaml`
   Measures whether conservative task/advantage feedback helps.

2. `bfcl_tocf_ta_conservative.yaml` vs the four current run-matrix configs
   Isolates how much miss-function performance improves from category
   pressure, abstention-aware optimization, and tool-affordance discipline.

3. `bfcl_tocf_ta_combo.yaml` vs each single-module config
   Tests whether the modules compose or interfere.

## Archive

Older or secondary configs are under `examples/archive/`:

- `archive/bfcl_patch_ablations/`: isolated T/A/E/S patch ablations.
- `archive/no_coevo_diagnostics/`: older full-TAES, warmup, and early-stop diagnostics.
- `archive/coevo_paused/`: coevo configs paused while the no-coevo miss-function bottleneck is studied.
- `archive/legacy/`: generic starter configs and scripts not used for BFCL coevo.
- `archive/game/`: non-BFCL game experiments.
