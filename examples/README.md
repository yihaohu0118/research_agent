# BFCL Co-Evolution Experiment Configs

This folder keeps the primary experiment matrix plus a small set of targeted
stabilization configs for testing environment/model co-evolution.

## Primary Matrix

| Config | Role | What It Tests |
| --- | --- | --- |
| `bfcl_grpo.yaml` | Baseline | Vanilla BFCL GRPO, no TOCF adaptation, no co-evolution. |
| `bfcl_tocf_taes.yaml` | TOCF without coevo | Task/advantage/experience/strategy adaptation from environment feedback. |
| `bfcl_tocf_taes_coevo.yaml` | Full idea | TOCF + coevolution, with synthetic BFCL cases materialized and GT replay-verified. |
| `bfcl_tocf_taes_coevo_unverified.yaml` | Coevo ablation | Same synthetic generation loop, but without semantic alignment or GT replay gates. |

## Stabilization Configs

| Config | Role | What It Tests |
| --- | --- | --- |
| `bfcl_tocf_ta_conservative.yaml` | Conservative TOCF | Keeps task/advantage adaptation, disables self-evolution memory and strategy injection. |
| `bfcl_tocf_taes_warmup.yaml` | Warmup TOCF | Delays and softens adaptive sampling/advantage pressure to reduce early noisy feedback. |
| `bfcl_tocf_taes_earlystop.yaml` | Anti-collapse run | Caps training near the observed high-performing region and keeps best-checkpoint selection. |
| `bfcl_tocf_taes_coevo_smallmix.yaml` | Small-mix coevo | Uses verified synthetic BFCL tasks with a smaller generation budget and 5% mix ratio. |

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

3. `bfcl_tocf_taes_coevo_unverified.yaml` vs `bfcl_tocf_taes_coevo.yaml`
   Measures whether environment-grounded verification is better than simply
   adding synthetic BFCL tasks.

4. Inspect `experiments/bfcl_tocf_taes_coevo/bfcl_synthetic_cases`
   Confirms that coevo produced materialized synthetic BFCL cases and
   possible-answer entries rather than only runtime prompt overlays.

5. If `bfcl_tocf_taes.yaml` is unstable, compare the stabilization configs
   against the same baseline to isolate whether instability comes from E/S
   injection, early noisy feedback, late training collapse, or synthetic-task
   mix strength.

## Archive

Older or secondary configs are under `examples/archive/`:

- `archive/bfcl_patch_ablations/`: isolated T/A/E/S patch ablations.
- `archive/legacy/`: generic starter configs and scripts not used for BFCL coevo.
- `archive/game/`: non-BFCL game experiments.
