# GCCE: Gap-Conditioned Co-Evolution for LLM Agents

> *"When an agent fails, who is responsible — the environment, or the policy? And how much?"*

GCCE is AgentEvolver's environment-policy co-evolution framework. It extends
TOCF (environment side) and PACE (policy side) with a single new primitive:

**Counterfactual Gap Attribution (CGA)** — a per-category, reference-based
decomposition of the agent's total training regret into an environment-induced
part and a policy-induced part. The two parts then drive two structurally
different updates through a **gap-conditioned router**.

---

## 1. The identifiable decomposition

For every TOCF category `c` (BFCL: `multi_turn_base / miss_param / miss_func /
long_context`) we estimate two reference quantities:

```
Delta_E(c)  = J(E*, pi_t | c)      - J(E_t, pi_t | c)     # oracle env, current policy
Delta_pi(c) = J(E_t, pi_teacher|c) - J(E_t, pi_t | c)     # current env, teacher policy
```

Under the mild assumptions (A1–A3) stated in the accompanying paper, the
total regret obeys

```
Regret(E_t, pi_t) ≤ Σ_c p(c) · (Delta_E(c) + Delta_pi(c)) + ε
```

so minimising the two components individually bounds the total. Concretely:

* `Delta_E` is the *upper bound on how much pure environment shaping can help
  the current policy*. Estimated by running the current policy inside the
  **oracle environment E\*** — the TOCF O/C/F patches pushed to their maximal
  setting.
* `Delta_pi` is the *upper bound on how much a stronger policy could squeeze
  out of the current environment*. Estimated by running a **teacher policy**
  `pi_teacher` (Qwen-Max / GPT / etc.) in the current environment.

Bayesian shrinkage is applied to both estimators so small-count categories do
not dominate routing decisions.

## 2. The gap-conditioned router

For each category we derive two routing weights:

```
r_E(c)  = Delta_E(c) / (Delta_E(c) + Delta_pi(c) + η)
r_pi(c) = 1 − r_E(c)
```

These weights drive **two separate updates inside the same training round**:

| Side | Signal | Consumer |
|------|--------|----------|
| Environment | `r_E(c)` | TOCF T-Patch sampling weights via `task_manager.apply_tocf_patches` |
| Policy | `r_pi(c)` | Gap-weighted GRPO advantage via `gcce.apply_gcce_advantage_weighting` |

This replaces the EM-style alternation of plain TOCF+PACE with a **spatial
routing** protocol: different categories route their regret reduction through
different channels, in parallel, per epoch.

## 3. Code layout

```
agentevolver/module/gcce/
├── __init__.py                  # lazy-import public surface
├── teacher_cache.py             # offline teacher success cache  (Delta_pi side)
├── oracle_probe.py              # oracle-env probe scheduler     (Delta_E side)
├── gap_attributor.py            # Bayesian-shrunk (Delta_E, Delta_pi) estimator
├── router.py                    # (r_E, r_pi, patch_budget, env_weights)
└── advantage_weighting.py       # policy-side GRPO advantage reweighting hook

scripts/
├── prepare_bfcl_400_split.py    # 400/400 stratified split
└── precompute_teacher_scores.py # offline teacher cache generator (teacher | fake | fill-zero)

examples/
└── bfcl_gcce.yaml               # end-to-end GCCE training config (BFCL 400/400)
```

Trainer integration is in
`agentevolver/module/trainer/ae_ray_trainer.py`:

* `RayPPOTrainer.__init__` creates the `TeacherCache`, `OracleProbe`,
  `GapAttributor` and `GCCERouter` when `gcce.enable == true`.
* Inner training loop: `apply_gcce_advantage_weighting` (if GCCE is on) or
  `apply_curriculum_advantage_weighting` (PACE fallback) is applied after
  advantage computation.
* End-of-epoch: CGA attributes (Delta_E, Delta_pi) from the TOCF stats window,
  the router turns those into `r_E / r_pi / env_weights`, and
  `train_dataset.set_tocf_category_weights(env_weights)` writes back into the
  T-Patch sampling distribution.

## 4. Running it

### Step 0 — build the data splits (one-off)

```bash
python scripts/prepare_bfcl_400_split.py \
  --pool data/bfcl_train.parquet data/bfcl_test.parquet data/bfcl_test_600.parquet \
  --out-train data/bfcl_train_400.parquet \
  --out-test  data/bfcl_test_400.parquet \
  --train-size 400 --test-size 400 --seed 0 \
  --dump-split-json data/bfcl_400_split.json
```

Produces 100 tasks per BFCL category on each side of the split.

### Step 1 — precompute the teacher cache

For quick smoke tests (no network, no API):

```bash
python scripts/precompute_teacher_scores.py \
  --train-parquet data/bfcl_train_400.parquet \
  --output        data/teacher_scores_bfcl_400.json \
  --mode fake
```

For the real experiment, point at an OpenAI-compatible endpoint and switch
to `--mode teacher`:

```bash
OPENAI_API_KEY=sk-... python scripts/precompute_teacher_scores.py \
  --train-parquet data/bfcl_train_400.parquet \
  --output        data/teacher_scores_bfcl_400.json \
  --mode teacher \
  --teacher-model qwen-max \
  --teacher-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --env-url http://127.0.0.1:8082
```

The teacher rolls out directly against the BFCL env service, so per-task
success is on the exact same metric the trainer uses.

### Step 2 — launch the GCCE run

```bash
# start the BFCL env service in another shell
bash env_service/launch_script/bfcl.sh

# then
conda activate agentevolver
python launcher.py --conf examples/bfcl_gcce.yaml
```

## 5. Ablation matrix (recommended)

All five runs share the same `data/bfcl_train_400.parquet` /
`data/bfcl_test_400.parquet` split for fair comparison.

| Run | `tocf.enable` | `pace.advantage_weighting.enable` | `gcce.enable` | `gcce.advantage_weighting.enable` | `gcce.oracle_env.enable` |
|-----|---------------|-----------------------------------|----------------|------------------------------------|--------------------------|
| A. GRPO            | false | false | false | false | false |
| B. TOCF-only       | true  | false | false | false | false |
| C. TOCF + PACE     | true  | true  | false | false | false |
| D. GCCE (no probe) | true  | false | true  | true  | false |
| E. Full GCCE       | true  | false | true  | true  | true  |

Expected scientific contribution per row:
- B vs A  : isolates environment-side curriculum.
- C vs B  : isolates policy-side advantage reweighting driven by failure rate.
- D vs C  : **isolates the contribution of CGA-driven routing** over plain
            failure-rate routing. This is the first-order novelty of GCCE.
- E vs D  : isolates the additional information contributed by oracle-env
            probes on top of the teacher cache.

## 6. Fallback behaviour

The module is written so that failure of a GCCE component degrades cleanly
rather than crashing the run:

* Missing teacher cache -> `Delta_pi(c)` falls back to category failure rate
  (i.e. GCCE reduces to PACE on the policy side).
* Oracle probe disabled -> `Delta_E(c)` falls back to
  `oracle_success_prior - current_success` (conservative upper bound; the
  router is still well-defined).
* `gcce.enable = false` -> the trainer hits the original TOCF+PACE path.

This makes it safe to leave the `gcce` block present in every config and flip
it on per experiment.

## 7. Citation

A paper draft (title working: *"Who Failed? Attributing Agent Failures to
Environment or Policy, and Co-Evolving Both"*) is being prepared separately.
This README documents the implementation; for the formal statement of the
(A1)-(A3) assumptions and the regret decomposition inequality, see the paper
draft in `research/GCCE/paper/` (once added).
