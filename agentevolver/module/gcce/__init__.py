"""Gap-Conditioned Co-Evolution (GCCE).

Implements Counterfactual Gap Attribution (CGA) for agent training regret.
The total per-category regret is decomposed into:
    - Environment gap   Delta_E(c) = J(E_oracle, pi_t | c) - J(E_t, pi_t | c)
    - Policy gap        Delta_pi(c) = J(E_t, pi_teacher | c) - J(E_t, pi_t | c)

These two quantities drive a *spatial routing* protocol that replaces the
EM-style alternating optimisation used by plain TOCF:
    r_E(c)  -> TOCF T-Patch sampling weights and env-side patch budget
    r_pi(c) -> PACE-style advantage reweighting on the policy side

Concretely, this package provides:
    - teacher_cache.TeacherCache           : offline teacher-policy success cache
    - oracle_probe.OracleProbe             : schedules oracle-env validation probes
    - gap_attributor.GapAttributor         : estimates Delta_E, Delta_pi per category
    - router.GCCERouter                    : turns the two deltas into r_E, r_pi
    - advantage_weighting.apply_gcce_advantage_weighting : trainer-side hook
"""

from .gap_attributor import CategoryGap, GapAttributor
from .router import GCCERouter, RouteDecision
from .teacher_cache import TeacherCache
from .oracle_probe import OracleProbe, OracleProbeResult


def __getattr__(name):
    # Lazy import so that the torch dependency of ``advantage_weighting``
    # does not force a hard import when the gcce package is used from a
    # CPU-only context (tests, notebooks, offline scripts).
    if name in {
        "apply_gcce_advantage_weighting",
        "gcce_advantage_weighting_enabled",
        "gcce_enabled",
    }:
        from . import advantage_weighting as _adv

        return getattr(_adv, name)
    raise AttributeError(name)

__all__ = [
    "CategoryGap",
    "GapAttributor",
    "GCCERouter",
    "RouteDecision",
    "TeacherCache",
    "OracleProbe",
    "OracleProbeResult",
    "apply_gcce_advantage_weighting",
    "gcce_advantage_weighting_enabled",
    "gcce_enabled",
]
