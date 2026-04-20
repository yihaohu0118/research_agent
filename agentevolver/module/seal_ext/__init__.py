"""SEAL-style extensions grafted onto the GCCE pipeline.

Only two components are currently wired into the trainer:

* ``CapGapExplorer`` — turns hindsight-classified capability gaps into
  targeted re-exploration tasks (§3.4.2 of the SEAL paper).
* (future) Step-level attribution and policy trajectory exploration.

The modules are intentionally lazy-imported so that ``import torch`` is
not paid unless they are actually used.
"""

from .cap_gap_explorer import CapGapExplorer, cap_gap_explore_enabled

__all__ = ["CapGapExplorer", "cap_gap_explore_enabled"]
