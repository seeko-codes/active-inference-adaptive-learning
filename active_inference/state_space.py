"""
State space definition and discretization for the 5-state POMDP.

Each hidden state is discretized into bins aligned to action boundaries:
- RS (retrievability): 5 bins — action regimes from reteach to move on
- SS (stability): 4 bins — never learned to deeply established
- Schema: 3 bins — worked examples to full problems
- WM load: 3 bins — increase/apply/reduce difficulty
- Affect: 3 bins — frustrated/engaged/bored

Per-skill joint space: 5 × 4 × 3 × 3 × 3 = 540 states.
Mean-field factorization: each skill independent → 540 × 11 = 5,940 total.

Discriminability is tracked per skill-pair, not per individual skill,
so it lives outside the per-skill POMDP.
"""

import numpy as np
from domain.taxonomy import SKILLS, CONFUSABLE_PAIRS


# ==========================================
# State Dimensions and Bins
# ==========================================

# Retrievability (RS) — current recall probability
# Bin boundaries chosen to align with action transitions
RS_BINS = {
    "labels": ["very_low", "low", "moderate", "high", "very_high"],
    "boundaries": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "n": 5,
    "action_mapping": {
        "very_low":  "reteach",              # RS < 0.2: retrieval will fail → re-teach
        "low":       "desirable_difficulty",  # 0.2-0.4: struggle zone → test for max SS gain
        "moderate":  "practice",             # 0.4-0.6: productive practice
        "high":      "move_on",              # 0.6-0.8: well-remembered → deprioritize
        "very_high": "interleave_or_skip",   # > 0.8: fully accessible → interleave or skip
    },
}

# Stability (SS) — encoding depth, measured in days
# Higher stability = RS recovers faster after forgetting
SS_BINS = {
    "labels": ["very_low", "low", "moderate", "high"],
    "boundaries": [0.0, 1.0, 5.0, 30.0, float("inf")],
    "n": 4,
    "action_mapping": {
        "very_low":  "needs_initial_encoding",  # < 1 day: never really learned
        "low":       "fragile_learning",         # 1-5 days: learned but young
        "moderate":  "solid_learning",           # 5-30 days: well-established
        "high":      "deep_knowledge",           # > 30 days: deeply encoded
    },
}

# Schema level — knowledge organization
SCHEMA_BINS = {
    "labels": ["none", "partial", "full"],
    "n": 3,
    "action_mapping": {
        "none":    "worked_examples_blocked",    # No schema → full scaffolding
        "partial": "faded_examples",             # Partial → transitioning
        "full":    "full_problems_interleave",   # Full → remove scaffolding
    },
}

# Working memory load — current cognitive utilization
WM_BINS = {
    "labels": ["low", "moderate", "high"],
    "boundaries": [0.0, 0.33, 0.67, 1.0],
    "n": 3,
    "action_mapping": {
        "low":      "increase_challenge",   # Headroom → add difficulty
        "moderate": "apply_difficulty",      # Sweet spot → desirable difficulties viable
        "high":     "reduce_difficulty",     # Near capacity → reduce load
    },
}

# Affective state
AFFECT_BINS = {
    "labels": ["frustrated", "engaged", "bored"],
    "n": 3,
    "action_mapping": {
        "frustrated": "reduce_difficulty_scaffold",  # Break vicious cycle
        "engaged":    "deploy_optimal",              # Ideal → use best technique
        "bored":      "increase_challenge_test",     # Re-engage through difficulty
    },
}

# Discriminability per pair
DISCRIM_BINS = {
    "labels": ["low", "moderate", "high"],
    "boundaries": [0.0, 0.4, 0.7, 1.0],
    "n": 3,
    "action_mapping": {
        "low":      "must_interleave",    # Can't tell A from B → interleave
        "moderate": "optional_interleave", # Some confusion → interleave if time
        "high":     "no_interleave",       # Clear discrimination → skip
    },
}


# ==========================================
# Discretization Functions
# ==========================================

def discretize(value: float, boundaries: list[float]) -> int:
    """Map a continuous value to a bin index given boundary list."""
    for i in range(len(boundaries) - 1):
        if value < boundaries[i + 1]:
            return i
    return len(boundaries) - 2  # Last bin


def discretize_rs(rs: float) -> int:
    return discretize(rs, RS_BINS["boundaries"])


def discretize_ss(ss: float) -> int:
    return discretize(ss, SS_BINS["boundaries"])


def discretize_wm(wm: float) -> int:
    return discretize(wm, WM_BINS["boundaries"])


def discretize_schema(level: int) -> int:
    return min(level, 2)


def discretize_affect(affect: str) -> int:
    return AFFECT_BINS["labels"].index(affect)


def discretize_discrim(discrim: float) -> int:
    return discretize(discrim, DISCRIM_BINS["boundaries"])


def rs_label(rs: float) -> str:
    return RS_BINS["labels"][discretize_rs(rs)]


def ss_label(ss: float) -> str:
    return SS_BINS["labels"][discretize_ss(ss)]


def wm_label(wm: float) -> str:
    return WM_BINS["labels"][discretize_wm(wm)]


def discrim_label(discrim: float) -> str:
    return DISCRIM_BINS["labels"][discretize_discrim(discrim)]


# ==========================================
# Joint State Space
# ==========================================

# Per-skill dimensions
SKILL_STATE_DIMS = [RS_BINS["n"], SS_BINS["n"], SCHEMA_BINS["n"], WM_BINS["n"], AFFECT_BINS["n"]]
SKILL_STATE_SIZE = 1
for d in SKILL_STATE_DIMS:
    SKILL_STATE_SIZE *= d
# 5 × 4 × 3 × 3 × 3 = 540

# Factorized dimensions (for mean-field approximation)
FACTORIZED_DIMS = {
    "rs": RS_BINS["n"],       # 5
    "ss": SS_BINS["n"],       # 4
    "schema": SCHEMA_BINS["n"],  # 3
    "wm": WM_BINS["n"],       # 3
    "affect": AFFECT_BINS["n"],  # 3
}
FACTORIZED_TOTAL = sum(FACTORIZED_DIMS.values())  # 18 parameters per skill


def joint_index(rs_i: int, ss_i: int, schema_i: int, wm_i: int, affect_i: int) -> int:
    """Map 5 bin indices to a single joint state index (0 to 539)."""
    return (rs_i * SS_BINS["n"] * SCHEMA_BINS["n"] * WM_BINS["n"] * AFFECT_BINS["n"]
            + ss_i * SCHEMA_BINS["n"] * WM_BINS["n"] * AFFECT_BINS["n"]
            + schema_i * WM_BINS["n"] * AFFECT_BINS["n"]
            + wm_i * AFFECT_BINS["n"]
            + affect_i)


def decompose_index(idx: int) -> tuple[int, int, int, int, int]:
    """Map a joint state index back to 5 bin indices."""
    affect_i = idx % AFFECT_BINS["n"]
    idx //= AFFECT_BINS["n"]
    wm_i = idx % WM_BINS["n"]
    idx //= WM_BINS["n"]
    schema_i = idx % SCHEMA_BINS["n"]
    idx //= SCHEMA_BINS["n"]
    ss_i = idx % SS_BINS["n"]
    idx //= SS_BINS["n"]
    rs_i = idx
    return rs_i, ss_i, schema_i, wm_i, affect_i


def uniform_prior(dim: int) -> np.ndarray:
    """Uniform categorical distribution over dim bins."""
    return np.ones(dim) / dim


def init_skill_beliefs() -> dict[str, dict[str, np.ndarray]]:
    """
    Initialize factorized beliefs for all skills.
    Returns a dict: skill -> {dim_name -> probability vector}.
    All priors are uniform (maximum uncertainty).
    """
    return {
        skill: {
            dim_name: uniform_prior(dim_size)
            for dim_name, dim_size in FACTORIZED_DIMS.items()
        }
        for skill in SKILLS
    }


def init_discrim_beliefs() -> dict[tuple, np.ndarray]:
    """
    Initialize discriminability beliefs for all confusable pairs.
    Returns a dict: (skill_a, skill_b) -> probability vector over 3 bins.
    """
    return {
        pair: uniform_prior(DISCRIM_BINS["n"])
        for pair in CONFUSABLE_PAIRS
    }
