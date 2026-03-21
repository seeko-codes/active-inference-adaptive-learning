"""
Transition model (B matrix) encoding learning science dynamics.

For each action, defines how each state dimension changes probabilistically.
These are factorized: each state dimension has its own transition matrix per action,
consistent with the mean-field approximation.

Dynamics are derived from:
- Bjork: SS/RS dynamics, desirable difficulty
- Sweller: Cognitive load, element interactivity, worked example effect
- Kornell & Bjork: Interleaving and discriminability
- Chen, Paas & Sweller: Spacing/interleaving boundary conditions

Quantitative calibration comes from simulation (Phase 2 of build order).
These initial values encode the qualitative learning science.
"""

import numpy as np
from active_inference.state_space import (
    RS_BINS, SS_BINS, SCHEMA_BINS, WM_BINS, AFFECT_BINS, DISCRIM_BINS,
)


# ==========================================
# Actions
# ==========================================

ACTIONS = [
    "space_and_test",     # Low RS + high SS + low EI → test for max SS gain
    "reteach",            # Low RS + low SS → re-encode, don't test
    "worked_example",     # No schema + high EI → scaffolded schema building
    "faded_example",      # Partial schema → transition toward independence
    "interleave",         # Low discriminability + schemas present → contrastive processing
    "increase_challenge", # Bored + low WM → push difficulty to re-engage
    "reduce_load",        # Frustrated + high WM → reduce difficulty, scaffold
    "diagnostic_probe",   # High uncertainty → maximize information gain
]

NUM_ACTIONS = len(ACTIONS)
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row of a matrix to sum to 1."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return matrix / row_sums


def _identity_transition(n: int) -> np.ndarray:
    """No change — state stays the same."""
    return np.eye(n)


def _shift_right(n: int, prob: float) -> np.ndarray:
    """Probabilistic shift to next higher bin with given probability."""
    T = np.eye(n) * (1 - prob)
    for i in range(n - 1):
        T[i, i + 1] = prob
    T[n - 1, n - 1] = 1.0  # Can't go higher than max
    return T


def _shift_left(n: int, prob: float) -> np.ndarray:
    """Probabilistic shift to next lower bin with given probability."""
    T = np.eye(n) * (1 - prob)
    for i in range(1, n):
        T[i, i - 1] = prob
    T[0, 0] = 1.0  # Can't go lower than min
    return T


# ==========================================
# Per-Action Transition Matrices
# ==========================================
# Each function returns a dict: state_dim -> (n × n) transition matrix
# T[i, j] = P(next_state=j | current_state=i, action)


def _transitions_space_and_test():
    """
    Space + test: The classic desirable difficulty.
    Bjork: When RS is low but SS is high, effortful retrieval maximally increases SS.
    RS recovers (successful retrieval restores access).
    SS increases (the struggle is what deepens encoding).
    Schema unchanged (testing doesn't build new schemas).
    WM decreases slightly (successful retrieval is satisfying).
    Affect tends toward engaged (productive struggle).
    """
    return {
        "rs": _shift_right(RS_BINS["n"], 0.6),       # RS recovers after successful retrieval
        "ss": _shift_right(SS_BINS["n"], 0.4),        # SS increases from effortful recall
        "schema": _identity_transition(SCHEMA_BINS["n"]),  # Testing doesn't build schemas
        "wm": _shift_left(WM_BINS["n"], 0.2),         # Slight WM relief after success
        "affect": np.array([                           # Tends toward engaged
            [0.3, 0.6, 0.1],  # frustrated → mostly engaged
            [0.05, 0.85, 0.1],  # engaged → stays engaged
            [0.05, 0.5, 0.45],  # bored → might engage
        ]),
    }


def _transitions_reteach():
    """
    Reteach: Re-encode material the student has forgotten.
    RS recovers (they see the material again).
    SS increases slightly (but less than effortful retrieval — Bjork).
    Schema may begin forming (exposure to structure).
    WM decreases (re-teaching reduces load).
    Affect moves toward engaged (relief from failure).
    """
    return {
        "rs": _shift_right(RS_BINS["n"], 0.7),        # Strong RS recovery from re-exposure
        "ss": _shift_right(SS_BINS["n"], 0.15),        # Weak SS gain (no struggle)
        "schema": _shift_right(SCHEMA_BINS["n"], 0.1), # Slight schema improvement
        "wm": _shift_left(WM_BINS["n"], 0.4),          # Re-teaching reduces load
        "affect": np.array([
            [0.15, 0.7, 0.15],  # frustrated → engaged (relief)
            [0.05, 0.75, 0.2],  # engaged → stays or bored (too easy)
            [0.05, 0.45, 0.5],  # bored → might stay bored (no challenge)
        ]),
    }


def _transitions_worked_example():
    """
    Worked example: Scaffolded schema building (Sweller).
    RS unchanged (not practicing retrieval).
    SS unchanged (no retrieval effort).
    Schema improves (this is the primary mechanism — studying structure).
    WM decreases (worked examples reduce cognitive load by design).
    Affect tends toward engaged (manageable difficulty).
    """
    return {
        "rs": _identity_transition(RS_BINS["n"]),      # No retrieval practice
        "ss": _identity_transition(SS_BINS["n"]),       # No storage strengthening
        "schema": _shift_right(SCHEMA_BINS["n"], 0.35), # Main effect: schema building
        "wm": _shift_left(WM_BINS["n"], 0.5),           # Reduces cognitive load
        "affect": np.array([
            [0.1, 0.75, 0.15],  # frustrated → engaged (scaffolding helps)
            [0.05, 0.7, 0.25],  # engaged → stays or bored
            [0.05, 0.35, 0.6],  # bored → likely stays bored (too easy)
        ]),
    }


def _transitions_faded_example():
    """
    Faded example: Transitional scaffolding.
    RS slightly increases (some retrieval required for faded parts).
    SS slightly increases (some effort).
    Schema improves moderately (completing gaps builds understanding).
    WM moderate effect (partially scaffolded).
    Affect tends toward engaged (balanced difficulty).
    """
    return {
        "rs": _shift_right(RS_BINS["n"], 0.3),         # Some retrieval practice
        "ss": _shift_right(SS_BINS["n"], 0.2),          # Moderate effort
        "schema": _shift_right(SCHEMA_BINS["n"], 0.25), # Schema development
        "wm": _shift_left(WM_BINS["n"], 0.15),          # Some load reduction
        "affect": np.array([
            [0.15, 0.7, 0.15],  # frustrated → engaged
            [0.05, 0.85, 0.1],  # engaged → stays (sweet spot)
            [0.1, 0.6, 0.3],    # bored → more likely to engage
        ]),
    }


def _transitions_interleave():
    """
    Interleave: Contrastive processing between confusable categories.
    RS unchanged for individual skills.
    SS unchanged.
    Schema unchanged (interleaving targets between-category, not within).
    WM increases (interleaving is demanding — must compare and contrast).
    Affect depends on current state (engaging if ready, overwhelming if not).
    Discriminability improves (primary effect).
    """
    return {
        "rs": _identity_transition(RS_BINS["n"]),
        "ss": _identity_transition(SS_BINS["n"]),
        "schema": _identity_transition(SCHEMA_BINS["n"]),
        "wm": _shift_right(WM_BINS["n"], 0.3),         # Increases cognitive load
        "affect": np.array([
            [0.6, 0.3, 0.1],   # frustrated → stays frustrated (demanding)
            [0.15, 0.7, 0.15], # engaged → mostly stays
            [0.1, 0.65, 0.25], # bored → tends toward engaged (new challenge)
        ]),
    }


def _transitions_increase_challenge():
    """
    Increase challenge: Push difficulty to re-engage a bored student.
    RS may decrease (harder material → more likely to fail).
    SS increases if successful (effortful = high learning gain).
    Schema slightly stressed (harder problems test schema limits).
    WM increases (harder = more demanding).
    Affect: bored → engaged (ideal), or bored → frustrated (overshoot).
    """
    return {
        "rs": _shift_left(RS_BINS["n"], 0.2),          # Harder → more failures
        "ss": _shift_right(SS_BINS["n"], 0.3),          # High effort = high gain
        "schema": _identity_transition(SCHEMA_BINS["n"]),
        "wm": _shift_right(WM_BINS["n"], 0.4),          # More demanding
        "affect": np.array([
            [0.7, 0.2, 0.1],   # frustrated → stays frustrated (too much)
            [0.1, 0.7, 0.2],   # engaged → stays or overwhelmed
            [0.15, 0.6, 0.25], # bored → tends toward engaged
        ]),
    }


def _transitions_reduce_load():
    """
    Reduce load: Break the overload→frustration→more overload cycle.
    RS slightly increases (easier material → more success).
    SS barely changes (easy practice doesn't deepen much).
    Schema unchanged (no new structure being built).
    WM decreases strongly (primary effect).
    Affect: frustrated → engaged (relief), engaged → bored (risk).
    """
    return {
        "rs": _shift_right(RS_BINS["n"], 0.3),          # Easier → more success
        "ss": _shift_right(SS_BINS["n"], 0.05),          # Minimal SS gain
        "schema": _identity_transition(SCHEMA_BINS["n"]),
        "wm": _shift_left(WM_BINS["n"], 0.6),            # Strong load reduction
        "affect": np.array([
            [0.1, 0.75, 0.15],  # frustrated → engaged (relief)
            [0.05, 0.55, 0.4],  # engaged → might get bored
            [0.05, 0.25, 0.7],  # bored → stays bored (too easy)
        ]),
    }


def _transitions_diagnostic_probe():
    """
    Diagnostic probe: Problem chosen for information gain, not learning.
    All state dimensions approximately unchanged.
    WM may slightly increase (unfamiliar problem context).
    Affect neutral to slightly frustrated (diagnostic = not optimized for student).
    """
    return {
        "rs": _identity_transition(RS_BINS["n"]),
        "ss": _identity_transition(SS_BINS["n"]),
        "schema": _identity_transition(SCHEMA_BINS["n"]),
        "wm": _shift_right(WM_BINS["n"], 0.1),          # Slight load from novelty
        "affect": np.array([
            [0.5, 0.4, 0.1],   # frustrated → might improve
            [0.1, 0.75, 0.15], # engaged → mostly stays
            [0.1, 0.5, 0.4],   # bored → some engagement from novelty
        ]),
    }


# Discriminability transition for interleaving specifically
def discrim_transition_interleave() -> np.ndarray:
    """
    How discriminability changes when two skills are interleaved.
    Primary effect of interleaving: discriminability improves.
    """
    return np.array([
        [0.4, 0.5, 0.1],   # low → mostly moderate
        [0.05, 0.5, 0.45],  # moderate → often high
        [0.0, 0.05, 0.95],  # high → stays high
    ])


def discrim_transition_default() -> np.ndarray:
    """Discriminability unchanged for non-interleaving actions."""
    return _identity_transition(DISCRIM_BINS["n"])


# ==========================================
# B Matrix Assembly
# ==========================================

_TRANSITION_BUILDERS = {
    "space_and_test": _transitions_space_and_test,
    "reteach": _transitions_reteach,
    "worked_example": _transitions_worked_example,
    "faded_example": _transitions_faded_example,
    "interleave": _transitions_interleave,
    "increase_challenge": _transitions_increase_challenge,
    "reduce_load": _transitions_reduce_load,
    "diagnostic_probe": _transitions_diagnostic_probe,
}


def get_transition_matrices(action: str) -> dict[str, np.ndarray]:
    """
    Get the factorized transition matrices for a given action.

    Returns:
        dict mapping state dimension name to (n × n) transition matrix.
        T[i, j] = P(next_state=j | current_state=i, action)
    """
    builder = _TRANSITION_BUILDERS.get(action)
    if builder is None:
        raise ValueError(f"Unknown action: {action}")
    return builder()


def build_full_b_matrices() -> dict[str, dict[str, np.ndarray]]:
    """
    Build the complete B matrix set for all actions.

    Returns:
        dict: action -> {state_dim -> transition_matrix}
    """
    return {action: get_transition_matrices(action) for action in ACTIONS}


def apply_transition(belief: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """
    Apply a transition matrix to a belief (categorical distribution).

    Args:
        belief: Current belief vector (probabilities over bins)
        transition: Transition matrix T[i,j] = P(next=j | current=i)

    Returns:
        Updated belief vector
    """
    new_belief = belief @ transition
    # Normalize to handle numerical drift
    total = new_belief.sum()
    if total > 0:
        new_belief /= total
    return new_belief
