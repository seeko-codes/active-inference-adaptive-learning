"""
Monte Carlo simulation engine.

Runs N agents x M learner types x K policies, collects learning trajectories,
and produces the data needed for policy comparison and parameter recovery.

Policies:
1. Meta-function (rule-based action selection from current state)
2. Random (uniform random action selection)
3. FSRS-only (spacing based on RS/SS, ignoring schema/affect/WM/discriminability)
4. Fixed curriculum (textbook order: tier 0 → tier 7, blocked practice)

Each simulation runs a session of problems. Between sessions, forgetting is applied.
"""

import numpy as np
from dataclasses import dataclass, field

from domain.taxonomy import SKILLS, TIER_BASE_EI, CONFUSABLE_PAIRS, effective_ei
from active_inference.state_space import (
    rs_label, ss_label, wm_label, discretize_rs, discretize_ss,
    DISCRIM_BINS,
)
from active_inference.transition_model import ACTIONS
from meta_function import select_action
from simulation.simulated_agent import SimulatedAgent, Observation
from simulation.learner_types import (
    LearnerType, CognitiveParams,
    get_all_archetypes, sample_learner_type,
)


@dataclass
class SessionResult:
    """Results from one session of problems."""
    observations: list = field(default_factory=list)
    mastery_before: float = 0.0
    mastery_after: float = 0.0
    mastery_gain: float = 0.0
    n_correct: int = 0
    n_total: int = 0
    affect_counts: dict = field(default_factory=dict)


@dataclass
class TrajectoryResult:
    """Full trajectory of a simulated agent across multiple sessions."""
    learner_type: str = ""
    policy: str = ""
    seed: int = 0
    sessions: list = field(default_factory=list)
    final_mastery: float = 0.0
    mastery_trajectory: list = field(default_factory=list)
    total_problems: int = 0
    final_state: dict = field(default_factory=dict)


# ==========================================
# Policies
# ==========================================

class _MetaFunctionState:
    """Tracks per-skill rolling accuracy for the meta-function policy."""

    def __init__(self):
        self._recent_correct: dict[str, list[bool]] = {s: [] for s in SKILLS}

    def record(self, skill: str, correct: bool):
        history = self._recent_correct[skill]
        history.append(correct)
        if len(history) > 5:
            history.pop(0)

    def rolling_accuracy(self, skill: str) -> float:
        history = self._recent_correct[skill]
        if not history:
            return 1.0  # No data yet — assume OK
        return sum(history) / len(history)


def _policy_meta_function(agent: SimulatedAgent, rng: np.random.Generator) -> tuple:
    """
    Meta-function policy: use the rule-based action selection.

    Reads the agent's ground-truth state (simulating perfect inference)
    to select actions. In the full system, the inference engine's estimates
    would be used instead. The gap between this (oracle) and the inference-based
    version measures inference quality.
    """
    state = agent.state

    # Find the skill that most needs attention
    best_action = None
    best_urgency = -1

    # Get rolling accuracy tracker (attached to agent for persistence)
    if not hasattr(agent, '_mf_state'):
        agent._mf_state = _MetaFunctionState()

    for skill in SKILLS:
        rs = state.rs[skill]
        ss = state.ss[skill]
        schema = state.schema[skill]
        wm = state.wm_utilization

        # Get low-discriminability pairs involving this skill
        low_discrim_pairs = []
        for pair in CONFUSABLE_PAIRS:
            if skill in pair:
                d = state.discriminability.get(pair, 0.5)
                if d < 0.4:
                    low_discrim_pairs.append((pair, d))

        # Schema adequacy for discriminability check (continuous: >= 0.33 ~ partial)
        schema_adequate = {s: state.schema.get(s, 0) >= 0.33 for s in SKILLS}

        # Determine appropriate tier based on continuous schema level
        tier = min(7, max(0, int(schema * 4 + 1)))
        base_ei = TIER_BASE_EI.get(tier, 5)
        student_schemas = {s: state.schema[s] for s in SKILLS}
        eff_ei = effective_ei(base_ei, student_schemas, [skill])

        # Continuous schema → discrete label
        schema_label = "none" if schema < 0.33 else "partial" if schema < 0.67 else "full"

        action = select_action(
            skill=skill,
            rs_label=rs_label(rs),
            ss_label=ss_label(ss),
            schema_label=schema_label,
            wm_label=wm_label(wm),
            affect=state.affect,
            effective_ei=eff_ei,
            low_discrim_pairs=low_discrim_pairs,
            schema_adequate_for=schema_adequate,
            recent_accuracy=agent._mf_state.rolling_accuracy(skill),
        )

        # Urgency heuristic: prioritize safety actions, then learning actions
        urgency_map = {
            "reduce_load": 10,
            "reteach": 8,
            "increase_challenge": 7,
            "worked_example": 6,
            "interleave": 5,
            "faded_example": 4,
            "space_and_test": 3,
            "diagnostic_probe": 2,
        }
        urgency = urgency_map.get(action.action, 1)

        # Boost urgency for skills with low RS (need review soon)
        if rs < 0.3:
            urgency += 3

        if urgency > best_urgency:
            best_urgency = urgency
            best_action = action
            best_tier = tier

    if best_action is None:
        return SKILLS[0], "faded_example", 3, None

    interleave_pair = best_action.interleave_pair
    return best_action.skill, best_action.action, best_tier, interleave_pair


def _policy_random(agent: SimulatedAgent, rng: np.random.Generator) -> tuple:
    """Random policy: uniform random skill, action, tier."""
    skill = rng.choice(SKILLS)
    action = rng.choice(ACTIONS)
    tier = int(rng.integers(0, 8))
    interleave_pair = None
    if action == "interleave":
        pairs = [p for p in CONFUSABLE_PAIRS if skill in p]
        if pairs:
            interleave_pair = pairs[int(rng.integers(0, len(pairs)))]
    return skill, action, tier, interleave_pair


def _policy_fsrs_only(agent: SimulatedAgent, rng: np.random.Generator) -> tuple:
    """
    FSRS-only policy: spacing based on RS/SS, ignoring schema/affect/WM.

    This baseline uses only memory state to decide what to practice.
    It always tests (space_and_test) when RS is low, and reteaches when
    both RS and SS are low. No scaffolding, no affect management.
    """
    state = agent.state

    # Find skill with lowest RS (most in need of review)
    min_rs_skill = min(SKILLS, key=lambda s: state.rs[s])
    rs = state.rs[min_rs_skill]
    ss = state.ss[min_rs_skill]

    if rs < 0.3 and ss < 2.0:
        action = "reteach"
    elif rs < 0.5:
        action = "space_and_test"
    else:
        # All well-remembered → practice the weakest
        action = "faded_example"

    # Tier based on how advanced the skill is (crude)
    skill_tier = {"A-Comm": 1, "M-Comm": 1, "A-Assoc": 2, "M-Assoc": 2,
                  "A-Assoc-Rev": 2, "M-Assoc-Rev": 2, "Dist-Right": 4,
                  "Dist-Left": 4, "Factor": 4, "Sub-Def": 5, "Div-Def": 5}
    tier = skill_tier.get(min_rs_skill, 3)

    return min_rs_skill, action, tier, None


def _policy_fixed_curriculum(
    agent: SimulatedAgent, rng: np.random.Generator
) -> tuple:
    """
    Fixed curriculum: textbook order, blocked practice.

    Presents skills in order (A-Comm → ... → Div-Def), spending a fixed
    number of problems per skill before moving on. No adaptation.
    """
    state = agent.state

    # Curriculum order
    curriculum = SKILLS  # Already in pedagogical order

    # Find current position: first skill not yet "mastered" (schema < 0.85)
    for skill in curriculum:
        if state.schema[skill] < 0.85 or state.rs[skill] < 0.5:
            break
    else:
        skill = curriculum[-1]  # All mastered, keep practicing last

    # Always use faded_example (textbook default)
    action = "faded_example"
    n_seen = state.problems_per_skill.get(skill, 0)

    # Start with worked examples, then fade
    if n_seen < 3:
        action = "worked_example"
    elif n_seen < 8:
        action = "faded_example"
    else:
        action = "space_and_test"

    skill_tier = {"A-Comm": 1, "M-Comm": 1, "A-Assoc": 2, "M-Assoc": 2,
                  "A-Assoc-Rev": 2, "M-Assoc-Rev": 2, "Dist-Right": 4,
                  "Dist-Left": 4, "Factor": 4, "Sub-Def": 5, "Div-Def": 5}
    tier = skill_tier.get(skill, 3)

    return skill, action, tier, None


class _ActiveInferencePolicy:
    """
    Active inference policy: uses pymdp EFE minimization.

    Instantiated per-trajectory so the agent learns across problems
    within a trajectory but resets between trajectories.
    """

    def __init__(self):
        try:
            from active_inference.pomdp import ActiveInferenceAgent
            self.ai_agent = ActiveInferenceAgent()
        except Exception as e:
            raise RuntimeError(f"Cannot instantiate ActiveInferenceAgent: {e}")
        self._last_obs = None

    def __call__(self, agent: SimulatedAgent, rng: np.random.Generator) -> tuple:
        state = agent.state

        # Find skill with lowest RS (most in need of attention)
        min_rs_skill = min(SKILLS, key=lambda s: state.rs[s])

        if self._last_obs is None:
            # First problem: use diagnostic_probe on lowest-RS skill
            skill_tier = {"A-Comm": 1, "M-Comm": 1, "A-Assoc": 2, "M-Assoc": 2,
                          "A-Assoc-Rev": 2, "M-Assoc-Rev": 2, "Dist-Right": 4,
                          "Dist-Left": 4, "Factor": 4, "Sub-Def": 5, "Div-Def": 5}
            tier = skill_tier.get(min_rs_skill, 3)
            return min_rs_skill, "diagnostic_probe", tier, None

        # Use last observation to get active inference action
        obs = self._last_obs
        schema = state.schema[obs.skill]
        tier = min(7, max(0, int(schema * 4 + 1)))
        base_ei = TIER_BASE_EI.get(tier, 5)
        student_schemas = {s: state.schema[s] for s in SKILLS}
        eff_ei = effective_ei(base_ei, student_schemas, [obs.skill])

        action_name, info = self.ai_agent.step(
            correct=obs.correct,
            response_time_ms=obs.response_time_ms,
            explanation_quality=obs.explanation_quality,
            confidence=obs.confidence_report,
            ei_value=eff_ei,
        )

        # Select skill with lowest RS as target
        skill = min_rs_skill
        schema_val = state.schema[skill]
        action_tier = min(7, max(0, int(schema_val * 4 + 1)))

        interleave_pair = None
        if action_name == "interleave":
            pairs = [p for p in CONFUSABLE_PAIRS if skill in p]
            if pairs:
                interleave_pair = pairs[int(rng.integers(0, len(pairs)))]

        return skill, action_name, action_tier, interleave_pair

    def observe(self, obs):
        """Store latest observation for next step."""
        self._last_obs = obs


POLICIES = {
    "meta_function": _policy_meta_function,
    "random": _policy_random,
    "fsrs_only": _policy_fsrs_only,
    "fixed_curriculum": _policy_fixed_curriculum,
    "active_inference": _ActiveInferencePolicy,
}


# ==========================================
# Simulation Runner
# ==========================================

def run_session(
    agent: SimulatedAgent,
    policy_fn,
    rng: np.random.Generator,
    n_problems: int = 20,
) -> SessionResult:
    """
    Run one session of n_problems using the given policy.

    Returns:
        SessionResult with observations and mastery change
    """
    mastery_before = agent.mastery_score()
    result = SessionResult(mastery_before=mastery_before)
    affect_counts = {"frustrated": 0, "engaged": 0, "bored": 0}

    for _ in range(n_problems):
        skill, action, tier, interleave_pair = policy_fn(agent, rng)

        obs = agent.present_problem(
            skill=skill,
            action=action,
            tier=tier,
            interleave_pair=interleave_pair,
            prompt_confidence=True,
        )

        # Feed observation back to stateful policies
        if hasattr(policy_fn, 'observe'):
            policy_fn.observe(obs)
        # Track rolling accuracy for meta_function policy
        if hasattr(agent, '_mf_state'):
            agent._mf_state.record(obs.skill, obs.correct)

        result.observations.append(obs)
        result.n_total += 1
        if obs.correct:
            result.n_correct += 1
        affect_counts[agent.state.affect] = affect_counts.get(agent.state.affect, 0) + 1

    result.mastery_after = agent.mastery_score()
    result.mastery_gain = result.mastery_after - result.mastery_before
    result.affect_counts = affect_counts
    return result


def run_trajectory(
    learner_type: LearnerType,
    policy_name: str,
    n_sessions: int = 5,
    problems_per_session: int = 20,
    hours_between_sessions: float = 24.0,
    seed: int = 0,
) -> TrajectoryResult:
    """
    Run a full multi-session trajectory for one learner type + policy.

    Args:
        learner_type: The learner type to simulate
        policy_name: Which policy to use ("meta_function", "random", etc.)
        n_sessions: Number of learning sessions
        problems_per_session: Problems per session
        hours_between_sessions: Time gap between sessions (for forgetting)
        seed: Random seed for reproducibility

    Returns:
        TrajectoryResult with full trajectory data
    """
    rng = np.random.default_rng(seed)
    agent = SimulatedAgent(learner_type, rng=rng)
    policy_entry = POLICIES[policy_name]
    # Class-based policies (e.g. ActiveInferencePolicy) are instantiated per-trajectory
    if isinstance(policy_entry, type):
        policy_fn = policy_entry()
    else:
        policy_fn = policy_entry

    result = TrajectoryResult(
        learner_type=learner_type.name,
        policy=policy_name,
        seed=seed,
    )
    result.mastery_trajectory.append(agent.mastery_score())

    for session_idx in range(n_sessions):
        session = run_session(agent, policy_fn, rng, problems_per_session)
        result.sessions.append(session)
        result.mastery_trajectory.append(session.mastery_after)
        result.total_problems += session.n_total

        # Apply forgetting between sessions
        if session_idx < n_sessions - 1:
            agent.apply_forgetting(hours_between_sessions)
            result.mastery_trajectory.append(agent.mastery_score())

    result.final_mastery = agent.mastery_score()
    result.final_state = agent.get_state_snapshot()
    return result


def run_monte_carlo(
    learner_types: list[LearnerType] = None,
    policy_names: list[str] = None,
    n_seeds: int = 10,
    n_sessions: int = 5,
    problems_per_session: int = 20,
    hours_between_sessions: float = 24.0,
    base_seed: int = 42,
) -> list[TrajectoryResult]:
    """
    Run Monte Carlo simulation: N seeds x M learner types x K policies.

    Args:
        learner_types: List of learner types to simulate (default: all archetypes)
        policy_names: List of policies to compare (default: all)
        n_seeds: Number of random seeds per (learner_type, policy) pair
        n_sessions: Sessions per trajectory
        problems_per_session: Problems per session
        hours_between_sessions: Gap between sessions
        base_seed: Base seed for reproducibility

    Returns:
        List of TrajectoryResult, one per (learner_type, policy, seed)
    """
    if learner_types is None:
        learner_types = get_all_archetypes()
    if policy_names is None:
        policy_names = list(POLICIES.keys())

    results = []
    total = len(learner_types) * len(policy_names) * n_seeds
    completed = 0

    for lt in learner_types:
        for policy_name in policy_names:
            for seed_offset in range(n_seeds):
                seed = base_seed + seed_offset

                trajectory = run_trajectory(
                    learner_type=lt,
                    policy_name=policy_name,
                    n_sessions=n_sessions,
                    problems_per_session=problems_per_session,
                    hours_between_sessions=hours_between_sessions,
                    seed=seed,
                )
                results.append(trajectory)
                completed += 1

                if completed % 20 == 0 or completed == total:
                    print(f"  [{completed}/{total}] {lt.name} x {policy_name}")

    return results


def run_population_monte_carlo(
    n_students: int = 500,
    policy_names: list[str] = None,
    n_sessions: int = 5,
    problems_per_session: int = 20,
    hours_between_sessions: float = 24.0,
    base_seed: int = 42,
) -> list[TrajectoryResult]:
    """
    Population-level Monte Carlo: sample n_students from empirical distributions.

    Unlike run_monte_carlo() which uses fixed archetypes, this samples
    diverse learner profiles from the empirically grounded distributions
    in sample_learner_type(). Use this for research claims requiring
    statistical stability.

    Args:
        n_students: Number of students to sample
        policy_names: Policies to compare (default: all)
        n_sessions: Sessions per trajectory
        problems_per_session: Problems per session
        hours_between_sessions: Gap between sessions
        base_seed: Base seed for reproducibility

    Returns:
        List of TrajectoryResult, one per (student, policy)
    """
    if policy_names is None:
        policy_names = list(POLICIES.keys())

    rng = np.random.default_rng(base_seed)
    results = []

    for i in range(n_students):
        lt = sample_learner_type(rng, name=f"pop_{i:04d}")
        for policy_name in policy_names:
            trajectory = run_trajectory(
                learner_type=lt,
                policy_name=policy_name,
                n_sessions=n_sessions,
                problems_per_session=problems_per_session,
                hours_between_sessions=hours_between_sessions,
                seed=base_seed + i,
            )
            results.append(trajectory)
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{n_students}] students completed")

    return results
