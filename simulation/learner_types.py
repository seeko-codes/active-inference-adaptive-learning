"""
Learner type definitions for normative simulation.

Each learner type is a parameter vector that defines the ground-truth
cognitive dynamics of a simulated student. These are the "known states"
that the inference engine must recover from observations.

Parameters define:
- Prior knowledge: initial RS, SS, schema levels per skill
- WM capacity: maximum elements before overload
- Learning rate: how fast schemas form, SS increases per practice
- Retention rate: how fast RS decays (FSRS-like stability growth)
- Affect dynamics: frustration/boredom thresholds, recovery rates

The space of learner types is defined both as named archetypes (for
interpretability) and as continuous parameter ranges (for Monte Carlo
sampling).
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from domain.taxonomy import SKILLS


@dataclass
class CognitiveParams:
    """
    Ground-truth cognitive parameters for a simulated learner.

    These parameters define how the learner's internal state evolves
    in response to instruction. They are NOT observable — the inference
    engine must estimate them from behavior.
    """

    # --- Prior Knowledge (per-skill initial conditions) ---
    # Maps skill -> initial values. Missing skills default to novice.
    initial_rs: dict = field(default_factory=dict)      # Initial retrievability [0, 1]
    initial_ss: dict = field(default_factory=dict)      # Initial stability (days)
    initial_schema: dict = field(default_factory=dict)  # Initial schema level (0/1/2)

    # --- Working Memory ---
    wm_capacity: float = 5.0       # Max elements before overload (Cowan's ~4±1)
    wm_recovery_rate: float = 0.3  # How fast WM recovers between problems [0, 1]

    # --- Learning Rate ---
    schema_formation_rate: float = 0.3   # P(schema improves | appropriate instruction)
    ss_growth_rate: float = 1.0          # Multiplier on SS growth from practice (1.0 = baseline)
    rs_recovery_rate: float = 1.0        # Multiplier on RS recovery from review

    # --- Retention ---
    forgetting_rate: float = 1.0         # Multiplier on RS decay (>1 = forgets faster)

    # --- Affect Dynamics ---
    frustration_threshold: float = 0.7   # WM utilization level that triggers frustration [0, 1]
    boredom_threshold: float = 0.25      # WM utilization below which boredom occurs [0, 1]
    affect_inertia: float = 0.5          # Resistance to affect change [0, 1] (high = sticky)
    engagement_baseline: float = 0.6     # Base probability of remaining engaged

    # --- Response Characteristics ---
    base_response_time: float = 8000     # Baseline response time in ms
    response_time_variance: float = 0.3  # Coefficient of variation for response times
    explanation_quality: float = 0.5     # Baseline explanation quality [0, 1]

    # --- Metacognition ---
    metacognitive_bias: float = 0.0      # Positive = overconfident, negative = underconfident

    def get_initial_rs(self, skill: str) -> float:
        return self.initial_rs.get(skill, 0.1)

    def get_initial_ss(self, skill: str) -> float:
        return self.initial_ss.get(skill, 0.5)

    def get_initial_schema(self, skill: str) -> float:
        return self.initial_schema.get(skill, 0.0)


@dataclass
class LearnerType:
    """A named learner type with cognitive parameters and description."""
    name: str
    description: str
    params: CognitiveParams
    prevalence: float = 1.0  # Relative frequency in population (for weighted analysis)


# ==========================================
# Archetypal Learner Types
# ==========================================
# These span the parameter space to cover the major failure modes
# the system must handle. Named for interpretability.

def _make_novice() -> LearnerType:
    """
    Complete novice: no prior knowledge, average WM, slow learning.

    This learner has never seen algebra. Everything is new.
    Risks: overload on high-EI problems, frustration spiral.
    The system should deploy: worked examples, scaffolding, gradual introduction.
    """
    return LearnerType(
        name="novice",
        description="No prior knowledge, average capacity, slow schema formation",
        params=CognitiveParams(
            initial_rs={s: 0.05 for s in SKILLS},
            initial_ss={s: 0.1 for s in SKILLS},
            initial_schema={s: 0.0 for s in SKILLS},
            wm_capacity=4.0,
            wm_recovery_rate=0.25,
            schema_formation_rate=0.15,
            ss_growth_rate=0.7,
            rs_recovery_rate=0.8,
            forgetting_rate=1.3,
            frustration_threshold=0.6,
            boredom_threshold=0.2,
            affect_inertia=0.4,
            engagement_baseline=0.5,
            base_response_time=12000,
            response_time_variance=0.4,
            explanation_quality=0.3,
            metacognitive_bias=0.0,
        ),
        prevalence=1.5,  # Common
    )


def _make_fast_learner() -> LearnerType:
    """
    Fast learner: no prior knowledge, but high WM and rapid schema formation.

    Picks things up quickly but starts from zero. Can handle high EI early.
    Risks: boredom if system moves too slowly, under-challenged.
    The system should: ramp difficulty quickly, minimize scaffolding time.
    """
    return LearnerType(
        name="fast_learner",
        description="No prior knowledge, high WM, rapid schema formation",
        params=CognitiveParams(
            initial_rs={s: 0.05 for s in SKILLS},
            initial_ss={s: 0.1 for s in SKILLS},
            initial_schema={s: 0.0 for s in SKILLS},
            wm_capacity=6.0,
            wm_recovery_rate=0.4,
            schema_formation_rate=0.5,
            ss_growth_rate=1.4,
            rs_recovery_rate=1.2,
            forgetting_rate=0.7,
            frustration_threshold=0.8,
            boredom_threshold=0.35,
            affect_inertia=0.3,
            engagement_baseline=0.7,
            base_response_time=5000,
            response_time_variance=0.2,
            explanation_quality=0.7,
            metacognitive_bias=0.05,
        ),
        prevalence=0.8,
    )


def _make_partial_knowledge() -> LearnerType:
    """
    Partial knowledge: knows basic operations, gaps in distribution/factoring.

    Has schemas for commutativity and associativity, but not distribution.
    This is the classic "expertise reversal" test case — interleaving and
    testing are appropriate for known skills, worked examples for unknown.
    Risks: system treats uniformly, missing the mixed state.
    """
    known = ["A-Comm", "M-Comm", "A-Assoc", "M-Assoc"]
    unknown = ["Dist-Right", "Dist-Left", "Factor", "Sub-Def", "Div-Def"]
    return LearnerType(
        name="partial_knowledge",
        description="Strong on basics, gaps in distribution/factoring",
        params=CognitiveParams(
            initial_rs={s: (0.7 if s in known else 0.1) for s in SKILLS},
            initial_ss={s: (10.0 if s in known else 0.5) for s in SKILLS},
            initial_schema={s: (1.0 if s in known else 0.0) for s in SKILLS},
            wm_capacity=5.0,
            wm_recovery_rate=0.3,
            schema_formation_rate=0.3,
            ss_growth_rate=1.0,
            rs_recovery_rate=1.0,
            forgetting_rate=1.0,
            frustration_threshold=0.7,
            boredom_threshold=0.3,
            affect_inertia=0.5,
            engagement_baseline=0.6,
            base_response_time=8000,
            response_time_variance=0.3,
            explanation_quality=0.5,
            metacognitive_bias=0.0,
        ),
        prevalence=1.5,  # Very common
    )


def _make_forgetful() -> LearnerType:
    """
    Forgetful learner: can learn in-session but rapid decay between sessions.

    High forgetting rate means SS grows slowly and RS drops fast.
    The system should: space aggressively, over-test, maximize SS growth.
    Without spacing, this learner looks competent in-session but fails next day.
    """
    return LearnerType(
        name="forgetful",
        description="Learns in-session but rapid between-session decay",
        params=CognitiveParams(
            initial_rs={s: 0.3 for s in SKILLS},
            initial_ss={s: 1.0 for s in SKILLS},
            initial_schema={s: 0.5 for s in SKILLS},
            wm_capacity=5.0,
            wm_recovery_rate=0.3,
            schema_formation_rate=0.3,
            ss_growth_rate=0.6,
            rs_recovery_rate=1.0,
            forgetting_rate=1.8,
            frustration_threshold=0.7,
            boredom_threshold=0.3,
            affect_inertia=0.5,
            engagement_baseline=0.55,
            base_response_time=9000,
            response_time_variance=0.3,
            explanation_quality=0.4,
            metacognitive_bias=0.0,
        ),
        prevalence=1.0,
    )


def _make_low_wm() -> LearnerType:
    """
    Low WM capacity: understands concepts but overloads on complex problems.

    Has partial schemas but WM capacity is ~3 elements.
    Distribution problems (EI 5-8) consistently overwhelm.
    The system should: heavy scaffolding, avoid interleaving until schemas
    compress the elements, break problems into steps.
    """
    return LearnerType(
        name="low_wm",
        description="Partial knowledge, low WM capacity, overloads on high-EI problems",
        params=CognitiveParams(
            initial_rs={s: 0.4 for s in SKILLS},
            initial_ss={s: 3.0 for s in SKILLS},
            initial_schema={s: 0.5 for s in SKILLS},
            wm_capacity=3.0,
            wm_recovery_rate=0.2,
            schema_formation_rate=0.2,
            ss_growth_rate=0.8,
            rs_recovery_rate=0.9,
            forgetting_rate=1.1,
            frustration_threshold=0.55,
            boredom_threshold=0.2,
            affect_inertia=0.6,
            engagement_baseline=0.5,
            base_response_time=14000,
            response_time_variance=0.5,
            explanation_quality=0.35,
            metacognitive_bias=-0.05,
        ),
        prevalence=1.0,
    )


def _make_anxious() -> LearnerType:
    """
    Anxious learner: adequate knowledge but affect disrupts performance.

    Anxiety consumes WM (State 5 → State 3 coupling), creating the
    vicious cycle: anxiety → less WM → more errors → more anxiety.
    The system should: prioritize affect management, reduce load preemptively,
    build confidence via easier problems before challenging.
    """
    return LearnerType(
        name="anxious",
        description="Adequate knowledge but anxiety consumes WM, frustration spiral risk",
        params=CognitiveParams(
            initial_rs={s: 0.5 for s in SKILLS},
            initial_ss={s: 5.0 for s in SKILLS},
            initial_schema={s: 0.5 for s in SKILLS},
            wm_capacity=4.5,
            wm_recovery_rate=0.15,
            schema_formation_rate=0.25,
            ss_growth_rate=0.9,
            rs_recovery_rate=0.8,
            forgetting_rate=1.2,
            frustration_threshold=0.5,       # Low threshold — gets frustrated easily
            boredom_threshold=0.15,
            affect_inertia=0.7,              # Sticky affect — hard to recover
            engagement_baseline=0.4,
            base_response_time=11000,
            response_time_variance=0.5,
            explanation_quality=0.4,
            metacognitive_bias=-0.2,
        ),
        prevalence=1.0,
    )


def _make_overconfident() -> LearnerType:
    """
    Overconfident learner: thinks they know more than they do.

    Metacognitive miscalibration: high confidence, mediocre performance.
    Won't engage with scaffolding because they think they don't need it.
    The system should: use diagnostic probes to reveal gaps, space-and-test
    to create desirable difficulty, avoid over-scaffolding (which they'll resist).
    """
    return LearnerType(
        name="overconfident",
        description="Moderate knowledge, high confidence, resists scaffolding",
        params=CognitiveParams(
            initial_rs={s: 0.4 for s in SKILLS},
            initial_ss={s: 2.0 for s in SKILLS},
            initial_schema={s: 0.5 for s in SKILLS},
            wm_capacity=5.0,
            wm_recovery_rate=0.35,
            schema_formation_rate=0.25,
            ss_growth_rate=0.9,
            rs_recovery_rate=1.0,
            forgetting_rate=1.0,
            frustration_threshold=0.75,
            boredom_threshold=0.4,           # Gets bored easily
            affect_inertia=0.4,
            engagement_baseline=0.6,
            base_response_time=5000,         # Fast but careless
            response_time_variance=0.2,
            explanation_quality=0.5,
            metacognitive_bias=0.25,
        ),
        prevalence=0.8,
    )


def _make_advanced() -> LearnerType:
    """
    Advanced learner: strong prior knowledge, needs refinement not teaching.

    Full schemas on most skills, high SS. Only needs discriminability
    training and maintenance via spacing. Tests whether the system
    correctly shifts from teaching mode to maintenance mode.
    """
    return LearnerType(
        name="advanced",
        description="Strong prior knowledge, needs discriminability training and maintenance",
        params=CognitiveParams(
            initial_rs={s: 0.8 for s in SKILLS},
            initial_ss={s: 20.0 for s in SKILLS},
            initial_schema={s: 1.0 for s in SKILLS},
            wm_capacity=6.0,
            wm_recovery_rate=0.4,
            schema_formation_rate=0.5,
            ss_growth_rate=1.2,
            rs_recovery_rate=1.3,
            forgetting_rate=0.6,
            frustration_threshold=0.85,
            boredom_threshold=0.4,
            affect_inertia=0.3,
            engagement_baseline=0.7,
            base_response_time=4000,
            response_time_variance=0.15,
            explanation_quality=0.8,
            metacognitive_bias=0.1,
        ),
        prevalence=0.5,
    )


# All archetypal types
LEARNER_ARCHETYPES = {
    "novice": _make_novice(),
    "fast_learner": _make_fast_learner(),
    "partial_knowledge": _make_partial_knowledge(),
    "forgetful": _make_forgetful(),
    "low_wm": _make_low_wm(),
    "anxious": _make_anxious(),
    "overconfident": _make_overconfident(),
    "advanced": _make_advanced(),
}


# ==========================================
# Random Sampling (for Monte Carlo)
# ==========================================

# Parameter ranges for uniform sampling
PARAM_RANGES = {
    "wm_capacity": (2.5, 7.0),
    "wm_recovery_rate": (0.1, 0.5),
    "schema_formation_rate": (0.1, 0.6),
    "ss_growth_rate": (0.4, 1.6),
    "rs_recovery_rate": (0.5, 1.5),
    "forgetting_rate": (0.5, 2.0),
    "frustration_threshold": (0.4, 0.9),
    "boredom_threshold": (0.15, 0.45),
    "affect_inertia": (0.2, 0.8),
    "engagement_baseline": (0.3, 0.8),
    "base_response_time": (3000, 15000),
    "response_time_variance": (0.1, 0.6),
    "explanation_quality": (0.2, 0.9),
}

# Prior knowledge levels for sampling
KNOWLEDGE_LEVELS = {
    "zero": {"rs": 0.05, "ss": 0.1, "schema": 0.0},
    "exposure": {"rs": 0.2, "ss": 1.0, "schema": 0.1},
    "fragile": {"rs": 0.4, "ss": 3.0, "schema": 0.5},
    "solid": {"rs": 0.7, "ss": 10.0, "schema": 0.85},
    "mastered": {"rs": 0.9, "ss": 30.0, "schema": 1.0},
}


def sample_learner_params(rng: Optional[np.random.Generator] = None) -> CognitiveParams:
    """
    Sample a random learner from empirically grounded distributions.

    Parameter distributions and correlations based on:
    - WM capacity: Cowan (2001), Normal(4.0, 1.5)
    - Math anxiety prevalence: Ashcraft (2002), ~20%
    - Learning rate–WM correlation: r ≈ 0.5
    - Forgetting rate: FSRS empirical, log-normal

    Args:
        rng: NumPy random generator (for reproducibility)

    Returns:
        CognitiveParams with empirically sampled values
    """
    if rng is None:
        rng = np.random.default_rng()

    # WM capacity: Normal(4.0, 1.5), clamp to [2, 7] — Cowan (2001)
    wm_capacity_raw = np.clip(rng.normal(4.0, 1.5), 2.0, 7.0)

    # Math anxiety: 20% prevalence — Ashcraft (2002)
    # Mechanism: anxiety consumes central executive, reducing effective WM by ~1.5 items
    has_math_anxiety = rng.random() < 0.20
    anxiety_wm_reduction = rng.uniform(0.8, 1.5) if has_math_anxiety else 0.0
    effective_wm = max(1.5, wm_capacity_raw - anxiety_wm_reduction)

    # Learning rate: correlated with WM (r ≈ 0.5)
    wm_z = (wm_capacity_raw - 4.0) / 1.5  # standardize on raw (not anxiety-reduced)
    schema_formation_rate = np.clip(rng.normal(0.3 + 0.1 * wm_z, 0.1), 0.05, 0.65)
    ss_growth_rate = np.clip(rng.normal(1.0 + 0.15 * wm_z, 0.2), 0.3, 1.8)

    # Forgetting rate: log-normal (FSRS empirical data)
    forgetting_rate = np.clip(rng.lognormal(0.0, 0.35), 0.4, 2.5)

    # Frustration threshold: inversely related to anxiety
    frustration_threshold = np.clip(
        rng.normal(0.7 - 0.2 * has_math_anxiety, 0.1), 0.35, 0.9
    )

    # Metacognitive bias: Normal(0, 0.2), positive = overconfident
    metacognitive_bias = np.clip(rng.normal(0, 0.2), -0.5, 0.5)

    # Remaining parameters: normal distributions centered on population means
    wm_recovery_rate = np.clip(rng.normal(0.3, 0.1), 0.1, 0.5)
    rs_recovery_rate = np.clip(rng.normal(1.0, 0.2), 0.5, 1.5)
    boredom_threshold = np.clip(rng.normal(0.25, 0.08), 0.1, 0.45)
    affect_inertia = np.clip(rng.normal(0.5, 0.15), 0.2, 0.8)
    engagement_baseline = np.clip(rng.normal(0.6, 0.1), 0.3, 0.8)
    base_response_time = np.clip(rng.normal(8000, 3000), 3000, 15000)
    response_time_variance = np.clip(rng.normal(0.3, 0.1), 0.1, 0.6)
    explanation_quality = np.clip(rng.normal(0.5, 0.15), 0.2, 0.9)

    # Prior knowledge per skill (skewed toward less knowledge)
    level_names = list(KNOWLEDGE_LEVELS.keys())
    level_probs = [0.3, 0.25, 0.2, 0.15, 0.1]

    initial_rs = {}
    initial_ss = {}
    initial_schema = {}

    for skill in SKILLS:
        level_name = rng.choice(level_names, p=level_probs)
        level = KNOWLEDGE_LEVELS[level_name]
        initial_rs[skill] = np.clip(level["rs"] + rng.normal(0, 0.05), 0.01, 0.99)
        initial_ss[skill] = max(0.1, level["ss"] * rng.lognormal(0, 0.3))
        initial_schema[skill] = np.clip(level["schema"] + rng.normal(0, 0.1), 0.0, 1.0)

    return CognitiveParams(
        initial_rs=initial_rs,
        initial_ss=initial_ss,
        initial_schema=initial_schema,
        wm_capacity=effective_wm,
        wm_recovery_rate=wm_recovery_rate,
        schema_formation_rate=schema_formation_rate,
        ss_growth_rate=ss_growth_rate,
        rs_recovery_rate=rs_recovery_rate,
        forgetting_rate=forgetting_rate,
        frustration_threshold=frustration_threshold,
        boredom_threshold=boredom_threshold,
        affect_inertia=affect_inertia,
        engagement_baseline=engagement_baseline,
        base_response_time=base_response_time,
        response_time_variance=response_time_variance,
        explanation_quality=explanation_quality,
        metacognitive_bias=metacognitive_bias,
    )


def sample_learner_type(
    rng: Optional[np.random.Generator] = None,
    name: Optional[str] = None,
) -> LearnerType:
    """
    Sample a random learner type.

    Args:
        rng: NumPy random generator
        name: Optional name (defaults to "sampled_N")

    Returns:
        LearnerType with randomly sampled parameters
    """
    params = sample_learner_params(rng)
    if name is None:
        name = f"sampled_{id(params) % 10000:04d}"
    return LearnerType(
        name=name,
        description="Randomly sampled learner type",
        params=params,
    )


def get_all_archetypes() -> list[LearnerType]:
    """Return all archetypal learner types."""
    return list(LEARNER_ARCHETYPES.values())


def get_archetype(name: str) -> LearnerType:
    """Get a specific archetype by name."""
    return LEARNER_ARCHETYPES[name]
