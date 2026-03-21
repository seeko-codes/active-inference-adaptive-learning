"""
Simulated active inference agent with ground-truth cognitive model.

This is the generative PROCESS — the "real" student whose internal states
the inference engine must recover. The agent:

1. Has ground-truth 5-state cognitive state (known to simulation, hidden from inference)
2. Receives a problem + action from a policy
3. Produces observable responses (correct/incorrect, response time, explanation)
4. Updates internal state according to learning science dynamics

The dynamics are modulated by the learner type's CognitiveParams:
- schema_formation_rate scales how fast schemas build
- forgetting_rate scales RS decay
- wm_capacity determines overload threshold
- affect dynamics determine frustration/boredom transitions

This is NOT the inference engine — it's what the inference engine is trying to model.
"""

import numpy as np
from dataclasses import dataclass, field

from domain.taxonomy import (
    SKILLS, CONFUSABLE_PAIRS, TIER_BASE_EI,
    SCHEMA_REDUCTION, effective_ei,
)
from active_inference.state_space import (
    RS_BINS, SS_BINS, WM_BINS,
    discretize_rs, discretize_ss, discretize_wm,
    rs_label, ss_label, wm_label,
)
from simulation.learner_types import CognitiveParams, LearnerType


@dataclass
class GroundTruthState:
    """
    The actual cognitive state of a simulated student.

    All values are continuous — the discretization is done by the
    inference engine, not by the ground truth.
    """
    # Per-skill states
    rs: dict = field(default_factory=dict)       # Retrievability [0, 1]
    ss: dict = field(default_factory=dict)       # Stability (days)
    schema: dict = field(default_factory=dict)   # Schema level (0/1/2) — kept discrete

    # Global states
    wm_utilization: float = 0.3    # Current WM load fraction [0, 1]
    affect: str = "engaged"        # Current affective state

    # Pairwise discriminability
    discriminability: dict = field(default_factory=dict)  # (skill_a, skill_b) -> [0, 1]

    # Tracking
    total_problems: int = 0
    problems_per_skill: dict = field(default_factory=dict)


@dataclass
class Observation:
    """
    What the inference engine can see after a problem is presented.

    This is the observation model — the mapping from hidden states to
    observable outcomes.
    """
    skill: str
    action: str
    correct: bool
    response_time_ms: float
    explanation_words: int
    explanation_quality: float   # 0-1, proxy for schema evidence in text
    confidence_report: int       # 1-5 self-report (if prompted, else 0)
    confused_with: str = ""      # If incorrect, which skill was applied instead
    tier: int = 0                # Problem tier (determines base EI)


class SimulatedAgent:
    """
    A simulated student with ground-truth cognitive dynamics.

    Usage:
        agent = SimulatedAgent(learner_type)
        obs = agent.present_problem(skill="Dist-Right", action="worked_example", tier=4)
        # obs contains all observables
        # agent.state contains ground truth (hidden from inference engine)
    """

    def __init__(
        self,
        learner_type: LearnerType,
        rng: np.random.Generator = None,
    ):
        self.learner_type = learner_type
        self.params = learner_type.params
        self.rng = rng or np.random.default_rng()

        # Initialize ground-truth state from learner type parameters
        self.state = GroundTruthState(
            rs={s: self.params.get_initial_rs(s) for s in SKILLS},
            ss={s: self.params.get_initial_ss(s) for s in SKILLS},
            schema={s: self.params.get_initial_schema(s) for s in SKILLS},
            wm_utilization=0.3,
            affect="engaged",
            discriminability={pair: 0.5 for pair in CONFUSABLE_PAIRS},
            problems_per_skill={s: 0 for s in SKILLS},
        )

        # Session history for behavioral signals
        self._recent_correct = []
        self._recent_response_times = []
        self._consecutive_errors = 0

    def present_problem(
        self,
        skill: str,
        action: str,
        tier: int = 3,
        prompt_confidence: bool = False,
        interleave_pair: tuple = None,
    ) -> Observation:
        """
        Present a problem to the agent and get observable response.

        This is the main simulation loop:
        1. Compute current WM load for this problem
        2. Determine if answer is correct (probabilistic)
        3. Generate response time and explanation
        4. Update internal state based on action + outcome

        Args:
            skill: Which skill this problem tests
            action: What pedagogical action was chosen
            tier: Problem tier (determines base EI)
            prompt_confidence: Whether to ask for confidence report
            interleave_pair: If action is interleave, which pair

        Returns:
            Observation visible to the inference engine
        """
        # --- Step 1: Compute WM load for this problem ---
        base_ei = TIER_BASE_EI.get(tier, 5)
        student_schemas = {s: self.state.schema[s] for s in SKILLS}
        relevant_skills = [skill]
        if interleave_pair:
            relevant_skills = list(interleave_pair)
        eff_ei = effective_ei(base_ei, student_schemas, relevant_skills)

        # WM load = effective EI / capacity, clamped to [0, 1]
        problem_wm = min(1.0, eff_ei / self.params.wm_capacity)

        # Affect impacts WM (anxiety consumes capacity)
        affective_load = {"frustrated": 0.2, "engaged": 0.0, "bored": -0.05}
        wm_with_affect = min(1.0, problem_wm + affective_load.get(self.state.affect, 0))

        # Update WM: exponential moving average toward current problem's load
        # Recovery rate determines how much of previous load persists
        self.state.wm_utilization = (
            self.state.wm_utilization * (1 - self.params.wm_recovery_rate) * 0.5
            + wm_with_affect * 0.5
        )
        self.state.wm_utilization = np.clip(self.state.wm_utilization, 0.05, 1.0)

        # --- Step 2: Determine correctness ---
        correct = self._determine_correct(skill, action, eff_ei)

        # Check for confusion with another skill
        confused_with = ""
        if not correct:
            confused_with = self._check_confusion(skill)

        # --- Step 3: Generate observables ---
        response_time = self._generate_response_time(eff_ei, action, correct)
        explanation_words, explanation_quality = self._generate_explanation(
            skill, action, correct
        )
        confidence = self._generate_confidence(skill, correct) if prompt_confidence else 0

        # --- Step 4: Update internal state ---
        self._update_state(skill, action, correct, eff_ei, interleave_pair)

        # Track history
        self.state.total_problems += 1
        self.state.problems_per_skill[skill] = self.state.problems_per_skill.get(skill, 0) + 1
        self._recent_correct.append(correct)
        self._recent_response_times.append(response_time)
        if len(self._recent_correct) > 20:
            self._recent_correct = self._recent_correct[-20:]
            self._recent_response_times = self._recent_response_times[-20:]

        if correct:
            self._consecutive_errors = 0
        else:
            self._consecutive_errors += 1

        return Observation(
            skill=skill,
            action=action,
            correct=correct,
            response_time_ms=response_time,
            explanation_words=explanation_words,
            explanation_quality=explanation_quality,
            confidence_report=confidence,
            confused_with=confused_with,
            tier=tier,
        )

    def _determine_correct(self, skill: str, action: str, eff_ei: float) -> bool:
        """
        Probabilistic correctness based on ground-truth state.

        P(correct) is a function of:
        - RS (can they recall it?)
        - Schema level (do they understand the structure?)
        - WM utilization (are they overloaded?)
        - Action type (worked examples → always "correct", diagnostic → neutral)
        """
        rs = self.state.rs[skill]
        schema = self.state.schema[skill]
        wm_util = self.state.wm_utilization

        # Base probability from retrievability
        p_correct = rs

        # Schema bonus: understanding helps beyond mere recall
        schema_bonus = {0: 0.0, 1: 0.15, 2: 0.3}
        p_correct += schema_bonus.get(schema, 0)

        # WM penalty: overload degrades performance
        if wm_util > 0.7:
            wm_penalty = (wm_util - 0.7) * 1.5  # Steep penalty above 70%
            p_correct -= wm_penalty

        # Action modifiers
        action_mods = {
            "worked_example": 0.4,     # Scaffolded → much easier
            "faded_example": 0.2,      # Partially scaffolded
            "reteach": 0.3,            # Re-presented material
            "reduce_load": 0.2,        # Easier problem
            "increase_challenge": -0.2, # Harder problem
            "diagnostic_probe": 0.0,   # Neutral difficulty
            "space_and_test": -0.1,    # Testing is harder (by design)
            "interleave": -0.15,       # Interleaving is demanding
        }
        p_correct += action_mods.get(action, 0)

        p_correct = np.clip(p_correct, 0.05, 0.98)
        return bool(self.rng.random() < p_correct)

    def _check_confusion(self, skill: str) -> str:
        """
        When incorrect, determine if a specific confusable skill was applied instead.

        P(confusion with B | error on A) ∝ (1 - discriminability(A, B))
        """
        confusable = []
        for pair in CONFUSABLE_PAIRS:
            if skill in pair:
                other = pair[1] if pair[0] == skill else pair[0]
                discrim = self.state.discriminability.get(pair, 0.5)
                # Reverse lookup for (B, A) ordering
                if pair not in self.state.discriminability:
                    rev = (pair[1], pair[0])
                    discrim = self.state.discriminability.get(rev, 0.5)
                confusable.append((other, 1.0 - discrim))

        if not confusable:
            return ""

        # Normalize confusion weights
        total = sum(w for _, w in confusable)
        if total < 0.01:
            return ""

        # Probability of any confusion (vs generic error)
        p_confusion = min(0.7, total / len(confusable))
        if self.rng.random() > p_confusion:
            return ""

        # Which skill was confused?
        probs = np.array([w for _, w in confusable])
        probs /= probs.sum()
        idx = self.rng.choice(len(confusable), p=probs)
        return confusable[idx][0]

    def _generate_response_time(
        self, eff_ei: float, action: str, correct: bool
    ) -> float:
        """
        Generate response time from ground-truth state.

        Response time reflects:
        - EI (more elements → slower)
        - WM load (higher load → slower)
        - Correctness (errors often faster [random guess] or slower [stuck])
        - Action (worked examples are fast, testing is slower)
        """
        base = self.params.base_response_time

        # EI scaling: more elements → more time
        ei_factor = eff_ei / 5.0  # Normalized around EI=5

        # WM load: high load → slower
        wm_factor = 1.0 + self.state.wm_utilization * 0.5

        # Action modifiers
        action_time_mods = {
            "worked_example": 0.5,      # Studying, not solving
            "faded_example": 0.8,
            "reteach": 0.6,
            "reduce_load": 0.7,
            "increase_challenge": 1.3,
            "space_and_test": 1.2,
            "interleave": 1.1,
            "diagnostic_probe": 1.0,
        }
        action_factor = action_time_mods.get(action, 1.0)

        # Correctness effect: incorrect responses are bimodal
        # (quick guess OR slow stuck), modeled as slight increase
        correct_factor = 1.0 if correct else 1.2

        mean_time = base * ei_factor * wm_factor * action_factor * correct_factor

        # Log-normal noise (response times are right-skewed)
        cv = self.params.response_time_variance
        log_std = np.sqrt(np.log(1 + cv**2))
        log_mean = np.log(mean_time) - 0.5 * log_std**2
        time = self.rng.lognormal(log_mean, log_std)

        return max(500, min(60000, time))  # Clamp to [0.5s, 60s]

    def _generate_explanation(
        self, skill: str, action: str, correct: bool
    ) -> tuple[int, float]:
        """
        Generate explanation observable (word count and quality).

        Explanation quality reflects schema level:
        - No schema → short, mechanical, may be wrong
        - Partial → moderate length, some structure
        - Full → concise but conceptually rich

        Returns:
            (word_count, quality)
        """
        schema = self.state.schema[skill]
        base_quality = self.params.explanation_quality

        # Schema drives explanation quality
        schema_quality = {0: 0.2, 1: 0.5, 2: 0.85}
        quality = (base_quality * 0.3 + schema_quality[schema] * 0.7)

        # Incorrect answers have lower quality explanations
        if not correct:
            quality *= 0.6

        # WM overload degrades explanation coherence
        if self.state.wm_utilization > 0.7:
            quality *= 0.7

        quality = np.clip(quality, 0.0, 1.0)

        # Word count: schema level affects verbosity
        # None → short (no understanding to articulate)
        # Partial → longer (trying to explain)
        # Full → moderate (concise understanding)
        word_counts = {0: 8, 1: 25, 2: 18}
        mean_words = word_counts[schema]

        # Worked examples don't need explanations
        if action == "worked_example":
            mean_words = max(5, mean_words // 2)

        words = max(3, int(self.rng.normal(mean_words, mean_words * 0.3)))

        return words, round(quality + self.rng.normal(0, 0.05), 3)

    def _generate_confidence(self, skill: str, correct: bool) -> int:
        """
        Generate self-reported confidence (1-5).

        Overconfident learners report higher than warranted.
        Anxious learners report lower than warranted.
        """
        rs = self.state.rs[skill]
        schema = self.state.schema[skill]

        # Ground-truth competence
        true_ability = rs * 0.5 + (schema / 2.0) * 0.5

        # Calibration noise
        noise = self.rng.normal(0, 0.15)

        # Learner type bias
        # Overconfident: consistently reports higher
        # Anxious: consistently reports lower
        # Others: roughly calibrated
        bias = 0.0
        if self.params.boredom_threshold > 0.35:  # Overconfident proxy
            bias = 0.2
        if self.params.frustration_threshold < 0.55:  # Anxious proxy
            bias = -0.15

        reported = true_ability + noise + bias
        return int(np.clip(round(reported * 4 + 1), 1, 5))

    def _update_state(
        self,
        skill: str,
        action: str,
        correct: bool,
        eff_ei: float,
        interleave_pair: tuple = None,
    ):
        """
        Update ground-truth cognitive state after a problem.

        This is the transition model applied to ground truth, modulated
        by the learner's CognitiveParams.
        """
        self._update_memory(skill, action, correct)
        self._update_schema(skill, action, correct)
        self._update_affect(action, correct, eff_ei)
        if interleave_pair and action == "interleave":
            self._update_discriminability(interleave_pair, correct)

    def _update_memory(self, skill: str, action: str, correct: bool):
        """Update RS and SS based on practice outcome."""
        rs = self.state.rs[skill]
        ss = self.state.ss[skill]

        if correct:
            # RS recovery: successful practice restores access
            rs_gain = {
                "space_and_test": 0.25,
                "reteach": 0.30,
                "worked_example": 0.05,
                "faded_example": 0.15,
                "interleave": 0.10,
                "increase_challenge": 0.20,
                "reduce_load": 0.15,
                "diagnostic_probe": 0.10,
            }
            rs += rs_gain.get(action, 0.1) * self.params.rs_recovery_rate
            rs = min(0.99, rs)

            # SS growth: Bjork — gain proportional to difficulty of retrieval
            # Lower RS at time of success → bigger SS gain
            difficulty_bonus = max(0, 1.0 - rs)  # More gain when RS was low
            ss_gain = {
                "space_and_test": 0.8,     # Effortful retrieval → max SS gain
                "reteach": 0.2,            # Easy re-exposure → minimal gain
                "worked_example": 0.05,
                "faded_example": 0.3,
                "interleave": 0.4,
                "increase_challenge": 0.6,
                "reduce_load": 0.1,
                "diagnostic_probe": 0.15,
            }
            ss += ss_gain.get(action, 0.2) * difficulty_bonus * self.params.ss_growth_rate
        else:
            # Incorrect: RS drops slightly (failed retrieval)
            rs *= 0.9
            # SS unchanged by failure (Bjork — failed retrieval doesn't damage SS)

        self.state.rs[skill] = np.clip(rs, 0.01, 0.99)
        self.state.ss[skill] = max(0.1, ss)

    def _update_schema(self, skill: str, action: str, correct: bool):
        """Update schema level based on action and outcome."""
        schema = self.state.schema[skill]
        if schema >= 2:
            return  # Already full

        # Schema formation probabilities by action
        schema_probs = {
            "worked_example": 0.3,     # Primary schema-building action
            "faded_example": 0.2,      # Also builds schema
            "reteach": 0.1,            # Some schema exposure
            "space_and_test": 0.05,    # Testing doesn't build schema directly
            "interleave": 0.05,
            "increase_challenge": 0.05,
            "reduce_load": 0.02,
            "diagnostic_probe": 0.02,
        }
        p_improve = schema_probs.get(action, 0.05) * self.params.schema_formation_rate

        # Correct answers while practicing contribute more
        if correct and action in ("faded_example", "space_and_test"):
            p_improve *= 1.5

        # Schema can't improve if WM is overloaded (can't form chunks under load)
        if self.state.wm_utilization > 0.8:
            p_improve *= 0.2

        if self.rng.random() < p_improve:
            self.state.schema[skill] = min(2, schema + 1)

    def _update_affect(self, action: str, correct: bool, eff_ei: float):
        """
        Update affective state based on WM load and outcomes.

        The key dynamics:
        - High WM + errors → frustrated (vicious cycle target)
        - Low WM + easy success → bored
        - Moderate challenge + success → engaged
        """
        wm = self.state.wm_utilization
        current = self.state.affect
        inertia = self.params.affect_inertia

        # Compute raw affect tendency
        if wm > self.params.frustration_threshold and not correct:
            target = "frustrated"
            strength = 0.7
        elif wm > self.params.frustration_threshold and self._consecutive_errors >= 2:
            target = "frustrated"
            strength = 0.8
        elif wm < self.params.boredom_threshold and correct:
            target = "bored"
            strength = 0.5
        elif correct and 0.3 < wm < 0.7:
            target = "engaged"
            strength = 0.6
        else:
            target = "engaged"
            strength = self.params.engagement_baseline

        # Apply inertia: current affect resists change
        if target == current:
            # Reinforce current state
            new_affect = current
        else:
            # Probability of transition = strength * (1 - inertia)
            p_transition = strength * (1 - inertia)
            if self.rng.random() < p_transition:
                new_affect = target
            else:
                new_affect = current

        # Action-specific overrides
        if action == "reduce_load" and current == "frustrated":
            # Reducing load helps frustrated students
            if self.rng.random() < 0.5:
                new_affect = "engaged"
        elif action == "increase_challenge" and current == "bored":
            # Challenge helps bored students
            if self.rng.random() < 0.4:
                new_affect = "engaged"

        self.state.affect = new_affect

    def _update_discriminability(self, pair: tuple, correct: bool):
        """Update discriminability for an interleaved pair."""
        if pair not in self.state.discriminability:
            pair = (pair[1], pair[0])
        if pair not in self.state.discriminability:
            return

        current = self.state.discriminability[pair]

        if correct:
            # Successful discrimination → improve
            gain = 0.1
            self.state.discriminability[pair] = min(0.99, current + gain)
        else:
            # Failed discrimination → slight decrease (but interleaving exposure helps)
            # Net effect is usually positive over multiple trials
            self.state.discriminability[pair] = max(0.05, current - 0.03)

    def apply_forgetting(self, hours_elapsed: float):
        """
        Apply forgetting (RS decay) between sessions.

        Uses power-law forgetting curve: RS(t) = (1 + t/S)^(-1)
        where S = stability and t = time elapsed.

        This is called between sessions to simulate the passage of time.
        """
        days = hours_elapsed / 24.0
        for skill in SKILLS:
            ss = self.state.ss[skill]
            # Power-law decay modulated by forgetting rate
            decay = (1 + days / ss) ** (-self.params.forgetting_rate)
            self.state.rs[skill] = np.clip(
                self.state.rs[skill] * decay, 0.01, 0.99
            )

    def get_state_snapshot(self) -> dict:
        """Get a readable snapshot of ground-truth state (for analysis)."""
        return {
            "total_problems": self.state.total_problems,
            "affect": self.state.affect,
            "wm_utilization": round(self.state.wm_utilization, 3),
            "skills": {
                skill: {
                    "rs": round(self.state.rs[skill], 3),
                    "rs_label": rs_label(self.state.rs[skill]),
                    "ss": round(self.state.ss[skill], 2),
                    "ss_label": ss_label(self.state.ss[skill]),
                    "schema": self.state.schema[skill],
                    "problems_seen": self.state.problems_per_skill.get(skill, 0),
                }
                for skill in SKILLS
            },
            "discriminability": {
                f"{a}-{b}": round(v, 3)
                for (a, b), v in self.state.discriminability.items()
            },
        }

    def mastery_score(self) -> float:
        """
        Overall mastery metric: weighted combination of RS, SS, and schema.

        Used as the primary outcome measure for policy comparison.
        """
        total = 0.0
        for skill in SKILLS:
            rs = self.state.rs[skill]
            ss = min(1.0, self.state.ss[skill] / 30.0)  # Normalize SS to [0, 1]
            schema = self.state.schema[skill] / 2.0

            # Mastery = RS * weight_rs + normalized_SS * weight_ss + schema * weight_schema
            skill_mastery = rs * 0.3 + ss * 0.4 + schema * 0.3
            total += skill_mastery

        return total / len(SKILLS)
