"""
State 2: Schema state and element interactivity inference.

Consumes LLM analyzer output to classify schema level per skill.
Maintains a rolling estimate using exponential moving average
to smooth noise across multiple LLM classifications.

Also computes effective element interactivity (EI) per problem
using predefined base EI adjusted by the student's schema state.
"""

from llm.analyzer import AnalysisResult
from domain.taxonomy import TIER_BASE_EI, SCHEMA_REDUCTION, effective_ei, SKILLS
from active_inference.state_space import SCHEMA_BINS


class SchemaTracker:
    """
    Tracks per-skill schema state for one student.

    Schema level is inferred from LLM analysis of student explanations,
    smoothed via exponential moving average to handle noisy classifications.
    """

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: EMA smoothing factor. Higher = more weight on recent observation.
                   0.3 means ~70% retention of prior estimate per update.
        """
        self.alpha = alpha
        # Per-skill: running EMA of schema level (0=none, 1=partial, 2=full)
        self._estimates: dict[str, float] = {skill: 0.0 for skill in SKILLS}
        self._observation_counts: dict[str, int] = {skill: 0 for skill in SKILLS}

    def update(self, skill: str, analysis: AnalysisResult) -> dict:
        """
        Update schema estimate for a skill based on LLM analysis.

        Args:
            skill: The skill being assessed
            analysis: AnalysisResult from the LLM analyzer

        Returns:
            Updated schema state dict
        """
        new_level = analysis.schema_level_int  # 0, 1, or 2

        if self._observation_counts[skill] == 0:
            # First observation: use directly
            self._estimates[skill] = float(new_level)
        else:
            # EMA update
            self._estimates[skill] = (
                self.alpha * new_level
                + (1 - self.alpha) * self._estimates[skill]
            )

        self._observation_counts[skill] += 1
        return self.get_state(skill)

    def get_state(self, skill: str) -> dict:
        """Get current schema state for a skill."""
        raw = self._estimates[skill]
        # Discretize: <0.67 = none, 0.67-1.33 = partial, >1.33 = full
        if raw < 0.67:
            level = 0
            label = "none"
        elif raw < 1.33:
            level = 1
            label = "partial"
        else:
            level = 2
            label = "full"

        return {
            "level": level,
            "label": label,
            "raw_estimate": round(raw, 3),
            "observations": self._observation_counts[skill],
        }

    def get_all_states(self) -> dict[str, dict]:
        """Get schema state for all skills."""
        return {skill: self.get_state(skill) for skill in SKILLS}

    def get_schema_levels(self) -> dict[str, int]:
        """Get just the discretized schema levels for all skills (for EI computation)."""
        return {
            skill: self.get_state(skill)["level"]
            for skill in SKILLS
        }

    def compute_effective_ei(self, base_ei: int, relevant_skills: list[str]) -> float:
        """
        Compute effective EI for a problem given this student's schema states.

        Args:
            base_ei: Base EI from the problem's density_load or tier default
            relevant_skills: Skills involved in this problem

        Returns:
            Effective EI (minimum 1)
        """
        return effective_ei(base_ei, self.get_schema_levels(), relevant_skills)
