"""
Student state container holding all 5 hidden states + confidence gap per skill.
"""

from dataclasses import dataclass, field
from typing import Optional
import time

from domain.taxonomy import SKILLS, CONFUSABLE_PAIRS


@dataclass
class MemoryState:
    """Per-skill memory state (State 1). Populated by FSRS."""
    stability: float = 1.0      # Days until retrievability drops to 90% (≈SS)
    difficulty: float = 0.5     # Item-intrinsic hardness [0, 1]
    retrievability: float = 0.9 # Current recall probability (≈RS) [0, 1]
    last_review: Optional[float] = None  # Timestamp of last review


@dataclass
class SchemaState:
    """Per-skill schema state (State 2)."""
    level: int = 0              # 0=none, 1=partial, 2=full
    confidence: float = 0.5     # How confident we are in this estimate [0, 1]
    observations: int = 0       # Number of LLM classifications contributing


@dataclass
class SkillState:
    """Combined state for a single skill."""
    memory: MemoryState = field(default_factory=MemoryState)
    schema: SchemaState = field(default_factory=SchemaState)


@dataclass
class StudentState:
    """
    Full state container for one student across all skills.

    Contains:
    - Per-skill states (memory + schema)
    - Global states (WM load, affect)
    - Pairwise states (discriminability)
    - Confidence gap
    - Behavioral history for trend detection
    """
    student_id: str = ""

    # Per-skill states (States 1 + 2)
    skills: dict = field(default_factory=lambda: {
        skill: SkillState() for skill in SKILLS
    })

    # Global states
    wm_load: str = "moderate"           # "low" / "moderate" / "high" (State 3)
    wm_load_behavioral: float = 0.5     # Raw behavioral WM estimate [0, 1]
    wm_load_derived: float = 0.5        # Derived from EI + affect [0, 1]

    affect: str = "engaged"             # "frustrated" / "engaged" / "bored" (State 5)

    # Pairwise discriminability (State 4)
    # Maps (skill_a, skill_b) -> discriminability value [0, 1]
    discriminability: dict = field(default_factory=lambda: {
        pair: 0.5 for pair in CONFUSABLE_PAIRS
    })

    # Confidence gap (observable, not hidden)
    confidence_gap: float = 0.0         # self_report - accuracy, positive = overconfident
    confidence_observations: int = 0

    # Behavioral history (rolling window for trend detection)
    response_times: list = field(default_factory=list)    # Recent response times (ms)
    accuracies: list = field(default_factory=list)        # Recent correct/incorrect
    explanation_lengths: list = field(default_factory=list)  # Recent explanation word counts

    # Session metadata
    problems_seen: int = 0
    session_start: float = field(default_factory=time.time)

    def get_memory(self, skill: str) -> MemoryState:
        return self.skills[skill].memory

    def get_schema(self, skill: str) -> SchemaState:
        return self.skills[skill].schema

    def get_discriminability(self, skill_a: str, skill_b: str) -> float:
        pair = (skill_a, skill_b)
        rev = (skill_b, skill_a)
        if pair in self.discriminability:
            return self.discriminability[pair]
        if rev in self.discriminability:
            return self.discriminability[rev]
        return 0.5  # Default: uncertain

    def update_behavioral_history(self, response_time_ms: float, correct: bool,
                                   explanation_word_count: int, window: int = 20):
        """Add a new observation to the behavioral history, keeping a rolling window."""
        self.response_times.append(response_time_ms)
        self.accuracies.append(1.0 if correct else 0.0)
        self.explanation_lengths.append(explanation_word_count)
        self.problems_seen += 1

        # Trim to window size
        if len(self.response_times) > window:
            self.response_times = self.response_times[-window:]
            self.accuracies = self.accuracies[-window:]
            self.explanation_lengths = self.explanation_lengths[-window:]

    def recent_accuracy(self, n: int = 5) -> float:
        """Rolling accuracy over last n problems."""
        if not self.accuracies:
            return 0.5
        recent = self.accuracies[-n:]
        return sum(recent) / len(recent)

    def response_time_zscore(self) -> float:
        """Z-score of most recent response time vs rolling baseline."""
        if len(self.response_times) < 3:
            return 0.0
        baseline = self.response_times[:-1]
        mean = sum(baseline) / len(baseline)
        variance = sum((t - mean) ** 2 for t in baseline) / len(baseline)
        std = variance ** 0.5
        if std < 1e-6:
            return 0.0
        return (self.response_times[-1] - mean) / std
