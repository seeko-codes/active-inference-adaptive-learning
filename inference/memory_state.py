"""
State 1: Per-skill memory state inference using FSRS.

Wraps py-fsrs to maintain per-skill storage strength (stability) and
retrieval strength (retrievability) for each student.

Observable inputs: (skill_id, correct/incorrect, timestamp)
Outputs: stability (≈SS), difficulty, retrievability (≈RS) per skill

FSRS internally uses:
- Power-law forgetting curve for retrievability decay
- 19 trainable weights (using defaults for prototype)
- 4 ratings: Again (1), Hard (2), Good (3), Easy (4)

We map binary correct/incorrect + response time to FSRS ratings:
- Incorrect → Again
- Correct + slow (high effort) → Hard
- Correct + normal → Good
- Correct + fast (automatic) → Easy
"""

from datetime import datetime, timezone, timedelta
from fsrs import Scheduler, Card, Rating

from domain.taxonomy import SKILLS
from active_inference.state_space import discretize_rs, discretize_ss, rs_label, ss_label


class MemoryTracker:
    """
    Tracks per-skill memory state for one student using FSRS.
    """

    def __init__(self, skills: list[str] = None):
        self.scheduler = Scheduler()
        self.skills = skills or SKILLS
        # One FSRS card per skill
        self.cards: dict[str, Card] = {skill: Card() for skill in self.skills}

    def get_state(self, skill: str, now: datetime = None) -> dict:
        """
        Get current memory state for a skill.

        Returns:
            dict with stability (≈SS), difficulty, retrievability (≈RS),
            plus discretized versions for the POMDP.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        card = self.cards[skill]
        retrievability = self.scheduler.get_card_retrievability(card, now)

        return {
            "stability": card.stability or 0.0,
            "difficulty": card.difficulty or 0.5,
            "retrievability": retrievability,
            "rs_bin": discretize_rs(retrievability),
            "rs_label": rs_label(retrievability),
            "ss_bin": discretize_ss(card.stability or 0.0),
            "ss_label": ss_label(card.stability or 0.0),
            "last_review": card.last_review,
            "fsrs_state": card.state.name,
        }

    def record_response(self, skill: str, correct: bool,
                         response_time_ms: float = None,
                         now: datetime = None) -> dict:
        """
        Record a student response and update the memory model.

        Args:
            skill: Which skill was tested
            correct: Whether the response was correct
            response_time_ms: Response time in milliseconds (optional, for rating refinement)
            now: Timestamp of the response

        Returns:
            Updated memory state dict
        """
        if now is None:
            now = datetime.now(timezone.utc)

        rating = self._map_to_rating(correct, response_time_ms)
        card = self.cards[skill]
        updated_card, review_log = self.scheduler.review_card(card, rating, now)
        self.cards[skill] = updated_card

        return self.get_state(skill, now)

    def _map_to_rating(self, correct: bool, response_time_ms: float = None) -> Rating:
        """
        Map binary outcome + response time to FSRS rating.

        Mapping:
        - Incorrect → Again (complete failure)
        - Correct + slow (> 2x baseline or > 15s) → Hard (effortful recall)
        - Correct + normal → Good (standard recall)
        - Correct + fast (< 3s) → Easy (automatic recall)

        Without response time, correct → Good, incorrect → Again.
        """
        if not correct:
            return Rating.Again

        if response_time_ms is None:
            return Rating.Good

        seconds = response_time_ms / 1000.0
        if seconds < 3.0:
            return Rating.Easy
        elif seconds > 15.0:
            return Rating.Hard
        else:
            return Rating.Good

    def get_all_states(self, now: datetime = None) -> dict[str, dict]:
        """Get memory state for all skills."""
        if now is None:
            now = datetime.now(timezone.utc)
        return {skill: self.get_state(skill, now) for skill in self.skills}

    def get_due_skills(self, now: datetime = None) -> list[tuple[str, float]]:
        """
        Get skills sorted by review urgency (lowest retrievability first).

        Returns:
            List of (skill, retrievability) tuples, most urgent first.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        urgency = []
        for skill in self.skills:
            card = self.cards[skill]
            r = self.scheduler.get_card_retrievability(card, now)
            urgency.append((skill, r))

        return sorted(urgency, key=lambda x: x[1])

    def to_dict(self) -> dict:
        """Serialize tracker state for persistence."""
        return {
            skill: self.cards[skill].to_dict()
            for skill in self.skills
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryTracker":
        """Restore tracker state from serialized dict."""
        tracker = cls(skills=list(data.keys()))
        for skill, card_data in data.items():
            tracker.cards[skill] = Card.from_dict(card_data)
        return tracker
