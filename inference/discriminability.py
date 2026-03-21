"""
State 4: Discriminability inference via confusion matrices.

Tracks pairwise discriminability between confusable skill pairs.
Updated from two sources:
1. Errors: when a student applies skill A where skill B was correct
2. LLM explanation flags: when the explanation reveals confusion even on correct answers

Uses recency-weighted confusion rates to adapt to learning over time.
"""

from domain.taxonomy import CONFUSABLE_PAIRS, SKILLS
from active_inference.state_space import discretize_discrim, discrim_label


class DiscriminabilityTracker:
    """
    Tracks pairwise discriminability between confusable skill pairs.

    discriminability(A, B) = 1 - confusion_rate(A↔B)
    """

    def __init__(self, decay: float = 0.95):
        """
        Args:
            decay: Recency weighting factor. Each old observation is weighted
                   by decay^(age). 0.95 means observations from 20 trials ago
                   have ~36% weight.
        """
        self.decay = decay

        # Per-pair: list of (confused: bool, weight) observations
        # True = confusion observed, False = correct discrimination
        self._observations: dict[tuple, list[tuple[bool, float]]] = {
            pair: [] for pair in CONFUSABLE_PAIRS
        }

    def _normalize_pair(self, skill_a: str, skill_b: str) -> tuple:
        """Ensure consistent pair ordering."""
        pair = (skill_a, skill_b)
        rev = (skill_b, skill_a)
        if pair in self._observations:
            return pair
        if rev in self._observations:
            return rev
        return None

    def record_error(self, correct_skill: str, applied_skill: str):
        """
        Record that the student applied the wrong skill.

        Args:
            correct_skill: The skill that should have been applied
            applied_skill: The skill the student actually applied
        """
        pair = self._normalize_pair(correct_skill, applied_skill)
        if pair is None:
            return  # Not a tracked confusable pair

        # Age existing observations
        self._observations[pair] = [
            (confused, weight * self.decay)
            for confused, weight in self._observations[pair]
        ]
        # Add confusion observation
        self._observations[pair].append((True, 1.0))

    def record_correct_discrimination(self, skill_a: str, skill_b: str):
        """
        Record that the student correctly distinguished between two skills.
        Called when the student correctly applies skill_a in a context where
        skill_b could have been confused.
        """
        pair = self._normalize_pair(skill_a, skill_b)
        if pair is None:
            return

        self._observations[pair] = [
            (confused, weight * self.decay)
            for confused, weight in self._observations[pair]
        ]
        self._observations[pair].append((False, 1.0))

    def record_llm_confusion(self, skill: str, confused_with: str):
        """
        Record confusion detected by LLM explanation analysis,
        even if the answer was correct.
        """
        pair = self._normalize_pair(skill, confused_with)
        if pair is None:
            return

        self._observations[pair] = [
            (confused, weight * self.decay)
            for confused, weight in self._observations[pair]
        ]
        # LLM-detected confusion counts but with slightly less weight
        # since the student still got the right answer
        self._observations[pair].append((True, 0.7))

    def get_discriminability(self, skill_a: str, skill_b: str) -> dict:
        """Get discriminability state for a pair."""
        pair = self._normalize_pair(skill_a, skill_b)
        if pair is None:
            return {"value": 0.5, "label": "moderate", "bin": 1, "observations": 0}

        obs = self._observations[pair]
        if not obs:
            return {"value": 0.5, "label": "moderate", "bin": 1, "observations": 0}

        total_weight = sum(w for _, w in obs)
        confusion_weight = sum(w for confused, w in obs if confused)

        confusion_rate = confusion_weight / total_weight if total_weight > 0 else 0.0
        discrim = 1.0 - confusion_rate

        return {
            "value": round(discrim, 3),
            "label": discrim_label(discrim),
            "bin": discretize_discrim(discrim),
            "observations": len(obs),
        }

    def get_all_states(self) -> dict[tuple, dict]:
        """Get discriminability for all tracked pairs."""
        return {
            pair: self.get_discriminability(pair[0], pair[1])
            for pair in CONFUSABLE_PAIRS
        }

    def get_low_discriminability_pairs(self, threshold: float = 0.4) -> list[tuple]:
        """
        Get pairs with low discriminability (below threshold).
        These are candidates for interleaving.
        """
        low_pairs = []
        for pair in CONFUSABLE_PAIRS:
            state = self.get_discriminability(pair[0], pair[1])
            if state["value"] < threshold:
                low_pairs.append((pair, state["value"]))
        return sorted(low_pairs, key=lambda x: x[1])
