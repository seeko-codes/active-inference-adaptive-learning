"""
Confidence gap tracking (observable, not a hidden state).

Compares student self-reported confidence with actual performance
to detect metacognitive miscalibration:
- Positive gap = overconfident (thinks they know it, don't)
- Negative gap = underconfident (thinks they can't, but can)
- Zero gap = well-calibrated
"""


class ConfidenceTracker:
    """
    Tracks confidence-accuracy gap for a student.
    """

    def __init__(self, window: int = 10):
        self.window = window
        self._observations: list[tuple[float, float]] = []  # (confidence, accuracy)

    def record(self, confidence: int, correct: bool):
        """
        Record a confidence self-report paired with actual performance.

        Args:
            confidence: Self-reported confidence (1-5 scale)
            correct: Whether the response was actually correct
        """
        # Normalize confidence to [0, 1]
        conf_normalized = (confidence - 1) / 4.0
        acc = 1.0 if correct else 0.0

        self._observations.append((conf_normalized, acc))
        if len(self._observations) > self.window:
            self._observations = self._observations[-self.window:]

    def get_gap(self) -> dict:
        """
        Get current confidence-accuracy gap.

        Returns:
            dict with gap value, label, and observation count
        """
        if not self._observations:
            return {
                "gap": 0.0,
                "label": "unknown",
                "observations": 0,
                "mean_confidence": 0.5,
                "mean_accuracy": 0.5,
            }

        mean_conf = sum(c for c, _ in self._observations) / len(self._observations)
        mean_acc = sum(a for _, a in self._observations) / len(self._observations)
        gap = mean_conf - mean_acc

        if gap > 0.15:
            label = "overconfident"
        elif gap < -0.15:
            label = "underconfident"
        else:
            label = "calibrated"

        return {
            "gap": round(gap, 3),
            "label": label,
            "observations": len(self._observations),
            "mean_confidence": round(mean_conf, 3),
            "mean_accuracy": round(mean_acc, 3),
        }
