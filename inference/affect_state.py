"""
State 5: Affective / motivational state inference.

Combines:
1. LLM sentiment analysis (from the explanation analyzer)
2. Behavioral signal trends (response time, explanation effort, persistence)

Disambiguates frustrated vs. bored using WM load (State 3):
- Disengaged + high WM → frustrated
- Disengaged + low WM → bored

3-state output: frustrated / engaged / bored
"""

from models.student import StudentState


class AffectTracker:
    """
    Infers affective state from LLM analysis + behavioral signals.
    """

    def __init__(self, llm_weight: float = 0.4, behavioral_weight: float = 0.6):
        """
        Args:
            llm_weight: Weight given to LLM sentiment classification
            behavioral_weight: Weight given to behavioral signal composite
        """
        self.llm_weight = llm_weight
        self.behavioral_weight = behavioral_weight
        self._llm_affect_history: list[str] = []   # Recent LLM classifications
        self._window = 5  # Rolling window size

    def update_from_llm(self, affect: str):
        """Record an LLM affect classification."""
        self._llm_affect_history.append(affect)
        if len(self._llm_affect_history) > self._window:
            self._llm_affect_history = self._llm_affect_history[-self._window:]

    def infer(self, student: StudentState, wm_load: str = "moderate") -> dict:
        """
        Infer current affective state.

        Args:
            student: Student state with behavioral history
            wm_load: Current WM load estimate from State 3 ("low"/"moderate"/"high")

        Returns:
            dict with affect label, component scores, and confidence
        """
        behavioral = self._behavioral_score(student)
        llm = self._llm_score()

        # Weighted combination: each returns scores for [frustrated, engaged, bored]
        combined = [
            self.behavioral_weight * behavioral[i] + self.llm_weight * llm[i]
            for i in range(3)
        ]

        # Normalize
        total = sum(combined)
        if total > 0:
            combined = [c / total for c in combined]
        else:
            combined = [0.33, 0.34, 0.33]

        # WM-based disambiguation
        labels = ["frustrated", "engaged", "bored"]
        raw_label = labels[combined.index(max(combined))]

        # If disengaged (frustrated or bored is winning), use WM to disambiguate
        if raw_label in ("frustrated", "bored"):
            if wm_load == "high" and raw_label == "bored":
                # High WM + apparent boredom → more likely frustrated
                # (contradictory signal from plan: "Bored + High WM = extraneous load")
                # But also consider it might be frustration misread
                combined[0] += 0.15  # Boost frustrated
                combined[2] -= 0.15  # Reduce bored
            elif wm_load == "low" and raw_label == "frustrated":
                # Low WM + apparent frustration → might be boredom
                combined[2] += 0.1   # Boost bored
                combined[0] -= 0.1   # Reduce frustrated

        # Renormalize
        total = sum(combined)
        if total > 0:
            combined = [c / total for c in combined]

        final_label = labels[combined.index(max(combined))]

        return {
            "affect": final_label,
            "scores": {
                "frustrated": round(combined[0], 3),
                "engaged": round(combined[1], 3),
                "bored": round(combined[2], 3),
            },
            "behavioral_signals": {
                "response_time_trend": self._response_time_trend(student),
                "effort_trend": self._effort_trend(student),
                "recent_accuracy": round(student.recent_accuracy(), 3),
            },
        }

    def _behavioral_score(self, student: StudentState) -> list[float]:
        """
        Compute affect scores from behavioral signals.
        Returns [frustrated, engaged, bored] scores (unnormalized).
        """
        rt_trend = self._response_time_trend(student)
        effort_trend = self._effort_trend(student)
        accuracy = student.recent_accuracy()

        frustrated = 0.0
        engaged = 0.0
        bored = 0.0

        # Response time trend
        if rt_trend > 0.3:       # Slowing down → struggling
            frustrated += 0.3
        elif rt_trend < -0.3:    # Speeding up → rushing (bored or confident)
            bored += 0.2
        else:                     # Stable
            engaged += 0.3

        # Explanation effort trend
        if effort_trend < -0.3:   # Declining effort → disengagement
            bored += 0.3
            frustrated += 0.1     # Could also be frustration
        elif effort_trend > 0.1:  # Increasing effort → engaged
            engaged += 0.3
        else:
            engaged += 0.15

        # Accuracy
        if accuracy < 0.3:        # Failing a lot → frustrated
            frustrated += 0.3
        elif accuracy > 0.9:      # Too easy → bored
            bored += 0.2
            engaged += 0.1        # Could also be mastery engagement
        else:                     # Moderate → engaged
            engaged += 0.2

        return [frustrated, engaged, bored]

    def _llm_score(self) -> list[float]:
        """
        Compute affect scores from recent LLM classifications.
        Returns [frustrated, engaged, bored] scores.
        """
        if not self._llm_affect_history:
            return [0.33, 0.34, 0.33]

        counts = {"frustrated": 0, "engaged": 0, "bored": 0}
        for a in self._llm_affect_history:
            counts[a] = counts.get(a, 0) + 1

        total = len(self._llm_affect_history)
        return [counts["frustrated"] / total, counts["engaged"] / total, counts["bored"] / total]

    def _response_time_trend(self, student: StudentState) -> float:
        """
        Compute response time trend (slope of recent response times).
        Positive = slowing down, Negative = speeding up, ~0 = stable.
        Returns normalized trend in roughly [-1, 1].
        """
        times = student.response_times
        if len(times) < 3:
            return 0.0

        recent = times[-5:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator < 1e-6:
            return 0.0

        slope = numerator / denominator
        # Normalize by mean response time
        if y_mean > 0:
            return max(-1.0, min(1.0, slope / y_mean))
        return 0.0

    def _effort_trend(self, student: StudentState) -> float:
        """
        Compute explanation effort trend (slope of recent explanation lengths).
        Positive = increasing effort, Negative = declining effort.
        """
        lengths = student.explanation_lengths
        if len(lengths) < 3:
            return 0.0

        recent = lengths[-5:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator < 1e-6:
            return 0.0

        slope = numerator / denominator
        if y_mean > 0:
            return max(-1.0, min(1.0, slope / y_mean))
        return 0.0
