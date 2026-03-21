"""
State 3: Working memory utilization inference.

Dual estimation with cross-check:
1. Behavioral estimate: latency z-score, error spikes, coherence decline, help-seeking
2. Derived estimate: effective_EI(problem, student) + affective_load(affect)

Divergence between behavioral and derived estimates signals model miscalibration.
"""

from models.student import StudentState
from active_inference.state_space import discretize_wm, wm_label, WM_BINS


# Affective load contribution to WM
AFFECTIVE_LOAD = {
    "frustrated": 0.25,   # Anxiety/rumination consumes WM
    "engaged": 0.0,       # Neutral
    "bored": 0.0,         # Neutral (boredom doesn't consume WM, it just means low challenge)
}

# Weights for behavioral composite
BEHAVIORAL_WEIGHTS = {
    "latency_z": 0.30,
    "error_spike": 0.30,
    "coherence_decline": 0.20,
    "help_seeking": 0.20,
}

# Divergence threshold for model miscalibration warning
DIVERGENCE_THRESHOLD = 0.4


class WMLoadTracker:
    """
    Tracks working memory utilization via dual estimation.
    """

    def __init__(self):
        self._coherence_history: list[float] = []
        self._help_seeking_count: int = 0
        self._total_problems: int = 0

    def update_coherence(self, coherence: float):
        """Record a coherence value from LLM analysis."""
        self._coherence_history.append(coherence)
        if len(self._coherence_history) > 10:
            self._coherence_history = self._coherence_history[-10:]

    def record_help_seeking(self):
        """Record a help-seeking event (skip, 'idk', hint request)."""
        self._help_seeking_count += 1

    def record_problem(self):
        """Record that a problem was attempted."""
        self._total_problems += 1

    def estimate(self, student: StudentState, effective_ei: float,
                  affect: str) -> dict:
        """
        Compute WM load estimate with cross-check.

        Args:
            student: Student state with behavioral history
            effective_ei: Effective element interactivity for current problem
            affect: Current affect estimate from State 5

        Returns:
            dict with behavioral estimate, derived estimate, final estimate,
            divergence flag, and component signals
        """
        behavioral = self._behavioral_estimate(student)
        derived = self._derived_estimate(effective_ei, affect)

        # Final estimate: average of both
        final = 0.5 * behavioral + 0.5 * derived

        # Divergence detection
        divergence = abs(behavioral - derived)
        divergence_flag = divergence > DIVERGENCE_THRESHOLD

        final_label = wm_label(final)

        return {
            "wm_load": final_label,
            "wm_load_value": round(final, 3),
            "behavioral_estimate": round(behavioral, 3),
            "derived_estimate": round(derived, 3),
            "divergence": round(divergence, 3),
            "divergence_flag": divergence_flag,
            "bin": discretize_wm(final),
            "components": {
                "latency_z": round(self._latency_signal(student), 3),
                "error_spike": round(self._error_spike_signal(student), 3),
                "coherence_decline": round(self._coherence_decline_signal(), 3),
                "help_seeking": round(self._help_seeking_signal(), 3),
            },
        }

    def _behavioral_estimate(self, student: StudentState) -> float:
        """
        Composite behavioral WM estimate [0, 1].
        """
        signals = {
            "latency_z": self._latency_signal(student),
            "error_spike": self._error_spike_signal(student),
            "coherence_decline": self._coherence_decline_signal(),
            "help_seeking": self._help_seeking_signal(),
        }

        composite = sum(
            BEHAVIORAL_WEIGHTS[name] * value
            for name, value in signals.items()
        )
        return max(0.0, min(1.0, composite))

    def _derived_estimate(self, effective_ei: float, affect: str) -> float:
        """
        Derived WM estimate from EI + affective load [0, 1].
        """
        # Normalize EI to [0, 1] range (max EI is ~9 from density_load)
        ei_normalized = min(1.0, effective_ei / 9.0)

        affective_load = AFFECTIVE_LOAD.get(affect, 0.0)

        return min(1.0, ei_normalized + affective_load)

    def _latency_signal(self, student: StudentState) -> float:
        """
        Response time z-score → WM signal [0, 1].
        High z-score (much slower than baseline) = high load.
        """
        z = student.response_time_zscore()
        # Map z-score to [0, 1]: z=0 → 0.3, z=2 → 0.8, z=-2 → 0.0
        return max(0.0, min(1.0, 0.3 + z * 0.25))

    def _error_spike_signal(self, student: StudentState) -> float:
        """
        Detect sudden accuracy drops (capacity overflow).
        Not just low accuracy — sudden DROP from a higher baseline.
        """
        accs = student.accuracies
        if len(accs) < 5:
            return 0.0

        # Compare last 3 to previous 5
        recent = accs[-3:]
        baseline = accs[-8:-3] if len(accs) >= 8 else accs[:-3]

        if not baseline:
            return 0.0

        recent_acc = sum(recent) / len(recent)
        baseline_acc = sum(baseline) / len(baseline)

        drop = baseline_acc - recent_acc
        if drop > 0.4 and baseline_acc > 0.6:
            return 1.0  # Severe spike
        elif drop > 0.2 and baseline_acc > 0.5:
            return 0.6  # Moderate spike
        return 0.0

    def _coherence_decline_signal(self) -> float:
        """
        Detect declining explanation coherence (WM overload → fragmented explanations).
        """
        if len(self._coherence_history) < 3:
            return 0.0

        recent = self._coherence_history[-3:]
        baseline = self._coherence_history[:-3] if len(self._coherence_history) > 3 else self._coherence_history[:1]

        if not baseline:
            return 0.0

        recent_mean = sum(recent) / len(recent)
        baseline_mean = sum(baseline) / len(baseline)

        decline = baseline_mean - recent_mean
        return max(0.0, min(1.0, decline * 2.0))  # Scale: 0.5 decline → 1.0 signal

    def _help_seeking_signal(self) -> float:
        """
        Help-seeking rate → WM signal.
        """
        if self._total_problems < 3:
            return 0.0

        rate = self._help_seeking_count / self._total_problems
        return min(1.0, rate * 3.0)  # Scale: 33% help-seeking → 1.0 signal
