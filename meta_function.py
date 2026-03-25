"""
Meta function: action selection from joint 5-state belief.

This defines the action space (what actions exist and their preconditions)
using rule-based logic. In the full system, active inference (pymdp) wraps
this to decide WHICH action to deploy based on expected free energy.

Priority ordering (from the plan):
1. Break vicious cycles (frustrated + high WM)
2. Re-engage (bored + low WM)
3. Don't overload (high WM regardless of affect)
4. Discriminability-driven interleaving
5. Schema-driven technique selection
6. Memory-driven spacing

This priority ordering ensures safety constraints are checked first
before optimizing for learning.
"""

from dataclasses import dataclass
from typing import Optional

from active_inference.transition_model import ACTIONS


@dataclass
class ActionSelection:
    action: str
    reason: str
    skill: str                          # Which skill this action targets
    interleave_pair: Optional[tuple] = None  # For interleave action
    confidence: float = 1.0             # How confident the system is in this choice


def select_action(
    skill: str,
    rs_label: str,
    ss_label: str,
    schema_label: str,
    wm_label: str,
    affect: str,
    effective_ei: float,
    low_discrim_pairs: list = None,
    schema_adequate_for: dict = None,
    recent_accuracy: float = 1.0,
) -> ActionSelection:
    """
    Select pedagogical action from joint state.

    Args:
        skill: The skill being considered
        rs_label: Retrievability label (very_low/low/moderate/high/very_high)
        ss_label: Stability label (very_low/low/moderate/high)
        schema_label: Schema level (none/partial/full)
        wm_label: WM load label (low/moderate/high)
        affect: Affective state (frustrated/engaged/bored)
        effective_ei: Effective element interactivity for this student/problem
        low_discrim_pairs: List of (pair, discrim_value) for low-discriminability pairs
        schema_adequate_for: Dict mapping skill -> bool, whether schema is at least partial

    Returns:
        ActionSelection with chosen action and reasoning
    """
    if low_discrim_pairs is None:
        low_discrim_pairs = []
    if schema_adequate_for is None:
        schema_adequate_for = {}

    # ==========================================
    # Priority 1: Break vicious cycles
    # ==========================================
    if affect == "frustrated" and wm_label == "high":
        return ActionSelection(
            action="reduce_load",
            reason="Frustrated + high WM load: breaking overload→frustration cycle",
            skill=skill,
        )

    # ==========================================
    # Priority 2: Re-engage bored student
    # ==========================================
    if affect == "bored" and wm_label == "low":
        return ActionSelection(
            action="increase_challenge",
            reason="Bored + low WM load: increasing difficulty to re-engage",
            skill=skill,
        )

    # ==========================================
    # Priority 3: Don't overload (only if load is extraneous, not germane)
    # ==========================================
    if wm_label == "high" and recent_accuracy < 0.4:
        return ActionSelection(
            action="reduce_load",
            reason=f"High WM load + low accuracy ({recent_accuracy:.0%}): extraneous load, reducing",
            skill=skill,
        )

    # ==========================================
    # Priority 4: Discriminability-driven interleaving
    # ==========================================
    for pair, discrim_value in low_discrim_pairs:
        skill_a, skill_b = pair
        # Only interleave if both skills have at least partial schemas
        a_ok = schema_adequate_for.get(skill_a, False)
        b_ok = schema_adequate_for.get(skill_b, False)
        if a_ok and b_ok and wm_label != "high":
            return ActionSelection(
                action="interleave",
                reason=f"Low discriminability ({discrim_value:.2f}) between {skill_a} and {skill_b}, both have adequate schemas",
                skill=skill,
                interleave_pair=pair,
            )

    # ==========================================
    # Priority 5: Schema-driven technique selection
    # ==========================================
    if schema_label == "none" and effective_ei > 4:
        return ActionSelection(
            action="worked_example",
            reason=f"No schema + high EI ({effective_ei:.1f}): scaffolded schema building",
            skill=skill,
        )

    if schema_label == "partial":
        return ActionSelection(
            action="faded_example",
            reason="Partial schema: transitioning toward independence",
            skill=skill,
        )

    # ==========================================
    # Priority 6: Memory-driven spacing
    # ==========================================
    if rs_label in ("very_low", "low") and ss_label in ("moderate", "high"):
        return ActionSelection(
            action="space_and_test",
            reason=f"Low RS ({rs_label}) + solid SS ({ss_label}): desirable difficulty zone",
            skill=skill,
        )

    if rs_label in ("very_low", "low") and ss_label in ("very_low", "low"):
        return ActionSelection(
            action="reteach",
            reason=f"Low RS ({rs_label}) + weak SS ({ss_label}): needs re-encoding, not testing",
            skill=skill,
        )

    # ==========================================
    # Default: continue practice
    # ==========================================
    if schema_label == "full" and rs_label in ("high", "very_high"):
        return ActionSelection(
            action="space_and_test",
            reason="Full schema + high RS: maintain via spaced testing",
            skill=skill,
            confidence=0.7,
        )

    return ActionSelection(
        action="faded_example",
        reason="Default: moderate practice with partial scaffolding",
        skill=skill,
        confidence=0.5,
    )


def select_session_actions(
    skills_states: dict,
    n_problems: int = 10,
) -> list[ActionSelection]:
    """
    Select a sequence of actions for a session.

    Args:
        skills_states: Dict mapping skill -> {rs_label, ss_label, schema_label, etc.}
        n_problems: Number of problems to fill the session

    Returns:
        Ordered list of ActionSelections
    """
    actions = []
    for skill, state in skills_states.items():
        action = select_action(
            skill=skill,
            rs_label=state.get("rs_label", "moderate"),
            ss_label=state.get("ss_label", "very_low"),
            schema_label=state.get("schema_label", "none"),
            wm_label=state.get("wm_label", "moderate"),
            affect=state.get("affect", "engaged"),
            effective_ei=state.get("effective_ei", 5.0),
            low_discrim_pairs=state.get("low_discrim_pairs", []),
            schema_adequate_for=state.get("schema_adequate_for", {}),
        )
        actions.append(action)

    # Sort by urgency: reduce_load first, then reteach, then space_and_test, etc.
    priority_order = {
        "reduce_load": 0,
        "increase_challenge": 1,
        "reteach": 2,
        "worked_example": 3,
        "space_and_test": 4,
        "interleave": 5,
        "faded_example": 6,
        "diagnostic_probe": 7,
    }
    actions.sort(key=lambda a: priority_order.get(a.action, 99))

    return actions[:n_problems]
