"""
Simulation analysis: parameter recovery, policy comparison, state ablation.

This module consumes TrajectoryResult data from monte_carlo.py and answers
the four validation questions:

1. Parameter recovery: Can the inference engine recover known ground-truth
   states from simulated observations?
2. Policy comparison: Does the meta-function policy produce better learning
   trajectories than baselines?
3. State ablation: Does dropping any of the 5 states degrade performance?
4. Learning landscape: Which learner types benefit most from which policies?
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

from simulation.monte_carlo import TrajectoryResult, run_trajectory
from simulation.learner_types import (
    LearnerType, get_all_archetypes, SKILLS,
)
from active_inference.state_space import (
    rs_label, ss_label, discretize_rs, discretize_ss,
)


# ==========================================
# 1. Policy Comparison
# ==========================================

@dataclass
class PolicyComparison:
    """Summary statistics for one (learner_type, policy) combination."""
    learner_type: str
    policy: str
    n_runs: int = 0
    mean_final_mastery: float = 0.0
    std_final_mastery: float = 0.0
    mean_accuracy: float = 0.0
    mean_mastery_trajectory: list = field(default_factory=list)
    mean_time_frustrated: float = 0.0
    mean_time_bored: float = 0.0
    mean_time_engaged: float = 0.0


def compare_policies(results: list[TrajectoryResult]) -> list[PolicyComparison]:
    """
    Compare policies across learner types.

    Groups results by (learner_type, policy) and computes summary statistics.

    Returns:
        List of PolicyComparison, one per (learner_type, policy) pair
    """
    groups = defaultdict(list)
    for r in results:
        groups[(r.learner_type, r.policy)].append(r)

    comparisons = []
    for (lt_name, policy), trajectories in sorted(groups.items()):
        masteries = [t.final_mastery for t in trajectories]
        accuracies = []
        frustration_fracs = []
        boredom_fracs = []
        engaged_fracs = []

        for t in trajectories:
            total_correct = sum(s.n_correct for s in t.sessions)
            total_problems = sum(s.n_total for s in t.sessions)
            accuracies.append(total_correct / max(1, total_problems))

            total_affect = sum(
                sum(s.affect_counts.values()) for s in t.sessions
            )
            if total_affect > 0:
                frustration_fracs.append(
                    sum(s.affect_counts.get("frustrated", 0) for s in t.sessions) / total_affect
                )
                boredom_fracs.append(
                    sum(s.affect_counts.get("bored", 0) for s in t.sessions) / total_affect
                )
                engaged_fracs.append(
                    sum(s.affect_counts.get("engaged", 0) for s in t.sessions) / total_affect
                )

        # Align trajectories for mean computation
        max_len = max(len(t.mastery_trajectory) for t in trajectories)
        padded = []
        for t in trajectories:
            traj = t.mastery_trajectory
            if len(traj) < max_len:
                traj = traj + [traj[-1]] * (max_len - len(traj))
            padded.append(traj)
        mean_traj = np.mean(padded, axis=0).tolist()

        comp = PolicyComparison(
            learner_type=lt_name,
            policy=policy,
            n_runs=len(trajectories),
            mean_final_mastery=np.mean(masteries),
            std_final_mastery=np.std(masteries),
            mean_accuracy=np.mean(accuracies),
            mean_mastery_trajectory=mean_traj,
            mean_time_frustrated=np.mean(frustration_fracs) if frustration_fracs else 0,
            mean_time_bored=np.mean(boredom_fracs) if boredom_fracs else 0,
            mean_time_engaged=np.mean(engaged_fracs) if engaged_fracs else 0,
        )
        comparisons.append(comp)

    return comparisons


def format_comparison_table(comparisons: list[PolicyComparison]) -> str:
    """Format policy comparisons as a readable table."""
    # Group by learner type
    by_lt = defaultdict(list)
    for c in comparisons:
        by_lt[c.learner_type].append(c)

    lines = []
    lines.append(f"{'Learner Type':<22s} {'Policy':<18s} {'Mastery':>8s} {'Acc':>6s} "
                 f"{'Engaged':>8s} {'Frust':>6s} {'Bored':>6s}")
    lines.append("-" * 80)

    for lt_name in sorted(by_lt.keys()):
        for c in sorted(by_lt[lt_name], key=lambda x: -x.mean_final_mastery):
            lines.append(
                f"{c.learner_type:<22s} {c.policy:<18s} "
                f"{c.mean_final_mastery:>7.3f}  {c.mean_accuracy:>5.2f}  "
                f"{c.mean_time_engaged:>7.1%}  {c.mean_time_frustrated:>5.1%}  "
                f"{c.mean_time_bored:>5.1%}"
            )
        lines.append("")

    return "\n".join(lines)


# ==========================================
# 2. Parameter Recovery
# ==========================================

@dataclass
class RecoveryResult:
    """Results from parameter recovery analysis."""
    skill: str
    true_rs_bin: str
    true_ss_bin: str
    true_schema: int
    inferred_rs_bin: str = ""
    inferred_ss_bin: str = ""
    inferred_schema: int = -1
    rs_match: bool = False
    ss_match: bool = False
    schema_match: bool = False


def parameter_recovery(
    results: list[TrajectoryResult],
) -> dict:
    """
    Check if ground-truth states can be recovered from observations.

    For each trajectory, compare the agent's final ground-truth state
    (from the simulation) with what an inference engine would estimate
    from the sequence of observations.

    This is a simplified version — full parameter recovery requires
    running the actual inference engine on the observations. Here we
    check: are the observations informative enough to distinguish states?

    Returns:
        dict with recovery metrics per state dimension
    """
    rs_errors = []
    ss_errors = []
    schema_matches = []

    for traj in results:
        if not traj.final_state or "skills" not in traj.final_state:
            continue

        for skill, state in traj.final_state["skills"].items():
            # True state
            true_rs = state["rs"]
            true_ss = state["ss"]
            true_schema = state["schema"]

            # Naive inference from observations: use accuracy and response times
            # from the last session as a proxy for what the inference engine
            # would estimate
            if not traj.sessions:
                continue
            last_session = traj.sessions[-1]
            skill_obs = [o for o in last_session.observations if o.skill == skill]

            if len(skill_obs) < 2:
                continue

            # Infer RS from accuracy
            acc = sum(1 for o in skill_obs if o.correct) / len(skill_obs)
            inferred_rs = acc  # Crude but informative

            # Infer schema from explanation quality (continuous comparison)
            mean_quality = np.mean([o.explanation_quality for o in skill_obs])
            # Discretize both true and inferred for classification accuracy
            true_schema_bin = 0 if true_schema < 0.33 else 1 if true_schema < 0.67 else 2
            if mean_quality > 0.6:
                inferred_schema_bin = 2
            elif mean_quality > 0.35:
                inferred_schema_bin = 1
            else:
                inferred_schema_bin = 0

            # Record errors
            rs_errors.append(abs(true_rs - inferred_rs))
            schema_matches.append(1 if inferred_schema_bin == true_schema_bin else 0)

    return {
        "rs_mae": np.mean(rs_errors) if rs_errors else float("nan"),
        "rs_n": len(rs_errors),
        "schema_accuracy": np.mean(schema_matches) if schema_matches else float("nan"),
        "schema_n": len(schema_matches),
    }


# ==========================================
# 3. State Ablation
# ==========================================

def run_ablation(
    learner_types: list[LearnerType] = None,
    n_seeds: int = 5,
    n_sessions: int = 5,
    problems_per_session: int = 20,
    base_seed: int = 42,
) -> dict:
    """
    Ablation study: compare meta-function performance with and without
    each state dimension.

    Ablation method: for each state, fix that state to its default/uninformative
    value in the policy's input, forcing the policy to make decisions without it.

    Returns:
        dict mapping condition -> mean final mastery
    """
    from simulation.monte_carlo import run_trajectory
    from simulation.simulated_agent import SimulatedAgent

    if learner_types is None:
        learner_types = get_all_archetypes()[:4]  # Subset for speed

    conditions = {
        "full": {},  # No ablation
        "no_affect": {"affect": "engaged"},
        "no_wm": {"wm_label": "moderate"},
        "no_schema": {"schema_label": "none"},
        "no_memory": {"rs_label": "moderate", "ss_label": "moderate"},
        "no_discrim": {"low_discrim_pairs": []},
    }

    results = {}
    for condition_name, overrides in conditions.items():
        masteries = []

        for lt in learner_types:
            for seed_offset in range(n_seeds):
                seed = base_seed + seed_offset
                rng = np.random.default_rng(seed)
                agent = SimulatedAgent(lt, rng=rng)

                for session_idx in range(n_sessions):
                    for _ in range(problems_per_session):
                        # Get true state
                        state = agent.state
                        skill = min(SKILLS, key=lambda s: state.rs[s])

                        # Build action selection inputs
                        rs = state.rs[skill]
                        ss = state.ss[skill]
                        schema = state.schema[skill]
                        wm = state.wm_utilization

                        # Continuous schema → discrete label
                        schema_label = "none" if schema < 0.33 else "partial" if schema < 0.67 else "full"

                        kwargs = {
                            "skill": skill,
                            "rs_label": rs_label(rs),
                            "ss_label": ss_label(ss),
                            "schema_label": schema_label,
                            "wm_label": ("low" if wm < 0.33 else "high" if wm > 0.67 else "moderate"),
                            "affect": state.affect,
                            "effective_ei": 5.0,
                            "low_discrim_pairs": [],
                            "schema_adequate_for": {},
                        }

                        # Apply ablation overrides
                        kwargs.update(overrides)

                        from meta_function import select_action
                        action_sel = select_action(**kwargs)

                        tier = min(7, max(0, int(schema * 4 + 1)))
                        agent.present_problem(
                            skill=skill,
                            action=action_sel.action,
                            tier=tier,
                        )

                    if session_idx < n_sessions - 1:
                        agent.apply_forgetting(24)

                masteries.append(agent.mastery_score())

        results[condition_name] = {
            "mean_mastery": np.mean(masteries),
            "std_mastery": np.std(masteries),
            "n": len(masteries),
        }

    return results


def format_ablation_table(ablation_results: dict) -> str:
    """Format ablation results as a readable table."""
    lines = []
    lines.append(f"{'Condition':<18s} {'Mean Mastery':>13s} {'Std':>8s} {'Delta':>8s}")
    lines.append("-" * 50)

    full_mastery = ablation_results.get("full", {}).get("mean_mastery", 0)

    for condition, stats in sorted(
        ablation_results.items(),
        key=lambda x: -x[1]["mean_mastery"],
    ):
        delta = stats["mean_mastery"] - full_mastery
        delta_str = f"{delta:+.4f}" if condition != "full" else "  base"
        lines.append(
            f"{condition:<18s} {stats['mean_mastery']:>12.4f}  "
            f"{stats['std_mastery']:>7.4f}  {delta_str}"
        )

    return "\n".join(lines)


# ==========================================
# 4. Learning Landscape
# ==========================================

def learning_landscape(comparisons: list[PolicyComparison]) -> dict:
    """
    Map which policies work best for which learner types.

    Returns:
        dict with:
        - best_policy: learner_type -> best policy name
        - advantage: learner_type -> mastery advantage of best over second-best
        - vulnerability: which learner types are most sensitive to policy choice
    """
    by_lt = defaultdict(list)
    for c in comparisons:
        by_lt[c.learner_type].append(c)

    landscape = {
        "best_policy": {},
        "advantage": {},
        "vulnerability": {},
    }

    for lt_name, comps in by_lt.items():
        sorted_comps = sorted(comps, key=lambda x: -x.mean_final_mastery)

        best = sorted_comps[0]
        landscape["best_policy"][lt_name] = best.policy

        if len(sorted_comps) > 1:
            second = sorted_comps[1]
            advantage = best.mean_final_mastery - second.mean_final_mastery
            landscape["advantage"][lt_name] = round(advantage, 4)

            # Vulnerability = range / mean (coefficient of variation)
            all_masteries = [c.mean_final_mastery for c in sorted_comps]
            mean_m = np.mean(all_masteries)
            if mean_m > 0:
                landscape["vulnerability"][lt_name] = round(
                    (max(all_masteries) - min(all_masteries)) / mean_m, 3
                )

    return landscape


def format_landscape(landscape: dict) -> str:
    """Format learning landscape as readable text."""
    lines = []
    lines.append(f"{'Learner Type':<22s} {'Best Policy':<18s} {'Advantage':>10s} {'Vulnerability':>13s}")
    lines.append("-" * 65)

    for lt_name in sorted(landscape["best_policy"].keys()):
        lines.append(
            f"{lt_name:<22s} {landscape['best_policy'][lt_name]:<18s} "
            f"{landscape['advantage'].get(lt_name, 0):>9.4f}  "
            f"{landscape['vulnerability'].get(lt_name, 0):>12.3f}"
        )

    return "\n".join(lines)


# ==========================================
# Full Analysis Pipeline
# ==========================================

def run_full_analysis(
    results: list[TrajectoryResult],
    run_ablation_study: bool = False,
    ablation_n_seeds: int = 3,
) -> dict:
    """
    Run the complete analysis pipeline on Monte Carlo results.

    Args:
        results: List of TrajectoryResult from run_monte_carlo
        run_ablation_study: Whether to run the ablation study (slow)
        ablation_n_seeds: Seeds per condition for ablation

    Returns:
        dict with all analysis results
    """
    print("=== Policy Comparison ===")
    comparisons = compare_policies(results)
    print(format_comparison_table(comparisons))

    print("\n=== Parameter Recovery ===")
    recovery = parameter_recovery(results)
    print(f"RS Mean Absolute Error: {recovery['rs_mae']:.3f} (n={recovery['rs_n']})")
    print(f"Schema Classification Accuracy: {recovery['schema_accuracy']:.1%} (n={recovery['schema_n']})")

    print("\n=== Learning Landscape ===")
    landscape = learning_landscape(comparisons)
    print(format_landscape(landscape))

    analysis = {
        "comparisons": comparisons,
        "recovery": recovery,
        "landscape": landscape,
    }

    if run_ablation_study:
        print("\n=== State Ablation (this may take a moment) ===")
        ablation = run_ablation(n_seeds=ablation_n_seeds)
        print(format_ablation_table(ablation))
        analysis["ablation"] = ablation

    return analysis
