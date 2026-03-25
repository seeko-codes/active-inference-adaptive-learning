"""
Precompute simulation results for poster and frontend.

Strategy:
- 50,000 students × 4 non-AI policies = 200,000 trajectories (~13 min)
- 10,000 students × active_inference = 10,000 trajectories (parallelized across cores)
- Total: 210,000 trajectories

Outputs:
- poster/results/trajectories_non_ai.json  (200K trajectories, summary stats)
- poster/results/trajectories_ai.json      (10K trajectories, summary stats)
- poster/results/analysis.json             (policy comparison, landscape, recovery)
"""

import numpy as np
import json
import time
import os
import multiprocessing as mp
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, "/Users/aatutor/adaptive-learning")

from simulation.monte_carlo import run_trajectory, POLICIES
from simulation.learner_types import sample_learner_type, get_all_archetypes
from simulation.analysis import compare_policies, parameter_recovery, learning_landscape

RESULTS_DIR = Path("/Users/aatutor/adaptive-learning/poster/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SESSIONS = 5
PROBLEMS_PER_SESSION = 20
HOURS_BETWEEN = 24.0
BASE_SEED = 42

NON_AI_POLICIES = ["meta_function", "random", "fsrs_only", "fixed_curriculum"]
AI_POLICY = "active_inference"


def trajectory_to_summary(t):
    """Extract compact summary from a TrajectoryResult."""
    return {
        "learner_type": t.learner_type,
        "policy": t.policy,
        "seed": t.seed,
        "final_mastery": round(t.final_mastery, 5),
        "mastery_trajectory": [round(m, 5) for m in t.mastery_trajectory],
        "total_problems": t.total_problems,
        "accuracy": round(
            sum(s.n_correct for s in t.sessions) / max(1, sum(s.n_total for s in t.sessions)), 4
        ),
        "affect": {
            "frustrated": sum(s.affect_counts.get("frustrated", 0) for s in t.sessions),
            "engaged": sum(s.affect_counts.get("engaged", 0) for s in t.sessions),
            "bored": sum(s.affect_counts.get("bored", 0) for s in t.sessions),
        },
    }


def run_single_ai_trajectory(args):
    """Worker for multiprocessing: run one active_inference trajectory."""
    lt_dict, seed, idx, total = args
    # Reconstruct LearnerType
    from simulation.learner_types import LearnerType, CognitiveParams
    params = CognitiveParams(**lt_dict["params"])
    lt = LearnerType(
        name=lt_dict["name"],
        description=lt_dict.get("description", ""),
        params=params,
        prevalence=lt_dict.get("prevalence", 1.0),
    )
    t = run_trajectory(lt, AI_POLICY, N_SESSIONS, PROBLEMS_PER_SESSION, HOURS_BETWEEN, seed)
    if (idx + 1) % 25 == 0:
        print(f"  AI [{idx + 1}/{total}]", flush=True)
    return trajectory_to_summary(t)


def learner_type_to_dict(lt):
    """Serialize LearnerType for multiprocessing."""
    from dataclasses import asdict
    return {
        "name": lt.name,
        "description": lt.description,
        "params": asdict(lt.params),
        "prevalence": lt.prevalence,
    }


def run_non_ai(n_students=50000):
    """Run all non-AI policies. Fast — ~13 min for 50K students."""
    print(f"\n{'='*60}")
    print(f"NON-AI POLICIES: {n_students} students × {len(NON_AI_POLICIES)} policies")
    print(f"= {n_students * len(NON_AI_POLICIES):,} trajectories")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(BASE_SEED)
    results = []
    t0 = time.time()

    for i in range(n_students):
        lt = sample_learner_type(rng, name=f"pop_{i:05d}")
        for policy in NON_AI_POLICIES:
            t = run_trajectory(lt, policy, N_SESSIONS, PROBLEMS_PER_SESSION, HOURS_BETWEEN, BASE_SEED + i)
            results.append(trajectory_to_summary(t))

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_students - i - 1) / rate
            print(f"  [{i+1:,}/{n_students:,}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"\nDone: {len(results):,} trajectories in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    out_path = RESULTS_DIR / "trajectories_non_ai.json"
    with open(out_path, "w") as f:
        json.dump({"n_students": n_students, "policies": NON_AI_POLICIES, "results": results}, f)
    print(f"Saved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return results


def run_ai_parallel(n_students=10000, n_workers=None):
    """Run active_inference policy with multiprocessing."""
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"\n{'='*60}")
    print(f"ACTIVE INFERENCE: {n_students} students × 1 policy")
    print(f"= {n_students:,} trajectories on {n_workers} workers")
    print(f"Estimated: ~{n_students * 26 / n_workers / 3600:.1f} hours")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(BASE_SEED + 999999)  # Different seed space from non-AI
    tasks = []
    for i in range(n_students):
        lt = sample_learner_type(rng, name=f"ai_{i:05d}")
        tasks.append((learner_type_to_dict(lt), BASE_SEED + i, i, n_students))

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(run_single_ai_trajectory, tasks, chunksize=10)

    elapsed = time.time() - t0
    print(f"\nDone: {len(results):,} trajectories in {elapsed:.1f}s ({elapsed/3600:.1f} hours)")

    out_path = RESULTS_DIR / "trajectories_ai.json"
    with open(out_path, "w") as f:
        json.dump({"n_students": n_students, "policy": AI_POLICY, "results": results}, f)
    print(f"Saved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return results


def run_analysis(non_ai_results, ai_results):
    """Compute aggregate analysis from all results."""
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}\n")

    all_results = non_ai_results + ai_results

    # Group by (learner_type, policy)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        groups[(r["learner_type"], r["policy"])].append(r)

    # Policy-level aggregates
    policy_stats = defaultdict(list)
    for r in all_results:
        policy_stats[r["policy"]].append(r)

    policy_summary = {}
    for policy, runs in sorted(policy_stats.items()):
        masteries = [r["final_mastery"] for r in runs]
        accuracies = [r["accuracy"] for r in runs]
        total_affect = sum(sum(r["affect"].values()) for r in runs)
        policy_summary[policy] = {
            "n": len(runs),
            "mean_mastery": round(float(np.mean(masteries)), 5),
            "std_mastery": round(float(np.std(masteries)), 5),
            "median_mastery": round(float(np.median(masteries)), 5),
            "mean_accuracy": round(float(np.mean(accuracies)), 4),
            "pct_frustrated": round(
                sum(r["affect"]["frustrated"] for r in runs) / max(1, total_affect), 4
            ) if total_affect > 0 else 0,
            "pct_engaged": round(
                sum(r["affect"]["engaged"] for r in runs) / max(1, total_affect), 4
            ) if total_affect > 0 else 0,
            "pct_bored": round(
                sum(r["affect"]["bored"] for r in runs) / max(1, total_affect), 4
            ) if total_affect > 0 else 0,
        }
        print(f"  {policy:20s}: mastery={policy_summary[policy]['mean_mastery']:.4f} "
              f"± {policy_summary[policy]['std_mastery']:.4f}  "
              f"(n={len(runs):,})")

    # Mean trajectory per policy (for line chart)
    trajectory_by_policy = {}
    for policy, runs in sorted(policy_stats.items()):
        trajs = [r["mastery_trajectory"] for r in runs]
        max_len = max(len(t) for t in trajs)
        padded = [t + [t[-1]] * (max_len - len(t)) for t in trajs]
        mean_traj = np.mean(padded, axis=0).tolist()
        trajectory_by_policy[policy] = [round(m, 5) for m in mean_traj]

    # Learning landscape: best policy per learner type prefix
    # Group by learner type base name (strip pop_XXXXX suffix for archetypes)
    # For population runs, bin by archetype-like clusters
    # Simpler: just report policy-level stats and per-policy trajectories

    analysis = {
        "policy_summary": policy_summary,
        "trajectory_by_policy": trajectory_by_policy,
        "total_trajectories": len(all_results),
    }

    out_path = RESULTS_DIR / "analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {out_path}")
    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Precompute simulation results")
    parser.add_argument("--non-ai-students", type=int, default=50000)
    parser.add_argument("--ai-students", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--skip-ai", action="store_true", help="Skip AI policy (run non-AI only)")
    parser.add_argument("--skip-non-ai", action="store_true", help="Skip non-AI (run AI only)")
    args = parser.parse_args()

    non_ai_results = []
    ai_results = []

    if not args.skip_non_ai:
        non_ai_results = run_non_ai(args.non_ai_students)

    if not args.skip_ai:
        ai_results = run_ai_parallel(args.ai_students, args.workers)

    # Load from disk if one was skipped
    if args.skip_non_ai and (RESULTS_DIR / "trajectories_non_ai.json").exists():
        with open(RESULTS_DIR / "trajectories_non_ai.json") as f:
            non_ai_results = json.load(f)["results"]
    if args.skip_ai and (RESULTS_DIR / "trajectories_ai.json").exists():
        with open(RESULTS_DIR / "trajectories_ai.json") as f:
            ai_results = json.load(f)["results"]

    if non_ai_results or ai_results:
        run_analysis(non_ai_results, ai_results)

    print("\n" + "=" * 60)
    print("PRECOMPUTE COMPLETE")
    print("=" * 60)
