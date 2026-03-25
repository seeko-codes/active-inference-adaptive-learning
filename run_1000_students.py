#!/usr/bin/env python3
"""
Run 1000 synthetic AI students through the adaptive learning simulation.

Samples diverse learner profiles from empirically grounded distributions
(WM capacity, math anxiety, forgetting rate, etc.) and compares all
policies: meta_function, active_inference, random, fsrs_only, fixed_curriculum.

Usage:
    python run_1000_students.py                  # Full run (1000 students x 5 policies)
    python run_1000_students.py --students 100   # Quick test with 100 students
    python run_1000_students.py --no-ablation    # Skip ablation study
    python run_1000_students.py --output results.json  # Custom output path
"""

import argparse
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from simulation.monte_carlo import run_population_monte_carlo
from simulation.analysis import (
    run_full_analysis,
    compare_policies,
    format_comparison_table,
    learning_landscape,
    format_landscape,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic student simulation for policy comparison"
    )
    parser.add_argument(
        "--students", type=int, default=1000,
        help="Number of synthetic students to simulate (default: 1000)"
    )
    parser.add_argument(
        "--sessions", type=int, default=5,
        help="Learning sessions per student (default: 5)"
    )
    parser.add_argument(
        "--problems", type=int, default=20,
        help="Problems per session (default: 20)"
    )
    parser.add_argument(
        "--hours-between", type=float, default=24.0,
        help="Hours between sessions for forgetting (default: 24)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--policies", nargs="+", default=None,
        help="Policies to compare (default: all). Options: meta_function, active_inference, random, fsrs_only, fixed_curriculum"
    )
    parser.add_argument(
        "--no-ablation", action="store_true",
        help="Skip the state ablation study"
    )
    parser.add_argument(
        "--output", type=str, default="simulation_results.json",
        help="Output file path for JSON results (default: simulation_results.json)"
    )
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Adaptive Learning Simulation: {args.students} Synthetic Students")
    print(f"{'='*60}")
    print(f"  Sessions per student: {args.sessions}")
    print(f"  Problems per session: {args.problems}")
    print(f"  Hours between sessions: {args.hours_between}")
    print(f"  Policies: {args.policies or 'all'}")
    print(f"  Random seed: {args.seed}")
    print(f"{'='*60}\n")

    # Run simulation
    t0 = time.time()
    print("Running population Monte Carlo simulation...")
    results = run_population_monte_carlo(
        n_students=args.students,
        policy_names=args.policies,
        n_sessions=args.sessions,
        problems_per_session=args.problems,
        hours_between_sessions=args.hours_between,
        base_seed=args.seed,
    )
    sim_time = time.time() - t0
    print(f"\nSimulation complete: {len(results)} trajectories in {sim_time:.1f}s\n")

    # Run analysis
    t1 = time.time()
    analysis = run_full_analysis(
        results,
        run_ablation_study=not args.no_ablation,
        ablation_n_seeds=3,
    )
    analysis_time = time.time() - t1

    # Build JSON-serializable output
    comparisons = compare_policies(results)
    landscape = learning_landscape(comparisons)

    output = {
        "config": {
            "n_students": args.students,
            "n_sessions": args.sessions,
            "problems_per_session": args.problems,
            "hours_between_sessions": args.hours_between,
            "seed": args.seed,
            "policies": args.policies or "all",
        },
        "summary": {
            "total_trajectories": len(results),
            "simulation_time_seconds": round(sim_time, 1),
            "analysis_time_seconds": round(analysis_time, 1),
        },
        "policy_comparison": [
            {
                "learner_type": c.learner_type,
                "policy": c.policy,
                "n_runs": c.n_runs,
                "mean_final_mastery": round(c.mean_final_mastery, 4),
                "std_final_mastery": round(c.std_final_mastery, 4),
                "mean_accuracy": round(c.mean_accuracy, 4),
                "mean_time_engaged": round(c.mean_time_engaged, 4),
                "mean_time_frustrated": round(c.mean_time_frustrated, 4),
                "mean_time_bored": round(c.mean_time_bored, 4),
            }
            for c in comparisons
        ],
        "parameter_recovery": {
            k: round(v, 4) if isinstance(v, float) and not np.isnan(v) else v
            for k, v in analysis["recovery"].items()
        },
        "learning_landscape": {
            "best_policy": landscape["best_policy"],
            "advantage": landscape["advantage"],
            "vulnerability": landscape["vulnerability"],
        },
    }

    if "ablation" in analysis:
        output["ablation"] = {
            cond: {
                "mean_mastery": round(stats["mean_mastery"], 4),
                "std_mastery": round(stats["std_mastery"], 4),
            }
            for cond, stats in analysis["ablation"].items()
        }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print final summary
    print(f"\n{'='*60}")
    print(f"  DONE — {args.students} students x {len(set(r.policy for r in results))} policies")
    print(f"  Total time: {sim_time + analysis_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
