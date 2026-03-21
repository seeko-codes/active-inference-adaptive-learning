import json
import os

from domain.knowledge_space import TIERS, generate_unified_hierarchy
from domain.render import render_problems, generate_summary, generate_pretty_browser, NUMBER_POOLS
from domain.axioms import SKILL_DESCRIPTIONS


if __name__ == "__main__":
    # Output directory: ../data/ relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    max_layer = 3

    print("KNOWLEDGE SPACE v2 - Active Inference Adaptive Learning")
    print("=" * 60)

    dataset, tier_summaries, ast_registry = generate_unified_hierarchy(TIERS, max_layer=max_layer)

    problems = render_problems(
        dataset,
        ast_registry,
        question_types=[
            "evaluate",
            "identify_property",
            "worked_example",
            "equivalent",
            "non_equivalent",
            "simplify",
            "expand",
            "find_error",
            "compare_methods",
            "sort_categorize",
        ],
        number_pool=NUMBER_POOLS["medium"],
        num_numeric_variants=5,
        rng_seed=42,
    )

    summary = generate_summary(dataset, tier_summaries)

    # Write outputs to ../data
    knowledge_space_path = os.path.join(data_dir, "knowledge_space.json")
    problems_path = os.path.join(data_dir, "student_problems.json")
    summary_path = os.path.join(data_dir, "knowledge_space_summary.json")
    browse_path = os.path.join(data_dir, "knowledge_space_browse.json")

    with open(knowledge_space_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    with open(problems_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pretty_output = generate_pretty_browser(dataset, problems)
    with open(browse_path, "w", encoding="utf-8") as f:
        json.dump(pretty_output, f, indent=2)

    # Report
    print(f"\n{'=' * 60}")
    print("KNOWLEDGE SPACE REPORT")
    print(f"{'=' * 60}")
    print(f"Total unique algebraic forms: {len(dataset)}")
    print(f"Total student problems:       {len(problems)}")
    print()

    print("HIERARCHY:")
    for ts in tier_summaries:
        prereq_str = f"requires Tier {ts['prerequisites']}" if ts["prerequisites"] else "entry point"
        print(f"  Tier {ts['tier']} ({ts['name']}) - {ts['total_forms']} forms - {prereq_str}")
        if ts.get("feature_ranges"):
            fr = ts["feature_ranges"]
            print(
                f"    Challenge: {fr['challenge_level']['min']:.3f}-{fr['challenge_level']['max']:.3f} "
                f"(mean {fr['challenge_level']['mean']:.3f})"
            )
            print(
                f"    Density:   {fr['density_load']['min']}-{fr['density_load']['max']} "
                f"(mean {fr['density_load']['mean']:.1f})"
            )
            print(
                f"    Transfer:  {fr['transfer_distance']['min']:.3f}-{fr['transfer_distance']['max']:.3f} "
                f"(mean {fr['transfer_distance']['mean']:.3f})"
            )
            print(f"    Scaffold:  {fr['scaffolding_demand']['min']:.3f}-{fr['scaffolding_demand']['max']:.3f}")
            print(f"    Diagnostic:{fr['diagnostic_value']['min']:.3f}-{fr['diagnostic_value']['max']:.3f}")
        if ts.get("cluster_distribution"):
            clusters = ", ".join(f"{k}: {v}" for k, v in ts["cluster_distribution"].items())
            print(f"    Clusters:  {clusters}")
            print(f"    Bridges:   {ts.get('bridge_count', 0)}")
    print()

    print("AXIOM COVERAGE:")
    for skill, count in summary["skill_coverage"].items():
        print(f"  {skill}: {count} forms")
    print()

    covered = set(summary["skill_coverage"].keys())
    all_axioms = set(SKILL_DESCRIPTIONS.keys())
    missing = all_axioms - covered
    print(f"Coverage: {len(covered)}/{len(all_axioms)} axioms")
    if missing:
        print(f"MISSING: {missing}")
    else:
        print("ALL AXIOMS COVERED")
    print()

    print("CLUSTER DISTRIBUTION:")
    for cluster, count in summary["cluster_distribution"].items():
        print(f"  {cluster:30s}: {count} forms")
    print(f"  Bridge problems: {summary['bridge_problems']} ({summary['bridge_ratio']:.1%} of total)")
    print()

    print("COGNITIVE FEATURE STATISTICS (across all forms):")
    for feat, stats in summary["cognitive_feature_statistics"].items():
        print(
            f"  {feat:25s}  min={stats['min']:.3f}  max={stats['max']:.3f}  "
            f"mean={stats['mean']:.3f}  median={stats['median']:.3f}"
        )
    print()

    print("FEATURE -> CURRICULUM VECTOR MAPPING:")
    for feat, desc in summary["feature_mapping_to_curriculum_vector"].items():
        print(f"  {feat:25s} -> {desc}")

    print(f"\n{'=' * 60}")
    print("SAMPLE ENTRY (first layer-2 problem):")
    print(f"{'=' * 60}")
    for e in dataset:
        if e["layer"] == 2:
            print(json.dumps(e, indent=2))
            break

    print("\nFiles written:")
    print(f"  {knowledge_space_path} ({len(dataset)} entries)")
    print(f"  {problems_path} ({len(problems)} entries)")
    print(f"  {summary_path}")
    print(f"  {browse_path} (human-readable browser)")
