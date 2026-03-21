"""
Export a representative sample of problems for LLM review.

Usage:
  python3 export_for_review.py              # curated sample (~200 problems)
  python3 export_for_review.py --full       # all 3,500+ problems
  python3 export_for_review.py --type boundary_test  # one type only
"""

import json
import random
import sys
from pathlib import Path

from domain.render import render_problems, render_supplementary_problems
from domain.knowledge_space import TIERS


def load_all():
    DATA_DIR = Path(__file__).parent / "data"
    ks_path = DATA_DIR / "knowledge_space.json"

    ast_registry = {}
    for tier_def in TIERS:
        for name, node in tier_def["seeds"].items():
            ast_registry[name] = node

    problems = []
    if ks_path.exists():
        with open(ks_path) as f:
            dataset = json.load(f)
        problems = render_problems(dataset, ast_registry, num_numeric_variants=3)
    problems.extend(render_supplementary_problems())
    return problems


def format_problem(p, idx):
    """Format a single problem for readable review."""
    lines = [f"### Problem {idx}"]
    lines.append(f"**Type:** {p.get('question_type')}  |  **Tier:** {p.get('tier')}  |  **Skills:** {', '.join(p.get('skills_tested', []))}")

    if p.get('question_type') == 'worked_example':
        lines.append(f"**Example:** {p.get('example_before')} = {p.get('example_after')}")
        if p.get('rule_demonstrated'):
            lines.append(f"**Rule:** {p.get('rule_demonstrated')}")
        if p.get('rule_description'):
            lines.append(f"**Description:** {p.get('rule_description')}")

    if p.get('question_type') == 'identify_property':
        lines.append(f"**Before:** {p.get('expression_before')}  →  **After:** {p.get('expression_after')}")

    if p.get('question_type') in ('equivalent', 'non_equivalent', 'boundary_test', 'custom_operation'):
        lines.append(f"**A:** {p.get('expression_a')}  vs  **B:** {p.get('expression_b')}")

    if p.get('question_type') == 'find_error':
        lines.append(f"**Original:** {p.get('original_expression')}  →  **Claimed:** {p.get('claimed_answer')}")

    if p.get('student_sees') and p.get('question_type') not in ('identify_property', 'equivalent', 'non_equivalent', 'boundary_test', 'custom_operation', 'worked_example', 'find_error'):
        lines.append(f"**Expression:** {p.get('student_sees')}")

    lines.append(f"**Prompt:** {p.get('prompt')}")

    if p.get('scaffolding', {}).get('hint'):
        lines.append(f"**Hint:** {p['scaffolding']['hint']}")

    if p.get('choices'):
        lines.append(f"**Choices:** {', '.join(str(c) for c in p['choices'])}")

    lines.append(f"**Expected Answer:** {p.get('expected_answer')}")

    if p.get('substitution'):
        lines.append(f"**Substitution:** {p['substitution']}")

    lines.append("")
    return "\n".join(lines)


def export_sample(problems, per_type=10):
    """Pick up to `per_type` problems per question type, spread across tiers."""
    rng = random.Random(42)
    by_type = {}
    for p in problems:
        qt = p.get("question_type", "?")
        by_type.setdefault(qt, []).append(p)

    selected = []
    for qt in sorted(by_type.keys()):
        pool = by_type[qt]
        # Spread across tiers
        by_tier = {}
        for p in pool:
            by_tier.setdefault(p.get("tier", 0), []).append(p)

        picks = []
        tiers = sorted(by_tier.keys())
        per_tier = max(1, per_type // max(1, len(tiers)))
        for t in tiers:
            sample = rng.sample(by_tier[t], min(per_tier, len(by_tier[t])))
            picks.extend(sample)

        if len(picks) > per_type:
            picks = rng.sample(picks, per_type)
        selected.extend(picks)

    return selected


def main():
    problems = load_all()

    # Parse args
    full = "--full" in sys.argv
    type_filter = None
    for i, arg in enumerate(sys.argv):
        if arg == "--type" and i + 1 < len(sys.argv):
            type_filter = sys.argv[i + 1]

    if type_filter:
        problems = [p for p in problems if p.get("question_type") == type_filter]
        selected = problems
        label = f"all {type_filter} problems"
    elif full:
        selected = problems
        label = "all problems"
    else:
        selected = export_sample(problems, per_type=10)
        label = "representative sample (10 per type)"

    # Stats
    types = {}
    for p in selected:
        qt = p.get("question_type", "?")
        types[qt] = types.get(qt, 0) + 1

    # Build output
    header = f"""# Adaptive Learning Problem Bank Review

**Source:** {len(problems)} total problems, exporting {len(selected)} ({label})
**Question types:** {len(types)}

## Stats
| Type | Count |
|------|-------|
"""
    for qt, count in sorted(types.items()):
        header += f"| {qt} | {count} |\n"

    header += f"""
## Skills Covered
The system tests 11 algebraic properties: A-Comm (additive commutativity), A-Assoc (additive associativity), M-Comm (multiplicative commutativity), M-Assoc (multiplicative associativity), Dist-Right/Dist-Left (distribution), Factor (reverse distribution), Sub-Def (subtraction as adding negatives), Div-Def (division as multiplying by reciprocal), A-Ident (additive identity), M-Ident (multiplicative identity).

## Review Criteria
For each problem, consider:
1. **Correctness** — Is the expected answer actually correct?
2. **Diagnostic value** — Does this problem actually test the claimed skill?
3. **Clarity** — Would a student understand what's being asked?
4. **Difficulty calibration** — Is the tier rating appropriate?
5. **Edge cases** — Are there inputs that could cause issues (division by zero, ambiguous answers)?
6. **Discrimination** — Does this problem distinguish understanding from memorization?
7. **AoPS standard** — For hard problems: would this hold up on an AoPS problem set?

---

"""

    body = ""
    for i, p in enumerate(selected, 1):
        body += format_problem(p, i) + "\n"

    output = header + body
    out_path = Path(__file__).parent / "data" / "problems_for_review.md"
    out_path.write_text(output)
    print(f"Exported {len(selected)} problems to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
