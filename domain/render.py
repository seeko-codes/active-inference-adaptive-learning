import random
from domain.ast_nodes import Op, pretty_print, substitute_numbers, evaluate
from domain.knowledge_space import TIERS
from domain.axioms import AXIOM_COGNITIVE_CLASS, SKILL_DESCRIPTIONS


# ==========================================
# 9. PROBLEM RENDERER
# ==========================================
NUMBER_POOLS = {
    "easy": [1, 2, 3, 4, 5, 6],
    "medium": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "hard": [2, 3, 5, 7, 11, 12, 15, 20, 25],
}

QUESTION_TEMPLATES = {
    "evaluate": "Evaluate the following expression:",
    "identify_property": "Which algebraic property was used to transform the first expression into the second?",
    "worked_example": "Study this step, then apply the same rule to a new expression:",
    "equivalent": "Are these two expressions equivalent? Explain why or why not.",
    "non_equivalent": "Are these two expressions equivalent? Explain why or why not.",
    "simplify": "Simplify the following expression:",
    "expand": "Expand the following expression:",
    "find_error": "This simplification contains an error. Find and correct it:",
    "compare_methods": "Which simplification path is more efficient, and why?",
    "sort_categorize": "Group these expressions by which ones are equivalent to each other:",
}

QUESTION_R_MULTIPLIER = {
    "worked_example": 0.2,
    "identify_property": 0.3,
    "equivalent": 0.4,
    "sort_categorize": 0.45,
    "non_equivalent": 0.5,
    "evaluate": 0.6,
    "expand": 0.8,
    "compare_methods": 0.85,
    "find_error": 0.9,
    "simplify": 1.0,
}


def render_problems(dataset, ast_registry, question_types=None, number_pool=None, num_numeric_variants=5, rng_seed=42):
    rng = random.Random(rng_seed)
    if question_types is None:
        question_types = list(QUESTION_TEMPLATES.keys())
    if number_pool is None:
        number_pool = NUMBER_POOLS["medium"]

    tier_variants = {
        0: num_numeric_variants + 3,
        1: num_numeric_variants + 2,
        2: num_numeric_variants + 1,
        3: num_numeric_variants,
        4: num_numeric_variants,
        5: num_numeric_variants - 1,
        6: max(2, num_numeric_variants - 2),
        7: max(2, num_numeric_variants - 2),
    }

    seed_asts = {}
    for tier_def in TIERS:
        for seed_name, seed_node in tier_def["seeds"].items():
            seed_asts[seed_name] = seed_node

    problems = []

    if "evaluate" in question_types:
        for entry in dataset:
            if entry["layer"] != 0:
                continue
            seed_node = seed_asts.get(entry["seed_name"])
            if not seed_node:
                continue

            variables = sorted(seed_node.get_variables())
            if not variables:
                continue

            cog = entry["cognitive_features"]
            tier = entry["tier"]
            n_vars = tier_variants.get(tier, num_numeric_variants)

            for vi in range(n_vars):
                var_map = {v: rng.choice(number_pool) for v in variables}
                numeric_node = substitute_numbers(seed_node, var_map)
                answer = evaluate(numeric_node)
                if answer is None:
                    continue
                if answer == int(answer):
                    answer = int(answer)
                elif abs(answer - round(answer, 2)) > 0.001:
                    continue  # Skip ugly repeating decimals

                sub_display = ", ".join(f"{v} = {var_map[v]}" for v in variables)
                problems.append({
                    "problem_id": f"{entry['problem_id']}_eval{vi}",
                    "tier": tier,
                    "tier_name": entry["tier_name"],
                    "layer": 0,
                    "global_difficulty": entry["global_difficulty"],
                    "prerequisite_tiers": entry["prerequisite_tiers"],
                    "skills_tested": [],
                    "derivation_path": [],
                    "seed_name": entry["seed_name"],
                    "cognitive_features": cog,
                    "question_type": "evaluate",
                    "retrieval_demand": round(cog["retrieval_base"] * QUESTION_R_MULTIPLIER["evaluate"], 3),
                    "prompt": f"If {sub_display}, evaluate:",
                    "student_sees": entry["pretty"],
                    "substitution": var_map,
                    "expected_answer": answer,
                })

    for entry in dataset:
        if entry["layer"] == 0:
            continue

        derivation = entry["derivation_path"]
        skill_set = entry["skill_set"]
        pretty_complex = entry["pretty"]
        pretty_simple = entry["seed_pretty"]
        seed_name = entry["seed_name"]
        cog = entry["cognitive_features"]
        tier = entry["tier"]

        base = {
            "problem_id": entry["problem_id"],
            "tier": tier,
            "tier_name": entry["tier_name"],
            "layer": entry["layer"],
            "global_difficulty": entry["global_difficulty"],
            "prerequisite_tiers": entry["prerequisite_tiers"],
            "skills_tested": skill_set,
            "derivation_path": derivation,
            "seed_name": seed_name,
            "cognitive_features": cog,
        }

        for qtype in question_types:
            r_demand = round(cog["retrieval_base"] * QUESTION_R_MULTIPLIER.get(qtype, 0.5), 3)

            if qtype == "simplify":
                problems.append({
                    **base,
                    "question_type": "simplify",
                    "retrieval_demand": r_demand,
                    "prompt": QUESTION_TEMPLATES["simplify"],
                    "student_sees": pretty_complex,
                    "expected_answer": pretty_simple,
                })

            elif qtype == "expand":
                if any(d in derivation for d in ("Dist-Right", "Dist-Left")):
                    problems.append({
                        **base,
                        "question_type": "expand",
                        "retrieval_demand": r_demand,
                        "prompt": QUESTION_TEMPLATES["expand"],
                        "student_sees": pretty_simple,
                        "expected_answer": pretty_complex,
                    })

            elif qtype == "identify_property":
                if entry["layer"] == 1 and derivation:
                    problems.append({
                        **base,
                        "question_type": "identify_property",
                        "retrieval_demand": r_demand,
                        "prompt": QUESTION_TEMPLATES["identify_property"],
                        "expression_before": pretty_simple,
                        "expression_after": pretty_complex,
                        "expected_answer": derivation[0],
                        "answer_description": SKILL_DESCRIPTIONS.get(derivation[0], ""),
                    })

            elif qtype == "equivalent":
                problems.append({
                    **base,
                    "question_type": "equivalent",
                    "retrieval_demand": r_demand,
                    "prompt": QUESTION_TEMPLATES["equivalent"],
                    "expression_a": pretty_simple,
                    "expression_b": pretty_complex,
                    "expected_answer": True,
                    "explanation": f"Connected by: {' -> '.join(derivation)}",
                })

            elif qtype == "non_equivalent":
                if entry["layer"] == 1 and tier >= 3:
                    if "Dist-Right" in derivation or "Dist-Left" in derivation:
                        seed_node = seed_asts.get(seed_name)
                        if seed_node and isinstance(seed_node, Op) and seed_node.op == "*":
                            inner = seed_node.right if seed_node.right else seed_node.left
                            if isinstance(inner, Op) and inner.op == "+":
                                wrong = Op("+", Op("*", seed_node.left.clone(), inner.left.clone()), inner.right.clone())
                                wrong_pretty = pretty_print(wrong)
                                problems.append({
                                    **base,
                                    "question_type": "non_equivalent",
                                    "retrieval_demand": r_demand,
                                    "prompt": QUESTION_TEMPLATES["non_equivalent"],
                                    "expression_a": pretty_simple,
                                    "expression_b": wrong_pretty,
                                    "expected_answer": False,
                                    "error_type": "partial_distribution",
                                    "explanation": f"{pretty_simple} != {wrong_pretty} - must distribute to ALL terms",
                                })

                    if "Sub-Def" in derivation:
                        wrong_pretty = pretty_simple.replace(" - ", " + ", 1)
                        if wrong_pretty != pretty_simple:
                            problems.append({
                                **base,
                                "question_type": "non_equivalent",
                                "retrieval_demand": r_demand,
                                "prompt": QUESTION_TEMPLATES["non_equivalent"],
                                "expression_a": pretty_simple,
                                "expression_b": wrong_pretty,
                                "expected_answer": False,
                                "error_type": "sign_error",
                                "explanation": "Subtraction != addition",
                            })

            elif qtype == "evaluate":
                seed_node = seed_asts.get(seed_name)
                if seed_node:
                    variables = sorted(seed_node.get_variables())
                    n_vars = tier_variants.get(tier, num_numeric_variants)
                    for vi in range(n_vars):
                        var_map = {v: rng.choice(number_pool) for v in variables}
                        numeric_node = substitute_numbers(seed_node, var_map)
                        answer = evaluate(numeric_node)
                        if answer is None:
                            continue
                        if answer == int(answer):
                            answer = int(answer)
                        elif abs(answer - round(answer, 2)) > 0.001:
                            continue  # Skip ugly repeating decimals
                        sub_display = ", ".join(f"{v} = {var_map[v]}" for v in variables)
                        problems.append({
                            **base,
                            "question_type": "evaluate",
                            "retrieval_demand": r_demand,
                            "problem_id": f"{entry['problem_id']}_num{vi}",
                            "prompt": f"If {sub_display}, evaluate:",
                            "student_sees": pretty_complex,
                            "substitution": var_map,
                            "expected_answer": answer,
                        })

            elif qtype == "worked_example":
                if entry["layer"] == 1 and derivation:
                    axiom_used = derivation[0]
                    problems.append({
                        **base,
                        "question_type": "worked_example",
                        "retrieval_demand": r_demand,
                        "prompt": QUESTION_TEMPLATES["worked_example"],
                        "example_before": pretty_simple,
                        "example_after": pretty_complex,
                        "rule_demonstrated": axiom_used,
                        "rule_description": SKILL_DESCRIPTIONS.get(axiom_used, ""),
                    })

            elif qtype == "find_error":
                if entry["layer"] >= 1:
                    errors = _generate_error_variants(
                        entry, derivation, skill_set, pretty_complex, pretty_simple,
                        seed_name, seed_asts, base, r_demand
                    )
                    problems.extend(errors)

            elif qtype == "compare_methods":
                if entry["layer"] == 2 and len(derivation) == 2:
                    path_a = " -> ".join(derivation)
                    path_b = " -> ".join(reversed(derivation))
                    if path_a != path_b:
                        problems.append({
                            **base,
                            "question_type": "compare_methods",
                            "retrieval_demand": r_demand,
                            "prompt": QUESTION_TEMPLATES["compare_methods"],
                            "expression": pretty_complex,
                            "target": pretty_simple,
                            "path_a": path_a,
                            "path_b": path_b,
                            "path_a_steps": derivation,
                            "path_b_steps": list(reversed(derivation)),
                        })

    if "sort_categorize" in question_types:
        seed_families = {}
        for e in dataset:
            seed_families.setdefault(e["seed_name"], []).append(e)

        families_by_tier = {}
        for sn, entries in seed_families.items():
            t = entries[0]["tier"]
            families_by_tier.setdefault(t, []).append(sn)

        sort_id = 0
        for tier_num, tier_families in families_by_tier.items():
            if len(tier_families) < 2:
                continue

            for i in range(len(tier_families)):
                for j in range(i + 1, len(tier_families)):
                    fam_a_name = tier_families[i]
                    fam_b_name = tier_families[j]
                    fam_a = seed_families[fam_a_name]
                    fam_b = seed_families[fam_b_name]

                    pick_a = [e for e in fam_a if e["layer"] <= 2][:3]
                    pick_b = [e for e in fam_b if e["layer"] <= 2][:3]
                    if len(pick_a) < 2 or len(pick_b) < 2:
                        continue

                    items, group_a_ids, group_b_ids = [], [], []

                    for e in pick_a:
                        item_id = f"item_{len(items)}"
                        items.append({"id": item_id, "expression": e["pretty"]})
                        group_a_ids.append(item_id)

                    for e in pick_b:
                        item_id = f"item_{len(items)}"
                        items.append({"id": item_id, "expression": e["pretty"]})
                        group_b_ids.append(item_id)

                    rng.shuffle(items)

                    ref_entry = pick_a[-1] if pick_a[-1]["layer"] >= pick_b[-1]["layer"] else pick_b[-1]
                    cog = ref_entry["cognitive_features"]
                    r_demand = round(cog["retrieval_base"] * QUESTION_R_MULTIPLIER["sort_categorize"], 3)

                    problems.append({
                        "problem_id": f"sort_{tier_num}_{sort_id}",
                        "tier": tier_num,
                        "tier_name": ref_entry["tier_name"],
                        "layer": max(e["layer"] for e in pick_a + pick_b),
                        "global_difficulty": cog["challenge_level"],
                        "prerequisite_tiers": ref_entry["prerequisite_tiers"],
                        "skills_tested": sorted(set(s for e in pick_a + pick_b for s in e["skill_set"])),
                        "derivation_path": [],
                        "seed_name": f"{fam_a_name}+{fam_b_name}",
                        "cognitive_features": cog,
                        "question_type": "sort_categorize",
                        "retrieval_demand": r_demand,
                        "prompt": QUESTION_TEMPLATES["sort_categorize"],
                        "items": items,
                        "expected_groups": [
                            {"group": f"Equivalent to {pick_a[0]['pretty']}", "item_ids": group_a_ids},
                            {"group": f"Equivalent to {pick_b[0]['pretty']}", "item_ids": group_b_ids},
                        ],
                        "family_a": fam_a_name,
                        "family_b": fam_b_name,
                        "tests_construct": "K_org",
                    })
                    sort_id += 1

    return problems


def _generate_error_variants(entry, derivation, skill_set, pretty_complex, pretty_simple, seed_name, seed_asts, base, r_demand):
    errors = []

    if any(ax in derivation for ax in ("Dist-Right", "Dist-Left")):
        seed_node = seed_asts.get(seed_name)
        if seed_node and isinstance(seed_node, Op) and seed_node.op == "*":
            inner = (
                seed_node.right if isinstance(seed_node.right, Op) and seed_node.right.op == "+"
                else (seed_node.left if isinstance(seed_node.left, Op) and seed_node.left.op == "+" else None)
            )
            outer = seed_node.left if inner == seed_node.right else seed_node.right
            if inner and outer:
                wrong = Op("+", Op("*", outer.clone(), inner.left.clone()), inner.right.clone())
                # Correct answer is always the fully distributed form
                correct = Op("+", Op("*", outer.clone(), inner.left.clone()),
                             Op("*", outer.clone(), inner.right.clone()))
                errors.append({
                    **base,
                    "question_type": "find_error",
                    "retrieval_demand": r_demand,
                    "prompt": QUESTION_TEMPLATES["find_error"],
                    "original_expression": pretty_simple,
                    "student_sees": f"{pretty_simple} = {pretty_print(wrong)}",
                    "claimed_answer": pretty_print(wrong),
                    "correct_answer": pretty_print(correct),
                    "error_type": "partial_distribution",
                    "error_description": "Only distributed to the first term, not both",
                })

    if "Sub-Def" in derivation and entry["layer"] == 1:
        # Generate a concrete wrong step: dropping the negative sign
        # e.g., a - b should become a + (-b), wrong version: a + b
        seed_node = seed_asts.get(seed_name)
        if seed_node and isinstance(seed_node, Op) and seed_node.op == "-":
            wrong_expr = Op("+", seed_node.left.clone(), seed_node.right.clone())
            errors.append({
                **base,
                "question_type": "find_error",
                "retrieval_demand": r_demand,
                "prompt": QUESTION_TEMPLATES["find_error"],
                "original_expression": pretty_simple,
                "student_sees": f"{pretty_simple} = {pretty_print(wrong_expr)}",
                "claimed_answer": pretty_print(wrong_expr),
                "correct_answer": pretty_complex,
                "error_type": "sign_error",
                "error_description": "Dropped the negative sign when converting subtraction to addition",
            })

    if "Sub-Def" in skill_set and "A-Comm" in skill_set and entry["layer"] == 1:
        seed_node = seed_asts.get(seed_name)
        if seed_node and isinstance(seed_node, Op) and seed_node.op == "-":
            # Concrete: a - b claimed equal to b - a
            swapped = Op("-", seed_node.right.clone(), seed_node.left.clone())
            errors.append({
                **base,
                "question_type": "find_error",
                "retrieval_demand": r_demand,
                "prompt": "A student claims these are equal because subtraction is commutative. Find the error:",
                "original_expression": pretty_simple,
                "student_sees": f"{pretty_simple} = {pretty_print(swapped)}",
                "claimed_answer": pretty_print(swapped),
                "correct_answer": f"Subtraction is NOT commutative. {pretty_simple} != {pretty_print(swapped)}",
                "error_type": "commutativity_overgeneralization",
                "error_description": "Applied commutativity to subtraction — only works for addition",
            })

    return errors


# ==========================================
# 10. SUMMARY GENERATOR
# ==========================================
def generate_summary(dataset, tier_summaries):
    layers = {}
    tiers_count = {}
    skill_coverage = {}
    skill_combos = {}

    all_features = {
        "density_load": [],
        "challenge_level": [],
        "scaffolding_demand": [],
        "transfer_distance": [],
        "transfer_structural": [],
        "transfer_surface": [],
        "cross_cluster_demand": [],
        "retrieval_base": [],
        "spacing_sensitivity": [],
        "diagnostic_value": [],
    }

    cluster_counts = {}
    bridge_count = 0

    for entry in dataset:
        layer = entry["layer"]
        tier = entry["tier"]

        layers[layer] = layers.get(layer, 0) + 1
        tiers_count[tier] = tiers_count.get(tier, 0) + 1

        for skill in entry["skill_set"]:
            skill_coverage[skill] = skill_coverage.get(skill, 0) + 1

        combo = " + ".join(entry["skill_set"]) if entry["skill_set"] else "(seed)"
        skill_combos[combo] = skill_combos.get(combo, 0) + 1

        cog = entry.get("cognitive_features", {})
        for key in all_features:
            if key in cog:
                all_features[key].append(cog[key])

        pc = cog.get("primary_cluster", "unknown")
        cluster_counts[pc] = cluster_counts.get(pc, 0) + 1
        if cog.get("is_bridge", False):
            bridge_count += 1

    feature_stats = {}
    for key, values in all_features.items():
        if values:
            sv = sorted(values)
            feature_stats[key] = {
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "mean": round(sum(values) / len(values), 3),
                "median": round(sv[len(sv) // 2], 3),
            }

    return {
        "total_unique_forms": len(dataset),
        "forms_by_tier": dict(sorted(tiers_count.items())),
        "forms_by_layer": dict(sorted(layers.items())),
        "skill_coverage": dict(sorted(skill_coverage.items(), key=lambda x: -x[1])),
        "unique_skill_combinations": len(skill_combos),
        "skill_combination_detail": dict(sorted(skill_combos.items(), key=lambda x: -x[1])),
        "cognitive_feature_statistics": feature_stats,
        "cluster_distribution": dict(sorted(cluster_counts.items(), key=lambda x: -x[1])),
        "bridge_problems": bridge_count,
        "bridge_ratio": round(bridge_count / max(1, len(dataset)), 3),
        "tier_hierarchy": tier_summaries,
        "skill_descriptions": SKILL_DESCRIPTIONS,
        "axiom_cognitive_classes": AXIOM_COGNITIVE_CLASS,
        "prerequisite_graph": {
            "Tier 0 (Raw Arithmetic)": "No prerequisites - entry point.",
            "Tier 1 (Commutativity)": "Requires Tier 0.",
            "Tier 2 (Associativity)": "Requires Tier 1.",
            "Tier 3 (PEMA / Order of Ops)": "Requires Tier 1 + 2.",
            "Tier 4 (Distribution)": "Requires Tier 1 + 2 + 3.",
            "Tier 5 (Signed Ops / PEMA)": "Requires Tier 1.",
            "Tier 6 (Signed Distribution)": "Requires Tier 4 + 5.",
            "Tier 7 (Complex Combinations)": "Requires Tier 4 + 5 + 6.",
        },
        "feature_mapping_to_curriculum_vector": {
            "density_load": "D",
            "scaffolding_demand": "I",
            "transfer_distance": "V",
            "transfer_structural": "V",
            "transfer_surface": "V",
            "primary_cluster": "V",
            "cross_cluster_demand": "V",
            "retrieval_base": "R",
            "challenge_level": "C",
            "feedback_ceiling": "F",
            "spacing_sensitivity": "S",
        },
    }


# ==========================================
# 11. PRETTY BROWSER
# ==========================================
def generate_pretty_browser(dataset, problems):
    by_tier = {}
    for e in dataset:
        tier_key = f"Tier {e['tier']}: {e['tier_name']}"
        if tier_key not in by_tier:
            by_tier[tier_key] = {"seeds": [], "derived": {}}

        if e["layer"] == 0:
            by_tier[tier_key]["seeds"].append({
                "expression": e["pretty"],
                "seed_name": e["seed_name"],
            })
        else:
            layer_key = f"Layer {e['layer']}"
            by_tier[tier_key]["derived"].setdefault(layer_key, []).append({
                "expression": e["pretty"],
                "from_seed": e["seed_pretty"],
                "via": " -> ".join(e["derivation_path"]),
                "skills": e["skill_set"],
                "cluster": e["cognitive_features"]["primary_cluster"],
                "bridge": e["cognitive_features"]["is_bridge"],
                "challenge": e["cognitive_features"]["challenge_level"],
                "transfer": e["cognitive_features"]["transfer_distance"],
                "density": e["cognitive_features"]["density_load"],
            })

    for tier_key in by_tier:
        for layer_key in by_tier[tier_key]["derived"]:
            by_tier[tier_key]["derived"][layer_key].sort(key=lambda x: x["challenge"])

    by_cluster = {}
    for e in dataset:
        cluster = e["cognitive_features"]["primary_cluster"]
        by_cluster.setdefault(cluster, []).append({
            "expression": e["pretty"],
            "tier": e["tier"],
            "layer": e["layer"],
            "challenge": e["cognitive_features"]["challenge_level"],
            "bridge": e["cognitive_features"]["is_bridge"],
        })

    for cluster in by_cluster:
        by_cluster[cluster].sort(key=lambda x: (x["tier"], x["layer"], x["challenge"]))

    by_seed = {}
    for e in dataset:
        seed = e["seed_name"]
        by_seed.setdefault(seed, {"canonical": None, "forms": []})
        if e["layer"] == 0:
            by_seed[seed]["canonical"] = e["pretty"]
        else:
            by_seed[seed]["forms"].append({
                "expression": e["pretty"],
                "layer": e["layer"],
                "via": " -> ".join(e["derivation_path"]),
                "transfer": e["cognitive_features"]["transfer_distance"],
            })

    for seed in by_seed:
        by_seed[seed]["forms"].sort(key=lambda x: (x["layer"], x["transfer"]))

    by_qtype = {}
    for p in problems:
        qtype = p["question_type"]
        by_qtype.setdefault(qtype, {"count": 0, "samples": []})
        by_qtype[qtype]["count"] += 1

        if len(by_qtype[qtype]["samples"]) < 5:
            sample = {"tier": p["tier"], "layer": p["layer"]}

            if qtype == "simplify":
                sample["prompt"] = f"Simplify: {p['student_sees']}"
                sample["answer"] = p["expected_answer"]
            elif qtype == "expand":
                sample["prompt"] = f"Expand: {p['student_sees']}"
                sample["answer"] = p["expected_answer"]
            elif qtype == "identify_property":
                sample["prompt"] = f"What property? {p['expression_before']} -> {p['expression_after']}"
                sample["answer"] = f"{p['expected_answer']} ({p.get('answer_description', '')})"
            elif qtype == "equivalent":
                sample["prompt"] = f"Equivalent? {p['expression_a']} vs {p['expression_b']}"
                sample["answer"] = p["expected_answer"]
            elif qtype == "non_equivalent":
                sample["prompt"] = f"Equivalent? {p['expression_a']} vs {p['expression_b']}"
                sample["answer"] = f"No - {p.get('error_type', 'different')}"
            elif qtype == "evaluate":
                sample["prompt"] = f"{p['prompt']} {p['student_sees']}"
                sample["answer"] = p["expected_answer"]
            elif qtype == "worked_example":
                sample["prompt"] = f"Study: {p.get('example_before', '')} -> {p.get('example_after', '')} ({p.get('rule_demonstrated', '')})"
                sample["answer"] = f"Apply {p.get('rule_description', '')}"
            elif qtype == "find_error":
                sample["prompt"] = f"Find error: {p.get('student_sees', p.get('original_expression', ''))}"
                sample["answer"] = f"Error: {p.get('error_type', '?')}"
            elif qtype == "compare_methods":
                sample["prompt"] = f"Compare: {p.get('path_a', '')} vs {p.get('path_b', '')}"
                sample["answer"] = f"Paths for {p.get('expression', '')}"
            elif qtype == "sort_categorize":
                items_str = ", ".join(it["expression"] for it in p.get("items", []))
                sample["prompt"] = f"Group equivalent: [{items_str}]"
                sample["answer"] = "Grouping expected"
            else:
                sample["prompt"] = str(p.get("prompt", ""))
                sample["answer"] = str(p.get("expected_answer", ""))

            sample["skills"] = p.get("skills_tested", [])
            sample["challenge"] = p.get("cognitive_features", {}).get("challenge_level", 0)
            sample["retrieval"] = p.get("retrieval_demand", 0)
            by_qtype[qtype]["samples"].append(sample)

    return {
        "_description": "Human-readable view of the knowledge space.",
        "by_tier": by_tier,
        "by_cluster": by_cluster,
        "by_seed_family": by_seed,
        "by_question_type": by_qtype,
    }


# ==========================================
# 12. SUPPLEMENTARY PROBLEMS
# ==========================================
# Problems for comprehensive schema building that don't come from
# axiom rewrites. These test boundaries, strategic application,
# inverse connections (PEMA), order of operations, and property
# discrimination in multi-step contexts.

def render_supplementary_problems(rng_seed=42):
    """Generate problems for comprehensive schema building beyond axiom rewrites."""
    rng = random.Random(rng_seed)
    problems = []
    problems.extend(_render_boundary_tests(rng))
    problems.extend(_render_strategic_compute(rng))
    problems.extend(_render_inverse_rewrite(rng))
    problems.extend(_render_order_of_ops(rng))
    problems.extend(_render_property_chains(rng))
    problems.extend(_render_custom_operations(rng))
    problems.extend(_render_competition_problems(rng))
    problems.extend(_render_fill_in_blank(rng))
    problems.extend(_render_parentheses_placement(rng))
    problems.extend(_render_proof_disproof(rng))
    return problems


def _supp_base(pid, tier, qtype, skills):
    """Shared fields for supplementary problems."""
    return {
        "problem_id": pid,
        "tier": tier,
        "tier_name": "",
        "layer": 0,
        "global_difficulty": tier / 7,
        "prerequisite_tiers": [],
        "skills_tested": skills,
        "derivation_path": skills,
        "seed_name": "supplementary",
        "cognitive_features": {
            "density_load": tier + 2,
            "challenge_level": tier / 7,
            "retrieval_base": 0.3,
            "diagnostic_value": 0.7,
            "primary_cluster": "supplementary",
            "is_bridge": False,
        },
        "question_type": qtype,
    }


def _render_boundary_tests(rng):
    """Where do commutativity and associativity break?

    These are the most diagnostic questions: students who only memorize
    'you can swap' will incorrectly apply it to subtraction/division.
    Students with real schemas know WHY it works for +/* and fails for -/÷.
    """
    problems = []
    pool = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]

    # --- Commutativity boundary ---
    for i in range(6):
        a, b = rng.sample(pool, 2)

        # Addition IS commutative (true)
        problems.append({
            **_supp_base(f"bnd_comm_add_t{i}", 1, "boundary_test", ["A-Comm"]),
            "prompt": "Is addition commutative? Do these always give the same result?",
            "expression_a": f"{a} + {b}",
            "expression_b": f"{b} + {a}",
            "expected_answer": True,
        })

        # Subtraction is NOT commutative (false)
        problems.append({
            **_supp_base(f"bnd_comm_sub_f{i}", 5, "boundary_test", ["Sub-Def", "A-Comm"]),
            "prompt": "Is subtraction commutative? Do these always give the same result?",
            "expression_a": f"{a} - {b}",
            "expression_b": f"{b} - {a}",
            "expected_answer": False,
        })

        # Multiplication IS commutative (true)
        problems.append({
            **_supp_base(f"bnd_comm_mul_t{i}", 1, "boundary_test", ["M-Comm"]),
            "prompt": "Is multiplication commutative? Do these always give the same result?",
            "expression_a": f"{a} * {b}",
            "expression_b": f"{b} * {a}",
            "expected_answer": True,
        })

        # Division is NOT commutative (false)
        problems.append({
            **_supp_base(f"bnd_comm_div_f{i}", 5, "boundary_test", ["Div-Def", "M-Comm"]),
            "prompt": "Is division commutative? Do these always give the same result?",
            "expression_a": f"{a} / {b}",
            "expression_b": f"{b} / {a}",
            "expected_answer": False,
        })

    # --- Associativity boundary ---
    for i in range(6):
        a, b, c = rng.sample(pool, 3)

        # Addition IS associative (true)
        problems.append({
            **_supp_base(f"bnd_assoc_add_t{i}", 2, "boundary_test", ["A-Assoc"]),
            "prompt": "Is addition associative? Does regrouping change the result?",
            "expression_a": f"({a} + {b}) + {c}",
            "expression_b": f"{a} + ({b} + {c})",
            "expected_answer": True,
        })

        # Subtraction is NOT associative (false)
        problems.append({
            **_supp_base(f"bnd_assoc_sub_f{i}", 5, "boundary_test", ["Sub-Def", "A-Assoc"]),
            "prompt": "Is subtraction associative? Does regrouping change the result?",
            "expression_a": f"({a} - {b}) - {c}",
            "expression_b": f"{a} - ({b} - {c})",
            "expected_answer": False,
        })

        # Multiplication IS associative (true)
        a2, b2, c2 = rng.sample([2, 3, 4, 5], 3)
        problems.append({
            **_supp_base(f"bnd_assoc_mul_t{i}", 2, "boundary_test", ["M-Assoc"]),
            "prompt": "Is multiplication associative? Does regrouping change the result?",
            "expression_a": f"({a2} * {b2}) * {c2}",
            "expression_b": f"{a2} * ({b2} * {c2})",
            "expected_answer": True,
        })

        # Division is NOT associative (false)
        problems.append({
            **_supp_base(f"bnd_assoc_div_f{i}", 5, "boundary_test", ["Div-Def", "M-Assoc"]),
            "prompt": "Is division associative? Does regrouping change the result?",
            "expression_a": f"({a} / {b}) / {c}",
            "expression_b": f"{a} / ({b} / {c})",
            "expected_answer": False,
        })

    return problems


def _render_strategic_compute(rng):
    """Problems requiring strategic application of properties.

    AoPS-style: the naive left-to-right approach works but is hard.
    The smart approach recognizes pairings/rearrangements.
    """
    problems = []

    # Pre-designed addition problems with nice pairings
    add_problems = [
        ("17 + 45 + 83 + 55", 200, "Pair 17+83=100 and 45+55=100"),
        ("28 + 37 + 72 + 63", 200, "Pair 28+72=100 and 37+63=100"),
        ("15 + 27 + 85 + 73", 200, "Pair 15+85=100 and 27+73=100"),
        ("36 + 48 + 64 + 52", 200, "Pair 36+64=100 and 48+52=100"),
        ("19 + 56 + 81 + 44", 200, "Pair 19+81=100 and 56+44=100"),
        ("23 + 38 + 77 + 62", 200, "Pair 23+77=100 and 38+62=100"),
    ]

    for idx, (expr, ans, strategy) in enumerate(add_problems):
        problems.append({
            **_supp_base(f"strat_add_{idx}", 2, "strategic_compute", ["A-Comm", "A-Assoc"]),
            "prompt": "Compute this efficiently. Can you find a clever way to group the numbers?",
            "student_sees": expr,
            "expected_answer": ans,
            "scaffolding": {"hint": "Look for pairs that add to a round number."},
        })

    # Multiplication reordering
    mult_problems = [
        ("4 * 13 * 25", 1300, "Reorder: 4*25=100, then *13"),
        ("2 * 37 * 50", 3700, "Reorder: 2*50=100, then *37"),
        ("5 * 19 * 20", 1900, "Reorder: 5*20=100, then *19"),
        ("25 * 7 * 4", 700, "Reorder: 25*4=100, then *7"),
        ("50 * 23 * 2", 2300, "Reorder: 50*2=100, then *23"),
        ("8 * 17 * 125", 17000, "Reorder: 8*125=1000, then *17"),
    ]

    for idx, (expr, ans, strategy) in enumerate(mult_problems):
        problems.append({
            **_supp_base(f"strat_mul_{idx}", 2, "strategic_compute", ["M-Comm", "M-Assoc"]),
            "prompt": "Compute this efficiently. Can you reorder the factors to make it easier?",
            "student_sees": expr,
            "expected_answer": ans,
            "scaffolding": {"hint": "Look for factor pairs that give a round number."},
        })

    # Factoring for efficiency (distribution in reverse)
    factor_problems = [
        ("7 * 13 + 7 * 87", 700, "Factor: 7*(13+87) = 7*100"),
        ("3 * 47 + 3 * 53", 300, "Factor: 3*(47+53) = 3*100"),
        ("9 * 82 + 9 * 18", 900, "Factor: 9*(82+18) = 9*100"),
        ("6 * 35 + 6 * 65", 600, "Factor: 6*(35+65) = 6*100"),
        ("11 * 57 + 11 * 43", 1100, "Factor: 11*(57+43) = 11*100"),
    ]

    for idx, (expr, ans, strategy) in enumerate(factor_problems):
        problems.append({
            **_supp_base(f"strat_factor_{idx}", 4, "strategic_compute", ["Factor"]),
            "prompt": "Compute this efficiently. Look for a common factor.",
            "student_sees": expr,
            "expected_answer": ans,
            "scaffolding": {"hint": "Both terms share a common factor. Can you pull it out?"},
        })

    # Gauss-style sequential sums
    gauss_problems = [
        (10, 55, "Pair: (1+10)+(2+9)+...+(5+6) = 5 pairs of 11"),
        (20, 210, "Pair: 10 pairs of 21"),
        (50, 1275, "Pair: 25 pairs of 51"),
        (100, 5050, "Pair: 50 pairs of 101"),
    ]

    for idx, (n, ans, strategy) in enumerate(gauss_problems):
        problems.append({
            **_supp_base(f"strat_gauss_{idx}", 2, "strategic_compute", ["A-Comm", "A-Assoc"]),
            "prompt": f"Find the sum of all integers from 1 to {n}. Can you find a pattern instead of adding one by one?",
            "student_sees": f"1 + 2 + 3 + ... + {n}",
            "expected_answer": ans,
            "scaffolding": {"hint": f"Try pairing the first and last numbers: 1 + {n} = {n + 1}. How many such pairs?"},
        })

    return problems


def _render_inverse_rewrite(rng):
    """Connect subtraction → addition and division → multiplication.

    This is the PEMA insight: once students see subtraction as 'adding negatives,'
    they understand WHY commutativity applies to terms-with-signs and WHY
    PEMDAS simplifies to PEMA.
    """
    problems = []
    pool = [2, 3, 4, 5, 6, 7, 8, 9]

    # --- Subtraction → addition of negatives ---
    for i in range(6):
        a, b = rng.sample(pool, 2)
        problems.append({
            **_supp_base(f"inv_sub_basic_{i}", 5, "inverse_rewrite", ["Sub-Def"]),
            "prompt": "Rewrite this using only addition (replace subtraction with adding a negative):",
            "student_sees": f"{a} - {b}",
            "expected_answer": f"{a} + (-{b})",
        })

    for i in range(6):
        a, b, c, d = rng.sample(pool, 4)
        problems.append({
            **_supp_base(f"inv_sub_chain_{i}", 5, "inverse_rewrite", ["Sub-Def", "A-Comm"]),
            "prompt": "Rewrite using only addition. Then: can you rearrange the terms?",
            "student_sees": f"{a} - {b} + {c} - {d}",
            "expected_answer": f"{a} + (-{b}) + {c} + (-{d})",
        })

    # Symbolic subtraction rewrite
    problems.append({
        **_supp_base("inv_sub_sym_0", 5, "inverse_rewrite", ["Sub-Def"]),
        "prompt": "Rewrite using only addition:",
        "student_sees": "a - b",
        "expected_answer": "a + (-b)",
    })
    problems.append({
        **_supp_base("inv_sub_sym_1", 5, "inverse_rewrite", ["Sub-Def", "A-Comm"]),
        "prompt": "Rewrite using only addition, then rearrange:",
        "student_sees": "a - b + c",
        "expected_answer": "a + (-b) + c",
    })

    # --- Division → multiplication by reciprocal ---
    for i in range(6):
        a, b = rng.sample(pool, 2)
        problems.append({
            **_supp_base(f"inv_div_basic_{i}", 5, "inverse_rewrite", ["Div-Def"]),
            "prompt": "Rewrite this using only multiplication (replace division with multiplying by the reciprocal):",
            "student_sees": f"{a} / {b}",
            "expected_answer": f"{a} * (1/{b})",
        })

    # Symbolic division rewrite
    problems.append({
        **_supp_base("inv_div_sym_0", 5, "inverse_rewrite", ["Div-Def"]),
        "prompt": "Rewrite using only multiplication:",
        "student_sees": "a / b",
        "expected_answer": "a * (1/b)",
    })

    # --- The PEMA insight: WHY properties now apply ---
    for i in range(4):
        a, b, c = rng.sample(pool, 3)
        # After rewriting subtraction as addition, commutativity applies
        problems.append({
            **_supp_base(f"inv_pema_add_{i}", 5, "inverse_rewrite", ["Sub-Def", "A-Comm"]),
            "prompt": (
                f"We know {a} - {b} + {c} = {a} + (-{b}) + {c}. "
                "Since this is now ALL addition, commutativity lets us rearrange. "
                "Rearrange to compute more easily:"
            ),
            "student_sees": f"{a} + (-{b}) + {c}",
            "expected_answer": f"{a} + {c} + (-{b})",
        })

    for i in range(4):
        a, b, c = rng.sample([2, 3, 4, 5, 6], 3)
        problems.append({
            **_supp_base(f"inv_pema_mul_{i}", 5, "inverse_rewrite", ["Div-Def", "M-Comm"]),
            "prompt": (
                f"We know {a} / {b} * {c} = {a} * (1/{b}) * {c}. "
                "Since this is now ALL multiplication, commutativity lets us reorder. "
                "Reorder to compute more easily:"
            ),
            "student_sees": f"{a} * (1/{b}) * {c}",
            "expected_answer": f"{a} * {c} * (1/{b})",
        })

    return problems


def _render_order_of_ops(rng):
    """Explicit PEMA / order of operations testing.

    Tests whether students apply multiplication before addition
    without extra parentheses. Common error: reading left-to-right
    and adding before multiplying.
    """
    problems = []
    pool = [2, 3, 4, 5, 6, 7, 8, 9]

    # --- Evaluation problems (PEMDAS traps) ---
    ooo_templates = [
        # (expression_str, correct_answer, common_wrong_answer, description)
        lambda a, b, c: (f"{a} + {b} * {c}", a + b * c, (a + b) * c, "multiply before add"),
        lambda a, b, c: (f"{a} * {b} + {c}", a * b + c, a * (b + c), "multiply before add"),
        lambda a, b, c: (f"{a} - {b} * {c}", a - b * c, (a - b) * c, "multiply before subtract"),
        lambda a, b, c: (f"{a} * {b} - {c}", a * b - c, a * (b - c), "multiply before subtract"),
    ]

    for i in range(5):
        a, b, c = rng.sample(pool, 3)
        for tidx, template in enumerate(ooo_templates):
            expr, correct, wrong, desc = template(a, b, c)
            if correct < 0:
                continue
            problems.append({
                **_supp_base(f"ooo_eval_{i}_{tidx}", 3, "order_of_ops", []),
                "prompt": "Evaluate this expression. Remember: multiplication and division come before addition and subtraction.",
                "student_sees": expr,
                "expected_answer": correct,
            })

    # --- Parentheses change the result ---
    for i in range(5):
        a, b, c = rng.sample(pool, 3)

        # a + b * c  vs  (a + b) * c
        val_no_parens = a + b * c
        val_with_parens = (a + b) * c
        if val_no_parens != val_with_parens:
            problems.append({
                **_supp_base(f"ooo_parens_add_{i}", 3, "boundary_test", []),
                "prompt": "Do these give the same result? Does the placement of parentheses matter?",
                "expression_a": f"{a} + {b} * {c}",
                "expression_b": f"({a} + {b}) * {c}",
                "expected_answer": False,
            })

        # a * b + c  vs  a * (b + c)
        val_no = a * b + c
        val_with = a * (b + c)
        if val_no != val_with:
            problems.append({
                **_supp_base(f"ooo_parens_mul_{i}", 3, "boundary_test", []),
                "prompt": "Do these give the same result? Does the placement of parentheses matter?",
                "expression_a": f"{a} * {b} + {c}",
                "expression_b": f"{a} * ({b} + {c})",
                "expected_answer": False,
            })

    # --- PEMA connection: why only 2 operations ---
    problems.append({
        **_supp_base("ooo_pema_0", 3, "order_of_ops", ["Sub-Def"]),
        "prompt": (
            "PEMA says we only need two operations: addition and multiplication. "
            "Rewrite 8 - 3 using only addition:"
        ),
        "student_sees": "8 - 3",
        "expected_answer": "8 + (-3)",
    })
    problems.append({
        **_supp_base("ooo_pema_1", 3, "order_of_ops", ["Div-Def"]),
        "prompt": (
            "PEMA says we only need two operations: addition and multiplication. "
            "Rewrite 12 / 4 using only multiplication:"
        ),
        "student_sees": "12 / 4",
        "expected_answer": "12 * (1/4)",
    })
    problems.append({
        **_supp_base("ooo_pema_2", 3, "order_of_ops", ["Sub-Def", "Div-Def"]),
        "prompt": (
            "Using PEMA (only + and *), evaluate: 10 - 3 * 2. "
            "First rewrite subtraction as addition, then apply order of operations."
        ),
        "student_sees": "10 - 3 * 2",
        "expected_answer": 4,
    })

    return problems


def _render_property_chains(rng):
    """Multi-step transformations testing property discrimination.

    The key diagnostic: can the student tell associativity from commutativity
    when both appear in the same simplification chain?
    """
    problems = []

    # Pre-defined chains: each is (before, after, property_used)
    chains = [
        {
            "steps": [
                ("(a + b) + c", "a + (b + c)", "A-Assoc"),
                ("a + (b + c)", "a + (c + b)", "A-Comm"),
            ],
            "tier": 2,
            "context": "Simplifying a three-term sum",
        },
        {
            "steps": [
                ("(a + b) + c", "(b + a) + c", "A-Comm"),
                ("(b + a) + c", "b + (a + c)", "A-Assoc"),
            ],
            "tier": 2,
            "context": "Rearranging a three-term sum",
        },
        {
            "steps": [
                ("(ab)c", "a(bc)", "M-Assoc"),
                ("a(bc)", "a(cb)", "M-Comm"),
            ],
            "tier": 2,
            "context": "Regrouping a three-factor product",
        },
        {
            "steps": [
                ("(ab)c", "(ba)c", "M-Comm"),
                ("(ba)c", "b(ac)", "M-Assoc"),
            ],
            "tier": 2,
            "context": "Reordering a three-factor product",
        },
        {
            "steps": [
                ("a(b + c)", "ab + ac", "Dist-Right"),
                ("ab + ac", "ac + ab", "A-Comm"),
            ],
            "tier": 4,
            "context": "Distributing then rearranging",
        },
        {
            "steps": [
                ("ab + ac", "a(b + c)", "Factor"),
                ("a(b + c)", "(b + c)a", "M-Comm"),
            ],
            "tier": 4,
            "context": "Factoring then reordering",
        },
        {
            "steps": [
                ("a - b", "a + (-b)", "Sub-Def"),
                ("a + (-b)", "(-b) + a", "A-Comm"),
            ],
            "tier": 5,
            "context": "Rewriting subtraction, then using commutativity",
        },
        {
            "steps": [
                ("a / b", "a * (1/b)", "Div-Def"),
                ("a * (1/b)", "(1/b) * a", "M-Comm"),
            ],
            "tier": 5,
            "context": "Rewriting division, then using commutativity",
        },
    ]

    # For each chain, generate one identify_property problem per step
    all_skills = list(SKILL_DESCRIPTIONS.keys())

    for cidx, chain in enumerate(chains):
        chain_skills = [s["property"] for s in chain["steps"] if isinstance(s, dict)]
        if isinstance(chain["steps"][0], tuple):
            chain_skills = [s[2] for s in chain["steps"]]

        for sidx, step in enumerate(chain["steps"]):
            if isinstance(step, tuple):
                before, after, prop = step
            else:
                before, after, prop = step["before"], step["after"], step["property"]

            # Build the full chain display for context
            step_strs = []
            for si, s in enumerate(chain["steps"]):
                sb, sa = (s[0], s[1]) if isinstance(s, tuple) else (s["before"], s["after"])
                marker = " <--" if si == sidx else ""
                step_strs.append(f"Step {si + 1}: {sb} = {sa}{marker}")
            chain_display = "\n".join(step_strs)

            # Choices: include both properties from chain + distractors
            choices = list(set(chain_skills))
            for s in all_skills:
                if s not in choices:
                    choices.append(s)
                if len(choices) >= 6:
                    break
            rng.shuffle(choices)

            problems.append({
                **_supp_base(f"chain_{cidx}_s{sidx}", chain["tier"], "identify_property", chain_skills),
                "prompt": (
                    f"{chain['context']}:\n{chain_display}\n\n"
                    f"Which property was used in the highlighted step (Step {sidx + 1})?"
                ),
                "expression_before": before,
                "expression_after": after,
                "expected_answer": prop,
                "choices": choices,
            })

    return problems


def _render_custom_operations(rng):
    """AoPS-style: define a novel binary operation, test algebraic properties.

    This is DOK 3-4.  The student must apply the *definition* of commutativity
    or associativity to an unfamiliar operation — pure transfer.
    """
    problems = []

    # Each entry: (symbol, definition_text, formula(a,b), is_commutative, is_associative)
    ops = [
        ("star", "a star b = a + 2b",
         lambda a, b: a + 2 * b, False, False),
        ("diamond", "a diamond b = ab + a + b",
         lambda a, b: a * b + a + b, True, False),
        ("circle", "a circle b = a + b + 1",
         lambda a, b: a + b + 1, True, True),
        ("triangle", "a triangle b = a^2 + b^2",
         lambda a, b: a ** 2 + b ** 2, True, False),
        ("hash", "a hash b = 2a + b",
         lambda a, b: 2 * a + b, False, False),
        ("at", "a at b = (a + b) / 2",
         lambda a, b: (a + b) / 2, True, False),
    ]

    for idx, (sym, defn, fn, is_comm, is_assoc) in enumerate(ops):
        a, b = rng.sample([2, 3, 4, 5], 2)

        # --- Commutativity test ---
        ab = fn(a, b)
        ba = fn(b, a)
        problems.append({
            **_supp_base(f"custom_comm_{idx}", 6, "custom_operation", ["A-Comm", "M-Comm"]),
            "prompt": (
                f"Define: {defn}.\n"
                f"Is this operation commutative? "
                f"Compute {a} {sym} {b} and {b} {sym} {a} to check."
            ),
            "student_sees": f"{a} {sym} {b}  vs  {b} {sym} {a}",
            "expected_answer": str(is_comm).lower(),
            "custom_op_definition": defn,
            "custom_op_values": {
                f"{a} {sym} {b}": ab,
                f"{b} {sym} {a}": ba,
            },
        })

        # --- Associativity test ---
        c = rng.choice([2, 3, 4, 5])
        ab_c = fn(fn(a, b), c)
        a_bc = fn(a, fn(b, c))
        problems.append({
            **_supp_base(f"custom_assoc_{idx}", 7, "custom_operation", ["A-Assoc", "M-Assoc"]),
            "prompt": (
                f"Define: {defn}.\n"
                f"Is this operation associative? "
                f"Compute ({a} {sym} {b}) {sym} {c} and {a} {sym} ({b} {sym} {c}) to check."
            ),
            "student_sees": f"({a} {sym} {b}) {sym} {c}  vs  {a} {sym} ({b} {sym} {c})",
            "expected_answer": str(is_assoc).lower(),
            "custom_op_definition": defn,
            "custom_op_values": {
                f"({a} {sym} {b}) {sym} {c}": ab_c,
                f"{a} {sym} ({b} {sym} {c})": a_bc,
            },
        })

        # --- Evaluation-only (just compute) ---
        x, y = rng.sample([3, 4, 5, 6, 7], 2)
        problems.append({
            **_supp_base(f"custom_eval_{idx}", 5, "evaluate", []),
            "prompt": f"Define: {defn}.\nCompute {x} {sym} {y}.",
            "student_sees": f"{x} {sym} {y}",
            "expected_answer": fn(x, y),
        })

    return problems


def _render_competition_problems(rng):
    """Competition / AoPS-style problems requiring strategic property use.

    These go beyond strategic_compute — they require recognizing algebraic
    structure (difference of squares, telescoping, decomposition).
    """
    problems = []

    # --- Mental math decomposition (99 * 37 style) ---
    decomp = [
        ("99 * 37", 3663,
         "Rewrite as (100 - 1) * 37 = 3700 - 37 = 3663",
         ["Dist-Right", "Sub-Def"]),
        ("101 * 23", 2323,
         "Rewrite as (100 + 1) * 23 = 2300 + 23 = 2323",
         ["Dist-Right"]),
        ("98 * 15", 1470,
         "Rewrite as (100 - 2) * 15 = 1500 - 30 = 1470",
         ["Dist-Right", "Sub-Def"]),
        ("997 * 8", 7976,
         "Rewrite as (1000 - 3) * 8 = 8000 - 24 = 7976",
         ["Dist-Right", "Sub-Def"]),
        ("25 * 44", 1100,
         "Rewrite as 25 * 4 * 11 = 100 * 11 = 1100",
         ["M-Assoc", "Factor"]),
        ("48 * 25", 1200,
         "Rewrite as 12 * 4 * 25 = 12 * 100 = 1200",
         ["M-Assoc", "Factor"]),
    ]

    for idx, (expr, ans, strategy, skills) in enumerate(decomp):
        problems.append({
            **_supp_base(f"comp_decomp_{idx}", 6, "strategic_compute", skills),
            "prompt": (
                "Compute this WITHOUT a calculator. "
                "Look for a way to decompose or regroup."
            ),
            "student_sees": expr,
            "expected_answer": ans,
            "scaffolding": {"hint": "Can you rewrite one factor as a round number plus or minus something small?"},
        })

    # --- Alternating sums ---
    alt_sums = [
        ("1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10", -5,
         "Pair consecutive: (1-2)+(3-4)+...+(9-10) = 5 * (-1) = -5",
         ["A-Assoc", "Sub-Def"]),
        ("1 - 2 + 3 - 4 + ... + 99 - 100", -50,
         "50 pairs of -1",
         ["A-Assoc", "Sub-Def"]),
        ("2 - 4 + 6 - 8 + 10 - 12 + 14 - 16 + 18 - 20", -10,
         "Factor 2: 2(1-2+3-4+...) = 2*(-5) = -10",
         ["Factor", "A-Assoc", "Sub-Def"]),
    ]

    for idx, (expr, ans, strategy, skills) in enumerate(alt_sums):
        problems.append({
            **_supp_base(f"comp_altsum_{idx}", 6, "strategic_compute", skills),
            "prompt": (
                "Find the value of this alternating sum. "
                "Don't add one by one — find a pattern."
            ),
            "student_sees": expr,
            "expected_answer": ans,
            "scaffolding": {"hint": "Try pairing consecutive terms: (1-2), (3-4), ..."},
        })

    # --- Difference of squares ---
    dos = [
        ("If a + b = 10 and a - b = 4, find a^2 - b^2.",
         40, "(a+b)(a-b) = 10 * 4 = 40",
         ["Factor", "Dist-Right"]),
        ("If x + y = 7 and x - y = 3, find x^2 - y^2.",
         21, "(x+y)(x-y) = 7 * 3 = 21",
         ["Factor", "Dist-Right"]),
        ("Compute 51^2 - 49^2 without squaring.",
         200, "(51+49)(51-49) = 100 * 2 = 200",
         ["Factor", "Dist-Right"]),
        ("Compute 103^2 - 97^2 without squaring.",
         1200, "(103+97)(103-97) = 200 * 6 = 1200",
         ["Factor", "Dist-Right"]),
    ]

    for idx, (prompt_text, ans, strategy, skills) in enumerate(dos):
        problems.append({
            **_supp_base(f"comp_dos_{idx}", 7, "strategic_compute", skills),
            "prompt": prompt_text,
            "student_sees": prompt_text.split(",")[-1].strip().rstrip(".") if "," in prompt_text else prompt_text.split(".")[-2].strip() + ".",
            "expected_answer": ans,
            "scaffolding": {"hint": "Remember: a^2 - b^2 = (a+b)(a-b)."},
        })

    # --- Telescoping sums ---
    tele = [
        ("1/(1*2) + 1/(2*3) + 1/(3*4) + 1/(4*5)",
         "4/5",
         "Partial fractions: 1/(n(n+1)) = 1/n - 1/(n+1). Series telescopes to 1 - 1/5.",
         ["Sub-Def", "A-Assoc"]),
        ("1/(1*2) + 1/(2*3) + ... + 1/(9*10)",
         "9/10",
         "Telescopes to 1 - 1/10 = 9/10",
         ["Sub-Def", "A-Assoc"]),
    ]

    for idx, (expr, ans, strategy, skills) in enumerate(tele):
        problems.append({
            **_supp_base(f"comp_tele_{idx}", 7, "strategic_compute", skills),
            "prompt": (
                "Find the sum. Hint: try writing each fraction as a difference of two simpler fractions."
            ),
            "student_sees": expr,
            "expected_answer": ans,
            "scaffolding": {"hint": "1/(n(n+1)) = 1/n - 1/(n+1). What happens when you add them up?"},
        })

    return problems


def _render_fill_in_blank(rng):
    """Fill-in-the-blank problems testing property recognition.

    Medium difficulty (DOK 2): student must determine what value makes
    the equation true, requiring understanding of the underlying property.
    """
    problems = []
    pool = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]

    # --- Commutativity fill-in ---
    for i in range(6):
        a, b = rng.sample(pool, 2)

        # a + ___ = b + a  (answer: b)
        problems.append({
            **_supp_base(f"fib_comm_add_{i}", 2, "fill_in_blank", ["A-Comm"]),
            "prompt": "Fill in the blank to make this equation true:",
            "student_sees": f"{a} + ___ = {b} + {a}",
            "expected_answer": b,
        })

        # a * ___ = b * a  (answer: b)
        problems.append({
            **_supp_base(f"fib_comm_mul_{i}", 2, "fill_in_blank", ["M-Comm"]),
            "prompt": "Fill in the blank to make this equation true:",
            "student_sees": f"{a} * ___ = {b} * {a}",
            "expected_answer": b,
        })

    # --- Associativity fill-in ---
    for i in range(4):
        a, b, c = rng.sample(pool[:8], 3)

        # (a + b) + c = a + (___ + c)  (answer: b)
        problems.append({
            **_supp_base(f"fib_assoc_add_{i}", 3, "fill_in_blank", ["A-Assoc"]),
            "prompt": "Fill in the blank to make this equation true:",
            "student_sees": f"({a} + {b}) + {c} = {a} + (___ + {c})",
            "expected_answer": b,
        })

        # (a * b) * c = a * (___ * c)  (answer: b)
        problems.append({
            **_supp_base(f"fib_assoc_mul_{i}", 3, "fill_in_blank", ["M-Assoc"]),
            "prompt": "Fill in the blank to make this equation true:",
            "student_sees": f"({a} * {b}) * {c} = {a} * (___ * {c})",
            "expected_answer": b,
        })

    # --- Distribution fill-in ---
    for i in range(4):
        a, b, c = rng.sample(pool[:8], 3)

        # a(b + c) = a*b + a*___  (answer: c)
        problems.append({
            **_supp_base(f"fib_dist_{i}", 3, "fill_in_blank", ["Dist-Right"]),
            "prompt": "Fill in the blank to make this equation true:",
            "student_sees": f"{a}({b} + {c}) = {a}*{b} + {a}*___",
            "expected_answer": c,
        })

    # --- Identity fill-in ---
    for i in range(3):
        a = rng.choice(pool)
        # a + ___ = a  (answer: 0)
        problems.append({
            **_supp_base(f"fib_id_add_{i}", 1, "fill_in_blank", ["A-Ident"]),
            "prompt": "Fill in the blank to make this equation true:",
            "student_sees": f"{a} + ___ = {a}",
            "expected_answer": 0,
        })
        # a * ___ = a  (answer: 1)
        problems.append({
            **_supp_base(f"fib_id_mul_{i}", 1, "fill_in_blank", ["M-Ident"]),
            "prompt": "Fill in the blank to make this equation true:",
            "student_sees": f"{a} * ___ = {a}",
            "expected_answer": 1,
        })

    # --- Inverse fill-in ---
    for i in range(3):
        a = rng.choice(pool[:8])
        # a + ___ = 0  (answer: -a)
        problems.append({
            **_supp_base(f"fib_inv_add_{i}", 3, "fill_in_blank", ["Sub-Def"]),
            "prompt": "Fill in the blank to make this equation true:",
            "student_sees": f"{a} + ___ = 0",
            "expected_answer": -a,
        })
        # a * ___ = 1  (answer: 1/a)
        problems.append({
            **_supp_base(f"fib_inv_mul_{i}", 3, "fill_in_blank", ["Div-Def"]),
            "prompt": "Fill in the blank to make this equation true:",
            "student_sees": f"{a} * ___ = 1",
            "expected_answer": f"1/{a}",
        })

    return problems


def _render_parentheses_placement(rng):
    """Place parentheses to make an equation true.

    DOK 2-3: requires understanding how grouping changes evaluation order.
    """
    problems = []

    # Pre-designed: (expression_without_parens, target_value, correct_parenthesized, skills)
    placement = [
        ("2 + 3 * 4", 20, "(2 + 3) * 4", []),
        ("5 + 1 * 6", 36, "(5 + 1) * 6", []),
        ("8 - 2 * 3", 18, "(8 - 2) * 3", []),
        ("10 - 4 * 2", 12, "(10 - 4) * 2", []),
        ("2 * 3 + 4 * 5", 50, "2 * (3 + 4) * 5", ["M-Assoc"]),
        ("1 + 2 * 3 + 4", 21, "(1 + 2) * (3 + 4)", []),
        ("6 / 2 + 1", 2, "6 / (2 + 1)", []),
        ("12 / 2 + 2", 3, "12 / (2 + 2)", []),
        ("3 + 5 * 2 - 1", 15, "(3 + 5) * (2 - 1)", ["Sub-Def"]),
        ("4 * 2 + 3 * 5", 100, "4 * (2 + 3) * 5", ["M-Assoc"]),
    ]

    for idx, (expr, target, answer, skills) in enumerate(placement):
        problems.append({
            **_supp_base(f"parens_{idx}", 4, "parentheses_placement", skills),
            "prompt": f"Place parentheses in this expression to make it equal {target}:",
            "student_sees": expr,
            "expected_answer": answer,
        })

    # --- OoO error diagnosis ---
    ooo_errors = [
        ("8 / 2 * 4", 16, 1,
         "Student divided 8 by (2*4)=8, getting 1. Correct: (8/2)*4 = 4*4 = 16. Multiplication and division go left to right.",
         []),
        ("6 - 3 + 2", 5, 1,
         "Student computed 6 - (3+2) = 1. Correct: (6-3)+2 = 3+2 = 5. Addition and subtraction go left to right.",
         ["Sub-Def"]),
        ("2 + 3^2", 11, 25,
         "Student computed (2+3)^2 = 25. Correct: 2 + 9 = 11. Exponents before addition.",
         []),
        ("4 * 2^3", 32, 512,
         "Student computed (4*2)^3 = 512. Correct: 4 * 8 = 32. Exponents before multiplication.",
         []),
    ]

    for idx, (expr, correct, wrong, explanation, skills) in enumerate(ooo_errors):
        problems.append({
            **_supp_base(f"ooo_diag_{idx}", 4, "find_error", skills),
            "prompt": f"A student evaluated {expr} and got {wrong}. What is the correct answer, and what was the error?",
            "student_sees": f"{expr} = {wrong}  (student's answer)",
            "original_expression": expr,
            "claimed_answer": str(wrong),
            "expected_answer": correct,
        })

    return problems


def _render_proof_disproof(rng):
    """Conceptual prompts about WHY properties hold or fail.

    DOK 3-4: requires constructing arguments or counterexamples.
    These use open-ended text answers graded by the explanation analysis.
    """
    problems = []

    # --- Disproof by counterexample ---
    disproofs = [
        {
            "prompt": (
                "Prove that subtraction is NOT commutative. "
                "Give a specific counterexample showing a - b != b - a."
            ),
            "student_sees": "Is a - b = b - a always true?",
            "expected_answer": "false",
            "skills": ["Sub-Def", "A-Comm"],
            "tier": 5,
        },
        {
            "prompt": (
                "Prove that subtraction is NOT associative. "
                "Give a specific counterexample showing (a - b) - c != a - (b - c)."
            ),
            "student_sees": "Is (a - b) - c = a - (b - c) always true?",
            "expected_answer": "false",
            "skills": ["Sub-Def", "A-Assoc"],
            "tier": 5,
        },
        {
            "prompt": (
                "Prove that division is NOT commutative. "
                "Give a specific counterexample showing a / b != b / a."
            ),
            "student_sees": "Is a / b = b / a always true?",
            "expected_answer": "false",
            "skills": ["Div-Def", "M-Comm"],
            "tier": 5,
        },
        {
            "prompt": (
                "Prove that division is NOT associative. "
                "Give a specific counterexample showing (a / b) / c != a / (b / c)."
            ),
            "student_sees": "Is (a / b) / c = a / (b / c) always true?",
            "expected_answer": "false",
            "skills": ["Div-Def", "M-Assoc"],
            "tier": 5,
        },
    ]

    for idx, d in enumerate(disproofs):
        problems.append({
            **_supp_base(f"proof_disproof_{idx}", d["tier"], "proof_disproof", d["skills"]),
            "prompt": d["prompt"],
            "student_sees": d["student_sees"],
            "expected_answer": d["expected_answer"],
        })

    # --- PEMA insight proofs ---
    pema_proofs = [
        {
            "prompt": (
                "Subtraction is NOT commutative: a - b != b - a. "
                "But if we rewrite a - b as a + (-b), then addition IS commutative: "
                "a + (-b) = (-b) + a. Explain why this works and what it means."
            ),
            "student_sees": "a - b  vs  a + (-b)",
            "expected_answer": "true",
            "skills": ["Sub-Def", "A-Comm"],
            "tier": 6,
        },
        {
            "prompt": (
                "Division is NOT commutative: a / b != b / a. "
                "But if we rewrite a / b as a * (1/b), then multiplication IS commutative: "
                "a * (1/b) = (1/b) * a. Explain why this works and what it means."
            ),
            "student_sees": "a / b  vs  a * (1/b)",
            "expected_answer": "true",
            "skills": ["Div-Def", "M-Comm"],
            "tier": 6,
        },
        {
            "prompt": (
                "Why does PEMDAS simplify to PEMA? Explain why subtraction and division "
                "don't need their own 'level' in the order of operations."
            ),
            "student_sees": "PEMDAS -> PEMA",
            "expected_answer": "true",
            "skills": ["Sub-Def", "Div-Def"],
            "tier": 6,
        },
    ]

    for idx, p in enumerate(pema_proofs):
        problems.append({
            **_supp_base(f"proof_pema_{idx}", p["tier"], "proof_disproof", p["skills"]),
            "prompt": p["prompt"],
            "student_sees": p["student_sees"],
            "expected_answer": p["expected_answer"],
        })

    return problems
