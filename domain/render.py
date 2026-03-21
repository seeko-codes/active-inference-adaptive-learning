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
                errors.append({
                    **base,
                    "question_type": "find_error",
                    "retrieval_demand": r_demand,
                    "prompt": QUESTION_TEMPLATES["find_error"],
                    "original_expression": pretty_simple,
                    "student_sees": f"{pretty_simple} = {pretty_print(wrong)}",
                    "claimed_answer": pretty_print(wrong),
                    "correct_answer": pretty_complex if derivation and "Dist" in derivation[0] else pretty_simple,
                    "error_type": "partial_distribution",
                    "error_description": "Only distributed to the first term, not both",
                })

    if "Sub-Def" in derivation:
        errors.append({
            **base,
            "question_type": "find_error",
            "retrieval_demand": r_demand,
            "prompt": QUESTION_TEMPLATES["find_error"],
            "original_expression": pretty_complex,
            "student_sees": f"Simplification attempt of {pretty_complex}",
            "claimed_answer": "(sign error - negation not fully distributed)",
            "correct_answer": pretty_simple,
            "error_type": "sign_error",
            "error_description": "Failed to apply negation to all terms",
        })

    if "Sub-Def" in skill_set and "A-Comm" in skill_set:
        errors.append({
            **base,
            "question_type": "find_error",
            "retrieval_demand": r_demand,
            "prompt": QUESTION_TEMPLATES["find_error"],
            "original_expression": pretty_complex,
            "student_sees": "Claim: subtraction is commutative",
            "claimed_answer": "a - b = b - a",
            "correct_answer": "Subtraction is NOT commutative. a - b != b - a (unless a = b)",
            "error_type": "commutativity_overgeneralization",
            "error_description": "Applied commutativity to subtraction",
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
