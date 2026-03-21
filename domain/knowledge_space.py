import uuid
from domain.ast_nodes import Leaf, Op, pretty_print
from domain.axioms import AXIOMS, get_all_valid_moves
from domain.cognitive_features import compute_cognitive_features

# ==========================================
# 7. TIER SYSTEM
# ==========================================
TIERS = [
    {
        "tier": 0, "name": "Raw Arithmetic",
        "description": "Pure computation with addition and multiplication. No symbolic manipulation.",
        "prerequisites": [],
        "unlocks": [],
        "seeds": {
            "0a_add_two": Op("+", Leaf("a"), Leaf("b")),
            "0b_add_three": Op("+", Op("+", Leaf("a"), Leaf("b")), Leaf("c")),
            "0c_add_four": Op("+", Op("+", Op("+", Leaf("a"), Leaf("b")), Leaf("c")), Leaf("d")),
            "0d_mult_two": Op("*", Leaf("a"), Leaf("b")),
            "0e_mult_three": Op("*", Op("*", Leaf("a"), Leaf("b")), Leaf("c")),
        },
    },
    {
        "tier": 1, "name": "Commutativity",
        "description": "Order doesn't matter for addition and multiplication.",
        "prerequisites": [0],
        "unlocks": ["A-Comm", "M-Comm"],
        "seeds": {
            "1a_add_two": Op("+", Leaf("a"), Leaf("b")),
            "1b_mult_two": Op("*", Leaf("a"), Leaf("b")),
        },
    },
    {
        "tier": 2, "name": "Associativity",
        "description": "Grouping doesn't matter for same operation.",
        "prerequisites": [1],
        "unlocks": ["A-Assoc", "A-Assoc-Rev", "M-Assoc", "M-Assoc-Rev"],
        "seeds": {
            "2a_add_three": Op("+", Op("+", Leaf("a"), Leaf("b")), Leaf("c")),
            "2b_mult_three": Op("*", Op("*", Leaf("a"), Leaf("b")), Leaf("c")),
            "2c_add_four": Op("+", Op("+", Op("+", Leaf("a"), Leaf("b")), Leaf("c")), Leaf("d")),
            "2d_mult_four": Op("*", Op("*", Op("*", Leaf("a"), Leaf("b")), Leaf("c")), Leaf("d")),
        },
    },
    {
        "tier": 3, "name": "Order of Operations (PEMA)",
        "description": "Multiplication before addition. No distribution yet.",
        "prerequisites": [1, 2],
        "unlocks": [],
        "seeds": {
            "3a_ma_right": Op("+", Op("*", Leaf("a"), Leaf("b")), Leaf("c")),
            "3b_ma_left": Op("+", Leaf("a"), Op("*", Leaf("b"), Leaf("c"))),
            "3c_ma_double": Op("+", Op("*", Leaf("a"), Leaf("b")), Op("*", Leaf("c"), Leaf("d"))),
            "3d_ma_triple": Op("+", Op("+", Op("*", Leaf("a"), Leaf("b")), Op("*", Leaf("c"), Leaf("d"))), Leaf("e")),
            "3e_parens": Op("*", Leaf("a"), Op("+", Leaf("b"), Leaf("c"))),
        },
    },
    {
        "tier": 4, "name": "Distribution",
        "description": "Bridge between multiplication and addition.",
        "prerequisites": [1, 2, 3],
        "unlocks": ["Dist-Right", "Dist-Left", "Factor"],
        "seeds": {
            "4a_dist": Op("*", Leaf("a"), Op("+", Leaf("b"), Leaf("c"))),
            "4b_dist_left": Op("*", Op("+", Leaf("a"), Leaf("b")), Leaf("c")),
            "4c_factor_ab_ac": Op("+", Op("*", Leaf("a"), Leaf("b")), Op("*", Leaf("a"), Leaf("c"))),
            "4d_factor_ba_bc": Op("+", Op("*", Leaf("b"), Leaf("a")), Op("*", Leaf("b"), Leaf("c"))),
            "4e_factor_three": Op("+", Op("+", Op("*", Leaf("a"), Leaf("b")), Op("*", Leaf("a"), Leaf("c"))), Op("*", Leaf("a"), Leaf("d"))),
        },
    },
    {
        "tier": 5, "name": "Signed Operations",
        "description": "Subtraction as +NEG and division as *INV.",
        "prerequisites": [1],
        "unlocks": ["Sub-Def", "Div-Def"],
        "seeds": {
            "5a_sub": Op("-", Leaf("a"), Leaf("b")),
            "5b_div": Op("/", Leaf("a"), Leaf("b")),
            "5c_sub_after_mult": Op("-", Op("*", Leaf("a"), Leaf("b")), Leaf("c")),
            "5d_sub_before_mult": Op("+", Leaf("a"), Op("*", Op("-", Leaf("0"), Leaf("b")), Leaf("c"))),
            "5e_div_then_add": Op("+", Op("/", Leaf("a"), Leaf("b")), Leaf("c")),
        },
    },
    {
        "tier": 6, "name": "Signed Distribution",
        "description": "Distribution with subtraction/division context.",
        "prerequisites": [4, 5],
        "unlocks": [],
        "seeds": {
            "6a_dist_sub": Op("*", Leaf("a"), Op("-", Leaf("b"), Leaf("c"))),
            "6b_dist_neg": Op("*", Op("-", Leaf("0"), Leaf("a")), Op("+", Leaf("b"), Leaf("c"))),
            "6c_div_add": Op("/", Op("+", Leaf("a"), Leaf("b")), Leaf("c")),
            "6d_div_product": Op("/", Op("*", Leaf("a"), Leaf("b")), Leaf("c")),
            "6e_div_sub": Op("/", Op("-", Leaf("a"), Leaf("b")), Leaf("c")),
        },
    },
    {
        "tier": 7, "name": "Complex Combinations",
        "description": "FOIL, signed FOIL, and multi-step composition.",
        "prerequisites": [4, 5, 6],
        "unlocks": [],
        "seeds": {
            "7a_foil": Op("*", Op("+", Leaf("a"), Leaf("b")), Op("+", Leaf("c"), Leaf("d"))),
            "7b_signed_foil": Op("*", Op("-", Leaf("a"), Leaf("b")), Op("+", Leaf("c"), Leaf("d"))),
            "7c_double_signed": Op("*", Op("-", Leaf("a"), Leaf("b")), Op("-", Leaf("c"), Leaf("d"))),
            "7d_sub_products": Op("-", Op("*", Leaf("a"), Leaf("b")), Op("*", Leaf("a"), Leaf("c"))),
            "7e_sum_products": Op("+", Op("*", Leaf("a"), Leaf("b")), Op("*", Leaf("a"), Leaf("c"))),
            "7f_dist_three": Op("*", Leaf("a"), Op("+", Op("+", Leaf("b"), Leaf("c")), Leaf("d"))),
        },
    },
]


# ==========================================
# 8. GENERATION ENGINE
# ==========================================
def generate_unified_hierarchy(tiers, max_layer=3):
    tier_max_layer = {
        0: max_layer, 1: max_layer, 2: max_layer, 3: max_layer,
        4: max_layer, 5: max_layer, 6: max(1, max_layer - 1), 7: max(1, max_layer - 1),
    }

    tier_max_forms_per_seed = {
        0: 50, 1: 20, 2: 40, 3: 40, 4: 40, 5: 30, 6: 30, 7: 25,
    }

    all_entries = []
    tier_summaries = []
    ast_registry = {}

    for tier_def in tiers:
        tier_num = tier_def["tier"]
        tier_name = tier_def["name"]
        prereqs = tier_def["prerequisites"]

        print(f"\n{'=' * 60}")
        print(f"TIER {tier_num}: {tier_name}")
        print(f"  Prerequisites: {prereqs if prereqs else 'None'}")
        print(f"  Unlocks: {tier_def['unlocks'] if tier_def['unlocks'] else 'Combinations of prior skills'}")
        print(f"{'=' * 60}")

        tier_entries = []

        for seed_name, seed_node in tier_def["seeds"].items():
            pretty = pretty_print(seed_node)
            print(f"\n  Seed: {seed_name} = {pretty}")

            queue = [(seed_node, [])]
            visited_hashes = {seed_node.to_string()}
            seed_form_count = 0
            max_forms = tier_max_forms_per_seed.get(tier_num, 30)

            seed_entry = {
                "problem_id": f"T{tier_num}_L0_{seed_name}",
                "tier": tier_num,
                "tier_name": tier_name,
                "layer": 0,
                "prerequisite_tiers": prereqs,
                "seed_name": seed_name,
                "ast_string": seed_node.to_string(),
                "pretty": pretty,
                "canonical_id": pretty.replace(" ", ""),
                "derivation_path": [],
                "skill_set": [],
                "seed_pretty": pretty,
            }

            seed_features = compute_cognitive_features(seed_node, seed_entry, AXIOMS, seed_node=seed_node)
            seed_entry["cognitive_features"] = seed_features
            seed_entry["global_difficulty"] = seed_features["challenge_level"]

            tier_entries.append(seed_entry)
            ast_registry[seed_entry["problem_id"]] = seed_node

            tier_depth = tier_max_layer.get(tier_num, max_layer)
            for current_layer in range(1, tier_depth + 1):
                if seed_form_count >= max_forms:
                    break

                next_queue = []
                layer_count = 0

                for current_node, history in queue:
                    if seed_form_count >= max_forms:
                        break

                    for ax_name, ax_func in AXIOMS.items():
                        if seed_form_count >= max_forms:
                            break

                        valid_results = get_all_valid_moves(current_node, ax_func)

                        for new_node in valid_results:
                            if seed_form_count >= max_forms:
                                break

                            node_hash = new_node.to_string()
                            if node_hash in visited_hashes:
                                continue

                            visited_hashes.add(node_hash)
                            new_history = history + [ax_name]
                            next_queue.append((new_node, new_history))
                            skill_set = sorted(set(new_history))

                            pid = f"T{tier_num}_L{current_layer}_{str(uuid.uuid4())[:8]}"
                            pretty_derived = pretty_print(new_node)

                            entry = {
                                "problem_id": pid,
                                "tier": tier_num,
                                "tier_name": tier_name,
                                "layer": current_layer,
                                "prerequisite_tiers": prereqs,
                                "seed_name": seed_name,
                                "ast_string": node_hash,
                                "pretty": pretty_derived,
                                "canonical_id": pretty_derived.replace(" ", ""),
                                "derivation_path": new_history,
                                "skill_set": skill_set,
                                "seed_pretty": pretty,
                            }

                            features = compute_cognitive_features(new_node, entry, AXIOMS, seed_node=seed_node)
                            entry["cognitive_features"] = features
                            entry["global_difficulty"] = features["challenge_level"]

                            tier_entries.append(entry)
                            ast_registry[pid] = new_node
                            layer_count += 1
                            seed_form_count += 1

                queue = next_queue
                print(f"    Layer {current_layer}: {layer_count} new forms (total from seed: {seed_form_count})")

        all_entries.extend(tier_entries)

        tier_skills = set()
        for e in tier_entries:
            for s in e["skill_set"]:
                tier_skills.add(s)

        tier_features = [e["cognitive_features"] for e in tier_entries]
        tier_summaries.append({
            "tier": tier_num,
            "name": tier_name,
            "description": tier_def["description"],
            "prerequisites": prereqs,
            "unlocks": tier_def["unlocks"],
            "total_forms": len(tier_entries),
            "skills_exercised": sorted(tier_skills),
            "seeds": {k: pretty_print(v) for k, v in tier_def["seeds"].items()},
            "feature_ranges": {
                "density_load": {
                    "min": min(f["density_load"] for f in tier_features),
                    "max": max(f["density_load"] for f in tier_features),
                    "mean": round(sum(f["density_load"] for f in tier_features) / len(tier_features), 2),
                },
                "challenge_level": {
                    "min": min(f["challenge_level"] for f in tier_features),
                    "max": max(f["challenge_level"] for f in tier_features),
                    "mean": round(sum(f["challenge_level"] for f in tier_features) / len(tier_features), 3),
                },
                "scaffolding_demand": {
                    "min": min(f["scaffolding_demand"] for f in tier_features),
                    "max": max(f["scaffolding_demand"] for f in tier_features),
                    "mean": round(sum(f["scaffolding_demand"] for f in tier_features) / len(tier_features), 3),
                },
                "transfer_distance": {
                    "min": min(f["transfer_distance"] for f in tier_features),
                    "max": max(f["transfer_distance"] for f in tier_features),
                    "mean": round(sum(f["transfer_distance"] for f in tier_features) / len(tier_features), 3),
                },
                "diagnostic_value": {
                    "min": min(f["diagnostic_value"] for f in tier_features),
                    "max": max(f["diagnostic_value"] for f in tier_features),
                    "mean": round(sum(f["diagnostic_value"] for f in tier_features) / len(tier_features), 3),
                },
            },
            "cluster_distribution": dict(sorted(
                ((pc, sum(1 for g in tier_features if g["primary_cluster"] == pc))
                 for pc in set(f["primary_cluster"] for f in tier_features)),
                key=lambda x: -x[1],
            )),
            "bridge_count": sum(1 for f in tier_features if f["is_bridge"]),
        })

    return all_entries, tier_summaries, ast_registry
