from collections import Counter
from domain.ast_nodes import Leaf
from domain.axioms import AXIOM_COGNITIVE_CLASS, AXIOM_MISCONCEPTION_RISK, get_all_valid_moves


# ==========================================
# 5. AST ANALYSIS FUNCTIONS
# ==========================================
def count_leaves(node):
    if isinstance(node, Leaf):
        return 1
    count = count_leaves(node.left)
    if node.right:
        count += count_leaves(node.right)
    return count


def ast_depth(node):
    if isinstance(node, Leaf):
        return 0
    left_d = ast_depth(node.left)
    right_d = ast_depth(node.right) if node.right else 0
    return 1 + max(left_d, right_d)


def count_operations(node):
    if isinstance(node, Leaf):
        return 0
    count = 1
    count += count_operations(node.left)
    if node.right:
        count += count_operations(node.right)
    return count


def get_operation_types(node):
    ops = set()
    if isinstance(node, Leaf):
        return ops
    ops.add(node.op)
    ops |= get_operation_types(node.left)
    if node.right:
        ops |= get_operation_types(node.right)
    return ops


def has_operation_mixing(node):
    ops = get_operation_types(node)
    additive = ops & {"+", "-", "NEG"}
    multiplicative = ops & {"*", "/", "INV"}
    return bool(additive) and bool(multiplicative)


def count_branching_moves(node, axioms):
    total_moves = 0
    for ax_func in axioms.values():
        total_moves += len(get_all_valid_moves(node, ax_func))
    return total_moves


def ast_edit_distance(node_a, node_b):
    if isinstance(node_a, Leaf) and isinstance(node_b, Leaf):
        return 0 if node_a.val == node_b.val else 0.5

    if isinstance(node_a, Leaf) or isinstance(node_b, Leaf):
        return 1 + (count_operations(node_a) if isinstance(node_b, Leaf) else count_operations(node_b))

    cost = 0
    if node_a.op != node_b.op:
        cost += 1

    cost += ast_edit_distance(node_a.left, node_b.left)

    if node_a.right and node_b.right:
        cost += ast_edit_distance(node_a.right, node_b.right)
    elif node_a.right or node_b.right:
        extra = node_a.right if node_a.right else node_b.right
        cost += 1 + count_operations(extra)

    return cost


def surface_dissimilarity(pretty_a, pretty_b):
    a = pretty_a.replace(" ", "")
    b = pretty_b.replace(" ", "")

    len_a = len(a)
    len_b = len(b)

    freq_a = Counter(a)
    freq_b = Counter(b)
    all_chars = set(freq_a.keys()) | set(freq_b.keys())
    overlap = sum(min(freq_a.get(c, 0), freq_b.get(c, 0)) for c in all_chars)
    total = max(len_a, len_b)

    if total == 0:
        return 0.0
    return 1.0 - (overlap / total)


# ==========================================
# 5b. CLUSTER CLASSIFICATION
# ==========================================
SKILL_TO_CLUSTER = {
    "A-Comm": "additive_structure",
    "A-Assoc": "additive_structure",
    "A-Assoc-Rev": "additive_structure",
    "M-Comm": "multiplicative_structure",
    "M-Assoc": "multiplicative_structure",
    "M-Assoc-Rev": "multiplicative_structure",
    "Dist-Right": "distribution",
    "Dist-Left": "distribution",
    "Factor": "distribution",
    "Sub-Def": "signed_arithmetic",
    "Div-Def": "signed_arithmetic",
}

SEED_CLUSTER = {
    "0a_add_two": "additive_structure",
    "0b_add_three": "additive_structure",
    "0c_add_four": "additive_structure",
    "0d_mult_two": "multiplicative_structure",
    "0e_mult_three": "multiplicative_structure",
    "1a_add_two": "additive_structure",
    "1b_mult_two": "multiplicative_structure",
    "2a_add_three": "additive_structure",
    "2b_mult_three": "multiplicative_structure",
    "2c_add_four": "additive_structure",
    "2d_mult_four": "multiplicative_structure",
    "3a_ma_right": "order_of_operations",
    "3b_ma_left": "order_of_operations",
    "3c_ma_double": "order_of_operations",
    "3d_ma_triple": "order_of_operations",
    "3e_parens": "order_of_operations",
    "4a_dist": "distribution",
    "4b_dist_left": "distribution",
    "4c_factor_ab_ac": "distribution",
    "4d_factor_ba_bc": "distribution",
    "4e_factor_three": "distribution",
    "5a_sub": "signed_arithmetic",
    "5b_div": "signed_arithmetic",
    "5c_sub_after_mult": "signed_arithmetic",
    "5d_sub_before_mult": "signed_arithmetic",
    "5e_div_then_add": "signed_arithmetic",
    "6a_dist_sub": "signed_distribution",
    "6b_dist_neg": "signed_distribution",
    "6c_div_add": "signed_distribution",
    "6d_div_product": "signed_distribution",
    "6e_div_sub": "signed_distribution",
    "7a_foil": "complex_composition",
    "7b_signed_foil": "complex_composition",
    "7c_double_signed": "complex_composition",
    "7d_sub_products": "complex_composition",
    "7e_sum_products": "complex_composition",
    "7f_dist_three": "complex_composition",
}


def classify_cluster(skill_set, seed_name):
    if not skill_set:
        return SEED_CLUSTER.get(seed_name, "unknown"), set(), False

    clusters_touched = set()
    for skill in skill_set:
        cluster = SKILL_TO_CLUSTER.get(skill)
        if cluster:
            clusters_touched.add(cluster)

    if not clusters_touched:
        primary = SEED_CLUSTER.get(seed_name, "unknown")
        return primary, clusters_touched, False

    cluster_counts = Counter(SKILL_TO_CLUSTER.get(s) for s in skill_set if SKILL_TO_CLUSTER.get(s))
    primary = cluster_counts.most_common(1)[0][0]
    is_bridge = len(clusters_touched) > 1
    return primary, clusters_touched, is_bridge


# ==========================================
# 6. COGNITIVE FEATURE COMPUTATION
# ==========================================
def compute_cognitive_features(node, entry, axioms, seed_node=None):
    n_leaves = count_leaves(node)
    depth = ast_depth(node)
    n_ops = count_operations(node)
    n_vars = len(node.get_variables())
    ops_present = get_operation_types(node)
    mixes_ops = has_operation_mixing(node)
    n_branches = count_branching_moves(node, axioms)

    derivation = entry.get("derivation_path", [])
    skill_set = entry.get("skill_set", [])
    tier = entry.get("tier", 1)
    layer = entry.get("layer", 0)
    n_steps = len(derivation)
    n_skills = len(skill_set)

    insight_steps = [s for s in derivation if AXIOM_COGNITIVE_CLASS.get(s) == "insight"]
    rewrite_steps = [s for s in derivation if AXIOM_COGNITIVE_CLASS.get(s) == "rewrite"]

    additive_skills = {"A-Comm", "A-Assoc", "A-Assoc-Rev", "Sub-Def"}
    multiplicative_skills = {"M-Comm", "M-Assoc", "M-Assoc-Rev", "Div-Def"}
    bridging_skills = {"Dist-Right", "Dist-Left", "Factor"}

    skills_as_set = set(skill_set)
    has_additive = bool(skills_as_set & additive_skills)
    has_multiplicative = bool(skills_as_set & multiplicative_skills)
    has_bridging = bool(skills_as_set & bridging_skills)
    skill_family_mixing = has_additive + has_multiplicative + has_bridging

    misconception_risk = max((AXIOM_MISCONCEPTION_RISK.get(s, 0) for s in derivation), default=0.0)

    sign_ops = ops_present & {"-", "NEG"}
    has_sign_complexity = len(sign_ops) > 0

    depth_penalty = max(0, depth - 1)
    mixing_penalty = 1 if mixes_ops else 0
    density_load = min(9, n_leaves + depth_penalty + mixing_penalty)

    step_component = min(1.0, n_steps / 5.0)
    insight_component = 0.3 * len(insight_steps)
    rewrite_component = 0.15 * len(rewrite_steps)
    branch_component = min(0.3, n_branches / 30.0)

    scaffolding_demand = min(
        1.0,
        0.3 * step_component + 0.3 * insight_component + 0.15 * rewrite_component + 0.25 * branch_component,
    )

    pretty_derived = entry.get("pretty", "")
    pretty_seed = entry.get("seed_pretty", "")

    if seed_node is not None and layer > 0:
        raw_structural = ast_edit_distance(seed_node, node)
        max_possible = count_operations(seed_node) + count_operations(node) + 2
        transfer_structural = min(1.0, raw_structural / max(1, max_possible))
        transfer_surface = surface_dissimilarity(pretty_seed, pretty_derived)
    else:
        transfer_structural = 0.0
        transfer_surface = 0.0

    transfer_distance = round(0.5 * transfer_structural + 0.5 * transfer_surface, 3)

    primary_cluster, clusters_touched, is_bridge = classify_cluster(skill_set, entry.get("seed_name", ""))

    n_clusters = len(clusters_touched)
    if n_clusters <= 1:
        cross_cluster_demand = 0.0
    elif n_clusters == 2:
        cross_cluster_demand = 0.5
    else:
        cross_cluster_demand = min(1.0, 0.5 + 0.2 * (n_clusters - 2))

    r_base = min(
        1.0,
        0.15 + 0.25 * min(1.0, n_steps / 4.0) + 0.30 * (len(insight_steps) / max(1, n_steps)) + 0.30 * transfer_distance,
    )

    question_r_multiplier = {
        "identify_property": 0.3,
        "equivalent": 0.4,
        "evaluate": 0.6,
        "expand": 0.8,
        "simplify": 1.0,
    }
    retrieval_by_qtype = {qtype: round(min(1.0, r_base * mult), 3) for qtype, mult in question_r_multiplier.items()}

    tier_difficulty = tier / 7.0
    tier_max = {0: 5, 1: 5, 2: 4, 3: 4, 4: 3, 5: 3, 6: 2, 7: 2}
    effective_max = tier_max.get(tier, 3)
    layer_difficulty = min(1.0, layer / max(1, effective_max))

    interaction_bonus = 0.0
    if has_bridging and (has_additive or has_multiplicative):
        interaction_bonus = 0.15
    if has_additive and has_multiplicative and has_bridging:
        interaction_bonus = 0.25

    insight_penalty = 0.1 * len(insight_steps)

    challenge_level = min(
        1.0,
        0.35 * tier_difficulty
        + 0.25 * layer_difficulty
        + 0.15 * interaction_bonus / 0.25
        + 0.15 * insight_penalty
        + 0.10 * misconception_risk,
    )

    if n_steps <= 1 and tier <= 3:
        feedback_ceiling = "Low"
        feedback_ceiling_score = 0.2
    elif n_steps <= 2 or (tier <= 4 and not insight_steps):
        feedback_ceiling = "Med"
        feedback_ceiling_score = 0.5
    else:
        feedback_ceiling = "High"
        feedback_ceiling_score = 0.8

    complexity_factor = min(1.0, (n_ops + n_leaves) / 10.0)
    sign_interference = 0.2 if has_sign_complexity else 0.0
    procedural_weight = 0.3 if tier <= 3 else 0.5

    spacing_sensitivity = min(
        1.0,
        0.35 * complexity_factor + 0.25 * sign_interference + 0.25 * procedural_weight + 0.15 * misconception_risk,
    )

    challenge_informativeness = 4.0 * challenge_level * (1.0 - challenge_level)
    skill_specificity = 1.0 / max(1, n_skills)
    branch_informativeness = min(1.0, n_branches / 15.0) * (1.0 - min(1.0, n_branches / 30.0))
    diagnostic_value = 0.40 * challenge_informativeness + 0.35 * skill_specificity + 0.25 * branch_informativeness

    return {
        "density_load": density_load,
        "scaffolding_demand": round(scaffolding_demand, 3),
        "transfer_distance": transfer_distance,
        "transfer_structural": round(transfer_structural, 3),
        "transfer_surface": round(transfer_surface, 3),
        "primary_cluster": primary_cluster,
        "clusters_touched": sorted(list(clusters_touched)),
        "is_bridge": is_bridge,
        "cross_cluster_demand": round(cross_cluster_demand, 3),
        "retrieval_base": round(r_base, 3),
        "retrieval_by_qtype": retrieval_by_qtype,
        "challenge_level": round(challenge_level, 3),
        "feedback_ceiling": feedback_ceiling,
        "feedback_ceiling_score": round(feedback_ceiling_score, 3),
        "spacing_sensitivity": round(spacing_sensitivity, 3),
        "diagnostic_value": round(diagnostic_value, 3),
        "n_leaves": n_leaves,
        "ast_depth": depth,
        "n_operations": n_ops,
        "n_variables": n_vars,
        "n_branching_moves": n_branches,
        "mixes_operations": mixes_ops,
        "has_sign_complexity": has_sign_complexity,
        "operation_types": sorted(list(ops_present)),
        "n_derivation_steps": n_steps,
        "n_distinct_skills": n_skills,
        "skill_families_involved": skill_family_mixing,
        "has_insight_steps": len(insight_steps) > 0,
        "insight_steps": insight_steps,
        "misconception_risk": round(misconception_risk, 3),
        "step_types": {
            "structural": len([s for s in derivation if AXIOM_COGNITIVE_CLASS.get(s) == "structural"]),
            "rewrite": len(rewrite_steps),
            "insight": len(insight_steps),
        },
    }
