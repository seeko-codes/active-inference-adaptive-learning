from domain.ast_nodes import Node, Leaf, Op

# ==========================================
# 3. Axioms
# ==========================================
def a_comm(n):
    if isinstance(n, Op) and n.op == "+":
        return Op("+", n.right.clone(), n.left.clone())
    return None


def a_assoc(n):
    if isinstance(n, Op) and n.op == "+" and isinstance(n.left, Op) and n.left.op == "+":
        return Op("+", n.left.left.clone(), Op("+", n.left.right.clone(), n.right.clone()))
    return None


def a_assoc_rev(n):
    if isinstance(n, Op) and n.op == "+" and isinstance(n.right, Op) and n.right.op == "+":
        return Op("+", Op("+", n.left.clone(), n.right.left.clone()), n.right.right.clone())
    return None


def m_comm(n):
    if isinstance(n, Op) and n.op == "*":
        return Op("*", n.right.clone(), n.left.clone())
    return None


def m_assoc(n):
    if isinstance(n, Op) and n.op == "*" and isinstance(n.left, Op) and n.left.op == "*":
        return Op("*", n.left.left.clone(), Op("*", n.left.right.clone(), n.right.clone()))
    return None


def m_assoc_rev(n):
    if isinstance(n, Op) and n.op == "*" and isinstance(n.right, Op) and n.right.op == "*":
        return Op("*", Op("*", n.left.clone(), n.right.left.clone()), n.right.right.clone())
    return None


def dist(n):
    if isinstance(n, Op) and n.op == "*" and isinstance(n.right, Op) and n.right.op == "+":
        return Op("+", Op("*", n.left.clone(), n.right.left.clone()),
                  Op("*", n.left.clone(), n.right.right.clone()))
    return None


def dist_left(n):
    if isinstance(n, Op) and n.op == "*" and isinstance(n.left, Op) and n.left.op == "+":
        return Op("+", Op("*", n.left.left.clone(), n.right.clone()),
                  Op("*", n.left.right.clone(), n.right.clone()))
    return None


def factor(n):
    if isinstance(n, Op) and n.op == "+":
        if isinstance(n.left, Op) and n.left.op == "*" and isinstance(n.right, Op) and n.right.op == "*":
            if n.left.left.to_string() == n.right.left.to_string():
                return Op("*", n.left.left.clone(), Op("+", n.left.right.clone(), n.right.right.clone()))
            if n.left.right.to_string() == n.right.right.to_string():
                return Op("*", Op("+", n.left.left.clone(), n.right.left.clone()), n.left.right.clone())
    return None


def sub_def(n):
    if isinstance(n, Op) and n.op == "-":
        return Op("+", n.left.clone(), Op("NEG", n.right.clone()))
    if isinstance(n, Op) and n.op == "+" and isinstance(n.right, Op) and n.right.op == "NEG":
        return Op("-", n.left.clone(), n.right.left.clone())
    return None


def div_def(n):
    if isinstance(n, Op) and n.op == "/":
        return Op("*", n.left.clone(), Op("INV", n.right.clone()))
    if isinstance(n, Op) and n.op == "*" and isinstance(n.right, Op) and n.right.op == "INV":
        return Op("/", n.left.clone(), n.right.left.clone())
    return None


AXIOMS = {
    "A-Comm": a_comm,
    "A-Assoc": a_assoc,
    "A-Assoc-Rev": a_assoc_rev,
    "M-Comm": m_comm,
    "M-Assoc": m_assoc,
    "M-Assoc-Rev": m_assoc_rev,
    "Dist-Right": dist,
    "Dist-Left": dist_left,
    "Factor": factor,
    "Sub-Def": sub_def,
    "Div-Def": div_def,
}

SKILL_DESCRIPTIONS = {
    "A-Comm": "Additive Commutativity (a + b = b + a)",
    "A-Assoc": "Additive Associativity, regroup left to right",
    "A-Assoc-Rev": "Additive Associativity, regroup right to left",
    "M-Comm": "Multiplicative Commutativity (a * b = b * a)",
    "M-Assoc": "Multiplicative Associativity, regroup left to right",
    "M-Assoc-Rev": "Multiplicative Associativity, regroup right to left",
    "Dist-Right": "Distribution / Expansion (right)",
    "Dist-Left": "Distribution / Expansion (left)",
    "Factor": "Factoring (reverse distribution)",
    "Sub-Def": "Subtraction as addition of negation",
    "Div-Def": "Division as multiplication by reciprocal",
}

AXIOM_COGNITIVE_CLASS = {
    "A-Comm": "structural",
    "A-Assoc": "structural",
    "A-Assoc-Rev": "structural",
    "M-Comm": "structural",
    "M-Assoc": "structural",
    "M-Assoc-Rev": "structural",
    "Dist-Right": "rewrite",
    "Dist-Left": "rewrite",
    "Factor": "insight",
    "Sub-Def": "rewrite",
    "Div-Def": "rewrite",
}

AXIOM_MISCONCEPTION_RISK = {
    "A-Comm": 0.05,
    "A-Assoc": 0.10,
    "A-Assoc-Rev": 0.10,
    "M-Comm": 0.05,
    "M-Assoc": 0.10,
    "M-Assoc-Rev": 0.10,
    "Dist-Right": 0.35,
    "Dist-Left": 0.35,
    "Factor": 0.40,
    "Sub-Def": 0.30,
    "Div-Def": 0.25,
}

# ==========================================
# 4. Move Scanner
# ==========================================
def get_all_valid_moves(node, axiom_func):
    moves = []
    mutated = axiom_func(node)
    if mutated:
        moves.append(mutated)

    if isinstance(node, Op):
        for left_mut in get_all_valid_moves(node.left, axiom_func):
            moves.append(Op(node.op, left_mut, node.right.clone() if node.right else None))
        if node.right:
            for right_mut in get_all_valid_moves(node.right, axiom_func):
                moves.append(Op(node.op, node.left.clone(), right_mut))

    return moves
