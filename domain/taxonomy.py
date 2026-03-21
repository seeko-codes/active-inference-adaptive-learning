"""
Skill taxonomy, confusable pairs, EI table, and schema rubrics for the algebra domain.

This module extends the existing axiom definitions with the structures needed
for the 5-state inference model: element interactivity values, confusable
skill pairs for discriminability tracking, and per-skill schema rubrics for
LLM-based explanation analysis.
"""

from domain.axioms import SKILL_DESCRIPTIONS

# ==========================================
# Skills (from axioms.py, for reference)
# ==========================================
SKILLS = list(SKILL_DESCRIPTIONS.keys())
# ['A-Comm', 'A-Assoc', 'A-Assoc-Rev', 'M-Comm', 'M-Assoc', 'M-Assoc-Rev',
#  'Dist-Right', 'Dist-Left', 'Factor', 'Sub-Def', 'Div-Def']


# ==========================================
# Confusable Pairs (for discriminability tracking)
# ==========================================
# Each pair represents skills a student might confuse with each other.
# Ordered by pedagogical importance (most commonly confused first).

CONFUSABLE_PAIRS = [
    # Structural confusion: same operation family, different grouping
    ("A-Comm", "A-Assoc"),         # both rearrange addition, different rules
    ("A-Comm", "A-Assoc-Rev"),
    ("M-Comm", "M-Assoc"),         # both rearrange multiplication
    ("M-Comm", "M-Assoc-Rev"),
    ("A-Assoc", "A-Assoc-Rev"),    # same axiom, opposite direction
    ("M-Assoc", "M-Assoc-Rev"),

    # Cross-operation confusion: additive vs multiplicative versions
    ("A-Comm", "M-Comm"),          # commutativity across operations
    ("A-Assoc", "M-Assoc"),        # associativity across operations

    # Distribution/factoring confusion (highest misconception risk)
    ("Dist-Right", "Dist-Left"),   # same operation, different side
    ("Dist-Right", "Factor"),      # distribution vs its inverse
    ("Dist-Left", "Factor"),

    # Signed operation confusion
    ("Sub-Def", "Div-Def"),        # both rewrite as inverse operations

    # Cross-family confusion: when to distribute vs when to associate
    ("Dist-Right", "M-Assoc"),     # both involve multiplication rearrangement
    ("Dist-Left", "M-Assoc-Rev"),
    ("Factor", "M-Comm"),          # both involve multiplication structure
]


# ==========================================
# Element Interactivity (EI) Base Table
# ==========================================
# Base EI values by tier. These represent the number of elements a student
# must hold in working memory simultaneously to process a problem at that tier,
# assuming NO schemas (worst case for a complete novice).
#
# density_load from cognitive_features.py maps directly to EI.
# These tier-level defaults are used when per-problem density_load is unavailable.

TIER_BASE_EI = {
    0: 2,   # Raw arithmetic: just two operands
    1: 3,   # Commutativity: two operands + the swap rule
    2: 4,   # Associativity: three operands + grouping rule
    3: 5,   # Order of operations: mixed ops + precedence
    4: 6,   # Distribution: factor + sum + the bridge between them
    5: 5,   # Signed operations: operation + sign handling
    6: 7,   # Signed distribution: distribution + sign complexity
    7: 8,   # Complex combinations: FOIL, multiple steps
}

# Schema reduction per skill when fully schematized.
# A full schema compresses these elements into one chunk.
SCHEMA_REDUCTION = {
    "A-Comm": 1,       # Swap is automatic
    "A-Assoc": 1,      # Regrouping is automatic
    "A-Assoc-Rev": 1,
    "M-Comm": 1,
    "M-Assoc": 1,
    "M-Assoc-Rev": 1,
    "Dist-Right": 2,   # Entire distribution step becomes one chunk
    "Dist-Left": 2,
    "Factor": 2,       # Recognizing common factor becomes automatic
    "Sub-Def": 1,      # Subtraction-as-negation is automatic
    "Div-Def": 1,      # Division-as-reciprocal is automatic
}


def effective_ei(base_ei, student_schemas, relevant_skills):
    """
    Compute effective element interactivity for a student on a problem.

    Args:
        base_ei: Base EI from the problem's density_load or tier default
        student_schemas: dict mapping skill -> schema level (0=none, 1=partial, 2=full)
        relevant_skills: list of skills involved in this problem

    Returns:
        Effective EI (minimum 1)
    """
    reduction = 0
    for skill in relevant_skills:
        level = student_schemas.get(skill, 0)
        if level == 2:  # Full schema
            reduction += SCHEMA_REDUCTION.get(skill, 0)
        elif level == 1:  # Partial schema
            reduction += SCHEMA_REDUCTION.get(skill, 0) * 0.5
    return max(1, base_ei - reduction)


# ==========================================
# Schema Rubrics (for LLM explanation analysis)
# ==========================================
# Per-skill rubrics that the LLM uses to classify student explanations
# into schema levels: none (0), partial (1), full (2).

SCHEMA_RUBRICS = {
    "A-Comm": {
        "skill_name": "Additive Commutativity",
        "description": "a + b = b + a",
        "levels": {
            "none": (
                "Student does not recognize that order can change in addition. "
                "May recompute or say the expressions are different. "
                "Treats a + b and b + a as separate facts."
            ),
            "partial": (
                "Student recognizes the swap but explains it mechanically: "
                "'you just switch them around.' Cannot articulate WHY order "
                "doesn't matter or connect it to the meaning of addition."
            ),
            "full": (
                "Student understands commutativity as a property of addition itself. "
                "Can explain that combining quantities doesn't depend on order. "
                "Applies it automatically without conscious effort."
            ),
        },
    },
    "A-Assoc": {
        "skill_name": "Additive Associativity (left to right)",
        "description": "(a + b) + c = a + (b + c)",
        "levels": {
            "none": (
                "Student cannot regroup addition. Treats (a+b)+c and a+(b+c) "
                "as different computations. May get confused by parentheses."
            ),
            "partial": (
                "Student can regroup when prompted but doesn't see why it works. "
                "Follows the rule mechanically. May struggle with four or more terms."
            ),
            "full": (
                "Student understands that addition is associative — grouping "
                "doesn't affect the total. Regroups fluently to simplify computation. "
                "Sees it as a fundamental property, not a rule to memorize."
            ),
        },
    },
    "A-Assoc-Rev": {
        "skill_name": "Additive Associativity (right to left)",
        "description": "a + (b + c) = (a + b) + c",
        "levels": {
            "none": "Same as A-Assoc none — cannot regroup.",
            "partial": "Can regroup in this direction when shown, but doesn't spontaneously choose it.",
            "full": "Fluently regroups in either direction as needed for simplification.",
        },
    },
    "M-Comm": {
        "skill_name": "Multiplicative Commutativity",
        "description": "a * b = b * a",
        "levels": {
            "none": (
                "Student does not recognize that multiplication order can change. "
                "May think 3*5 and 5*3 require different computations."
            ),
            "partial": (
                "Recognizes the swap rule but applies it mechanically. "
                "Cannot explain why multiplication is commutative."
            ),
            "full": (
                "Understands commutativity as inherent to multiplication. "
                "Uses it strategically (e.g., reordering to simplify). "
                "Sees it as the same structure as additive commutativity."
            ),
        },
    },
    "M-Assoc": {
        "skill_name": "Multiplicative Associativity (left to right)",
        "description": "(a * b) * c = a * (b * c)",
        "levels": {
            "none": "Cannot regroup multiplication. Confused by nested products.",
            "partial": "Can regroup when prompted but doesn't see strategic value.",
            "full": (
                "Fluently regroups products. Understands that multiplication "
                "associativity means grouping doesn't change the product. "
                "Uses it to simplify computations."
            ),
        },
    },
    "M-Assoc-Rev": {
        "skill_name": "Multiplicative Associativity (right to left)",
        "description": "a * (b * c) = (a * b) * c",
        "levels": {
            "none": "Same as M-Assoc none.",
            "partial": "Can regroup in this direction when shown.",
            "full": "Fluently regroups in either direction.",
        },
    },
    "Dist-Right": {
        "skill_name": "Distribution (right)",
        "description": "a * (b + c) = a*b + a*c",
        "levels": {
            "none": (
                "Student cannot distribute. May try to add first (ignoring "
                "order of operations) or apply the multiplier to only one term. "
                "Common error: a*(b+c) = ab + c."
            ),
            "partial": (
                "Can distribute in simple cases but struggles with nested expressions "
                "or multiple distributions. Follows FOIL mechanically without "
                "understanding it as repeated distribution. May forget to distribute "
                "to all terms."
            ),
            "full": (
                "Understands distribution as 'multiplication spreads over addition.' "
                "Applies it fluently in any context — simple, nested, with signed terms. "
                "Sees FOIL as a special case of distribution, not a separate rule."
            ),
        },
    },
    "Dist-Left": {
        "skill_name": "Distribution (left)",
        "description": "(a + b) * c = a*c + b*c",
        "levels": {
            "none": (
                "Cannot distribute when the sum is on the left. May not recognize "
                "this as the same operation as right-distribution."
            ),
            "partial": (
                "Can distribute left when prompted but doesn't spontaneously recognize "
                "the pattern. Treats left- and right-distribution as different rules."
            ),
            "full": (
                "Sees left-distribution as the same operation as right-distribution "
                "(via commutativity of multiplication). Applies fluently regardless "
                "of which side the sum is on."
            ),
        },
    },
    "Factor": {
        "skill_name": "Factoring (reverse distribution)",
        "description": "a*b + a*c = a*(b + c)",
        "levels": {
            "none": (
                "Student cannot identify common factors. Does not see factoring "
                "as the reverse of distribution. May not recognize that ab + ac "
                "has a shared structure."
            ),
            "partial": (
                "Can factor when the common factor is obvious and in the same position "
                "(e.g., 3x + 3y). Struggles when factors are in different positions "
                "or require commutativity to spot. Doesn't connect factoring to distribution."
            ),
            "full": (
                "Fluently identifies common factors regardless of position. "
                "Understands factoring as the inverse of distribution. Can factor "
                "complex expressions and recognizes when factoring simplifies a problem."
            ),
        },
    },
    "Sub-Def": {
        "skill_name": "Subtraction as Addition of Negation",
        "description": "a - b = a + (-b)",
        "levels": {
            "none": (
                "Student treats subtraction as a fundamentally different operation "
                "from addition. Cannot rewrite a - b as a + (-b). May be confused "
                "by negative numbers."
            ),
            "partial": (
                "Can rewrite subtraction as addition of negation when prompted, "
                "but doesn't spontaneously use this to simplify. May struggle "
                "with chained subtractions or signs."
            ),
            "full": (
                "Understands subtraction as a special case of addition. "
                "Fluently converts between forms. Uses this understanding to "
                "simplify complex expressions with mixed addition and subtraction."
            ),
        },
    },
    "Div-Def": {
        "skill_name": "Division as Multiplication by Reciprocal",
        "description": "a / b = a * (1/b)",
        "levels": {
            "none": (
                "Student treats division as fundamentally different from multiplication. "
                "Cannot rewrite a / b as a * (1/b)."
            ),
            "partial": (
                "Can rewrite division as multiplication by reciprocal when prompted. "
                "Doesn't spontaneously use this to simplify or connect it to "
                "fraction arithmetic."
            ),
            "full": (
                "Understands division as multiplication by the reciprocal. "
                "Uses this to simplify division in algebraic expressions. "
                "Connects it to fraction operations naturally."
            ),
        },
    },
}
