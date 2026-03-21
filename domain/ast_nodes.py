import json
import uuid

# ==========================================
# 1. AST Nodes
# ==========================================
class Node:
    pass


class Leaf(Node):
    def __init__(self, val):
        self.val = val

    def clone(self):
        return Leaf(self.val)

    def to_string(self):
        return str(self.val)

    def get_variables(self):
        if isinstance(self.val, str) and self.val.isalpha():
            return {self.val}
        return set()


class Op(Node):
    def __init__(self, op, left, right=None):
        self.op = op
        self.left = left
        self.right = right

    def clone(self):
        return Op(self.op, self.left.clone(), self.right.clone() if self.right else None)
   
    def to_string(self):
        if self.op == "NEG":
            return f"(-{self.left.to_string()})"
        if self.op == "INV":
            return f"(1/{self.left.to_string()})"
        return f"({self.left.to_string()} {self.op} {self.right.to_string()})"

    def get_variables(self):
        vs = self.left.get_variables()
        if self.right:
            vs |= self.right.get_variables()
        return vs


# ==========================================
# 2. Pretty Printer
# ==========================================
def pretty_print(node, parent_op=None, is_right=False):
    if isinstance(node, Leaf):
        return str(node.val)

    if node.op == "NEG":
        inner = pretty_print(node.left, "NEG")
        if isinstance(node.left, Op) and node.left.op in ("+", "-"):
            return f"-({inner})"
        return f"-{inner}"

    if node.op == "INV":
        inner = pretty_print(node.left, "INV")
        if isinstance(node.left, Op):
            return f"1/({inner})"
        return f"1/{inner}"

    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
    my_prec = precedence.get(node.op, 0)
    parent_prec = precedence.get(parent_op, 0)

    left_str = pretty_print(node.left, node.op, is_right=False)
    right_str = pretty_print(node.right, node.op, is_right=True)

    if node.op == "*":
        left_is_simple = isinstance(node.left, Leaf)
        right_is_simple = isinstance(node.right, Leaf)

        def needs_mult_parens(child_node, child_str):
            if isinstance(child_node, Leaf):
                return False
            if isinstance(child_node, Op) and child_node.op in ("+", "-"):
                return False
            if isinstance(child_node, Op) and child_node.op in ("*", "/"):
                return True
            return True

        if left_is_simple and right_is_simple:
            result = f"{left_str}{right_str}"
        elif left_is_simple:
            if needs_mult_parens(node.right, right_str):
                result = f"{left_str}({right_str})"
            else:
                result = f"{left_str}{right_str}"
        elif right_is_simple:
            if needs_mult_parens(node.left, left_str):
                result = f"({left_str}){right_str}"
            else:
                result = f"{left_str}{right_str}"
        else:
            l_wrap = f"({left_str})" if needs_mult_parens(node.left, left_str) else left_str
            r_wrap = f"({right_str})" if needs_mult_parens(node.right, right_str) else right_str
            result = f"{l_wrap}{r_wrap}"
    else:
        op_symbol = {"+": " + ", "-": " - ", "/": " / "}.get(node.op, f" {node.op} ")
        result = f"{left_str}{op_symbol}{right_str}"

    needs_parens = False
    if parent_op and my_prec < parent_prec:
        needs_parens = True
    if is_right and parent_op in ("-", "/") and my_prec == parent_prec:
        needs_parens = True

    if needs_parens:
        result = f"({result})"

    return result


def substitute_numbers(node, var_map):
    if isinstance(node, Leaf):
        if node.val in var_map:
            return Leaf(var_map[node.val])
        return node.clone()
    new_left = substitute_numbers(node.left, var_map)
    new_right = substitute_numbers(node.right, var_map) if node.right else None
    return Op(node.op, new_left, new_right)


def evaluate(node):
    if isinstance(node, Leaf):
        return float(node.val)

    if node.op == "NEG":
        return -evaluate(node.left)

    if node.op == "INV":
        val = evaluate(node.left)
        if val == 0:
            return None
        return 1.0 / val

    l = evaluate(node.left)
    r = evaluate(node.right) if node.right else None

    if l is None or r is None:
        return None

    if node.op == "+":
        return l + r
    if node.op == "-":
        return l - r
    if node.op == "*":
        return l * r
    if node.op == "/":
        if r == 0:
            return None
        return l / r

    return None
