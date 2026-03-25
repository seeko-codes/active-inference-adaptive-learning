"""Generate clean POMDP architecture diagram for poster — matches chart style."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Palette (matches poster charts) ──
WHITE = "#FFFFFF"
BG = "#FAFAFA"
BLACK = "#222222"
DARK = "#333333"
MED = "#666666"
LIGHT = "#E8E8E8"
ACCENT = "#D4880F"       # Gold — matches Active Inference bar
BLUE = "#1B6B93"         # Teal — matches Meta-Function
MUTED_GOLD = "#F5E6C8"   # Light gold fill
MUTED_BLUE = "#D8EAF0"   # Light blue fill

fig, ax = plt.subplots(figsize=(9, 14), dpi=300)
ax.set_xlim(0, 9)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor(WHITE)


def box(x, y, w, h, label, sub=None, fill=BG, edge=DARK, fontsize=11,
        bold=True, lw=1.3, radius=0.12):
    """Draw a rounded box with label and optional subtitle."""
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2,
    )
    ax.add_patch(b)
    weight = "bold" if bold else "normal"
    ty = y + h / 2 + (0.13 if sub else 0)
    ax.text(x + w / 2, ty, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=BLACK,
            zorder=5, fontfamily="serif")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.17, sub,
                ha="center", va="center", fontsize=7.5, color=MED,
                zorder=5, fontfamily="serif", style="italic")
    return b


def arrow(x1, y1, x2, y2, label=None, label_side="right", color=DARK, lw=1.3):
    """Draw a straight arrow with optional label."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        shrinkA=3, shrinkB=3),
        zorder=10,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ha = "left" if label_side == "right" else "right"
        dx = 0.15 if label_side == "right" else -0.15
        ax.text(mx + dx, my, label, ha=ha, va="center",
                fontsize=8, color=MED, fontfamily="serif", style="italic",
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.08", facecolor=WHITE,
                          edgecolor="none", alpha=0.9))


def curved_arrow(x1, y1, x2, y2, label=None, rad=0.3, loff=(0, 0), color=DARK):
    """Draw a curved arrow with optional label."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=3, shrinkB=3),
        zorder=10,
    )
    if label:
        mx = (x1 + x2) / 2 + loff[0]
        my = (y1 + y2) / 2 + loff[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=8, color=MED, fontfamily="serif", style="italic",
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.08", facecolor=WHITE,
                          edgecolor="none", alpha=0.9))


# ════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════
ax.text(4.5, 13.5, "POMDP Architecture", ha="center", va="center",
        fontsize=17, fontweight="bold", color=BLACK, fontfamily="serif")
ax.text(4.5, 13.15, "Adaptive teaching via active inference",
        ha="center", va="center", fontsize=9, color=MED, fontfamily="serif",
        style="italic")

# ════════════════════════════════════════════
# KNOWLEDGE SPACE (top)
# ════════════════════════════════════════════
box(1.5, 12.2, 6, 0.7, "Knowledge Space",
    "2,519 algebraic forms  |  10 question types  |  problem features",
    fill="#F0F0F0", edge=DARK)

# ════════════════════════════════════════════
# GENERATIVE MODEL region
# ════════════════════════════════════════════
region = FancyBboxPatch(
    (0.7, 7.5), 5.4, 4.35,
    boxstyle="round,pad=0.15",
    facecolor="#F8F8F8", edgecolor=MED,
    linewidth=0.8, linestyle=(0, (5, 3)), zorder=0,
)
ax.add_patch(region)
ax.text(1.0, 11.6, "Generative Model", fontsize=8.5, fontweight="bold",
        color=MED, fontfamily="serif", zorder=2)

# B matrix
box(1.0, 10.7, 4.8, 0.7, "Transition Model (B)",
    "how knowledge changes per problem",
    fill=MUTED_BLUE, edge=BLUE, lw=1.0)

# Hidden state
box(1.0, 8.8, 4.8, 1.5, "", fill=WHITE, edge=DARK)
ax.text(3.4, 10.05, "Hidden Student State", ha="center", va="center",
        fontsize=11, fontweight="bold", color=BLACK, fontfamily="serif", zorder=5)

# 5 dimensions as labeled boxes
dims = [
    ("RS", "Retrieval\nStrength"),
    ("SS", "Storage\nStrength"),
    ("Sch", "Schema\nLevel"),
    ("WM", "Working\nMemory"),
    ("Aff", "Affect"),
]
for i, (short, full) in enumerate(dims):
    cx = 1.5 + i * 0.88
    dim_box = FancyBboxPatch(
        (cx - 0.35, 8.95), 0.7, 0.8,
        boxstyle="round,pad=0.06",
        facecolor=MUTED_GOLD, edgecolor=ACCENT, linewidth=0.7, zorder=3,
    )
    ax.add_patch(dim_box)
    ax.text(cx, 9.5, short, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=ACCENT, fontfamily="serif", zorder=5)
    ax.text(cx, 9.15, full, ha="center", va="center",
            fontsize=5.5, color=MED, fontfamily="serif", zorder=5, linespacing=1.1)

ax.text(3.4, 8.88, "5 dims x 11 skills = 5,940 joint states",
        ha="center", va="center", fontsize=6.5, color=MED,
        fontfamily="serif", style="italic", zorder=5)

# A matrix
box(1.0, 7.7, 4.8, 0.7, "Observation Model (A)",
    "how knowledge produces responses",
    fill=MUTED_BLUE, edge=BLUE, lw=1.0)

# Arrows within generative model
arrow(3.4, 12.2, 3.4, 11.4)
arrow(3.4, 10.7, 3.4, 10.3)
arrow(3.4, 8.8, 3.4, 8.4)

# ════════════════════════════════════════════
# BELIEF UPDATE + POLICY SELECTION (right)
# ════════════════════════════════════════════
box(6.3, 10.2, 2.2, 0.8, "Belief\nUpdate",
    fill=MUTED_GOLD, edge=ACCENT, fontsize=10.5, lw=1.2)
ax.text(7.4, 9.9, "Bayesian inference", ha="center", va="center",
        fontsize=7, color=MED, fontfamily="serif", style="italic", zorder=5)

box(6.3, 8.3, 2.2, 0.9, "Policy\nSelection",
    fill=MUTED_GOLD, edge=ACCENT, fontsize=10.5, lw=1.2)
ax.text(7.4, 8.0, "min Expected\nFree Energy", ha="center", va="center",
        fontsize=7, color=MED, fontfamily="serif", style="italic", zorder=5,
        linespacing=1.1)

# Observation model → belief (curved up-right)
curved_arrow(5.8, 8.1, 6.3, 10.3, label="posterior", rad=-0.35, loff=(0.4, 0))

# Belief → B matrix (curved right-left)
curved_arrow(7.8, 11.0, 5.8, 11.2, label="prior", rad=-0.25, loff=(0.3, 0.2))

# Belief → policy
arrow(7.4, 10.2, 7.4, 9.2)

# ════════════════════════════════════════════
# OBSERVATIONS
# ════════════════════════════════════════════
box(1.5, 5.5, 6, 1.7, "", fill=WHITE, edge=DARK)
ax.text(4.5, 6.95, "Observations", ha="center", va="center",
        fontsize=11, fontweight="bold", color=BLACK, fontfamily="serif", zorder=5)

obs_items = [
    ("Accuracy", "correct / incorrect", "RS, WM, Affect"),
    ("Response Time", "fast / normal / slow", "WM, Affect"),
    ("Explanation", "low / medium / high", "Schema"),
    ("Confidence", "1-5 scale", "RS, Schema"),
]
for i, (name, vals, driven) in enumerate(obs_items):
    y_pos = 6.55 - i * 0.25
    ax.text(2.3, y_pos, name, ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=DARK, fontfamily="serif", zorder=5)
    ax.text(4.2, y_pos, vals, ha="left", va="center",
            fontsize=8, color=MED, fontfamily="serif", zorder=5)
    ax.text(6.5, y_pos, f"driven by {driven}", ha="left", va="center",
            fontsize=6.5, color=MED, fontfamily="serif", style="italic", zorder=5)

# A → observations
arrow(3.4, 7.7, 3.4, 7.2)

# Policy → actions
arrow(7.4, 8.3, 7.4, 4.55, label="selects", label_side="right")

# ════════════════════════════════════════════
# TEACHING ACTIONS
# ════════════════════════════════════════════
box(1.5, 2.8, 6, 1.7, "", fill=WHITE, edge=DARK)
ax.text(4.5, 4.25, "8 Teaching Actions", ha="center", va="center",
        fontsize=11, fontweight="bold", color=BLACK, fontfamily="serif", zorder=5)

actions = [
    "Space & Test", "Reteach", "Worked Example", "Faded Example",
    "Interleave", "Inc. Challenge", "Reduce Load", "Diagnostic Probe",
]
for i, action in enumerate(actions):
    col = i % 4
    row = i // 4
    cx = 2.3 + col * 1.5
    cy = 3.8 - row * 0.45
    ax.text(cx, cy, action, ha="center", va="center",
            fontsize=7.5, color=DARK, fontfamily="serif", zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", facecolor=LIGHT,
                      edgecolor="#BBBBBB", linewidth=0.6))

# ════════════════════════════════════════════
# STUDENT
# ════════════════════════════════════════════
box(2.5, 1.0, 4, 0.9, "Student",
    "problem + response", fontsize=13,
    fill="#F0F0F0", edge=DARK)

# Actions → student
arrow(3.5, 2.8, 3.5, 1.9, label="presents\nproblem", label_side="left")

# Student → observations
arrow(5.5, 1.9, 5.5, 5.5, label="responds", label_side="right")

# ════════════════════════════════════════════
# PER-PROBLEM LOOP (left margin)
# ════════════════════════════════════════════
# Clean vertical bracket with arrow
loop_x = 0.35
ax.annotate(
    "", xy=(loop_x, 2.0), xytext=(loop_x, 12.0),
    arrowprops=dict(arrowstyle="-|>", color=MED, lw=1.0,
                    connectionstyle="arc3,rad=0.05"),
    zorder=10,
)
ax.text(0.12, 7.0, "per-problem\nloop", ha="center", va="center",
        fontsize=7, color=MED, fontfamily="serif",
        rotation=90, style="italic", zorder=10, linespacing=1.3)

# ════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════
plt.tight_layout(pad=0.3)
fig.savefig("/Users/aatutor/adaptive-learning/poster/architecture_diagram.png",
            dpi=300, bbox_inches="tight", facecolor=WHITE)
print("Saved: poster/architecture_diagram.png")
