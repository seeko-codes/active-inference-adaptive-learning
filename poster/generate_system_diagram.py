"""Generate POMDP system diagram for poster — same style as chart figures."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Same rcParams as charts ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "text.color": "#222222",
    "figure.facecolor": "white",
})

# ── Palette (shared with charts) ──
WHITE   = "#FFFFFF"
BG      = "#FAFAFA"
BLACK   = "#222222"
DARK    = "#333333"
MED     = "#777777"
LIGHT   = "#ECECEC"
ACCENT  = "#D4880F"       # Gold
BLUE    = "#1B6B93"       # Teal
GOLD_BG = "#FDF3E1"       # Soft gold fill
BLUE_BG = "#E4F0F5"       # Soft blue fill
GREEN   = "#5C8A5C"       # For student/actions

OUT = "/Users/aatutor/adaptive-learning/poster"

fig, ax = plt.subplots(figsize=(10, 13), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis("off")
fig.patch.set_facecolor(WHITE)


# ── Drawing helpers ──

def rounded_box(x, y, w, h, fill=BG, edge=DARK, lw=1.2):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                       facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(b)
    return b

def label(x, y, text, fs=11, bold=True, color=BLACK, **kw):
    w = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight=w, color=color, fontfamily="serif", zorder=5, **kw)

def sublabel(x, y, text, fs=8, color=MED, **kw):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=color, fontfamily="serif", style="italic", zorder=5, **kw)

def straight_arrow(x1, y1, x2, y2, color=DARK, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=4, shrinkB=4), zorder=10)

def curved(x1, y1, x2, y2, rad=0.3, color=DARK, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=4, shrinkB=4), zorder=10)

def arrow_label(x, y, text, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=7.5, color=MED,
            fontfamily="serif", style="italic", zorder=11,
            bbox=dict(boxstyle="round,pad=0.06", fc=WHITE, ec="none", alpha=0.95))


# ═══════════════════════════════════════
# LAYOUT — top to bottom
# ═══════════════════════════════════════

# ── Title ──
label(5, 12.55, "System Architecture", fs=16)
sublabel(5, 12.2, "POMDP with active inference for adaptive teaching", fs=9.5)

# ── Knowledge Space ──
rounded_box(2, 11.35, 6, 0.6, fill=LIGHT, edge=DARK)
label(5, 11.72, "Knowledge Space", fs=11)
sublabel(5, 11.5, "2,519 algebraic forms  |  10 question types", fs=7.5)

straight_arrow(5, 11.35, 5, 10.95)

# ── Generative Model region ──
region = FancyBboxPatch((0.8, 6.9), 5.6, 4.0,
                        boxstyle="round,pad=0.15",
                        facecolor="#F7F7F7", edgecolor="#CCCCCC",
                        linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
ax.add_patch(region)
ax.text(1.25, 10.65, "Generative Model", ha="left", va="center",
        fontsize=8, fontweight="bold", color=MED, fontfamily="serif", zorder=5)

# ── Transition Model B ──
rounded_box(1.1, 9.95, 5.0, 0.65, fill=BLUE_BG, edge=BLUE, lw=1.0)
label(3.6, 10.35, "State Transition Model (B)", fs=10.5)
sublabel(3.6, 10.1, "how student knowledge changes per problem", fs=7)

straight_arrow(3.6, 9.95, 3.6, 9.65)

# ── Hidden Student State ──
rounded_box(1.1, 7.9, 5.0, 1.7, fill=WHITE, edge=DARK, lw=1.3)
label(3.6, 9.35, "Hidden Student State", fs=11.5)

# 5 cognitive dimensions — give them room
dims = [
    ("RS", "Retrieval Str."),
    ("SS", "Storage Str."),
    ("Sch", "Schema Level"),
    ("WM", "Working Mem."),
    ("Aff", "Affect"),
]
dim_w, dim_h = 0.85, 0.75
dim_start_x = 1.35
dim_gap = 0.95
dim_y = 8.1

for i, (short, full) in enumerate(dims):
    cx = dim_start_x + i * dim_gap
    d = FancyBboxPatch((cx, dim_y), dim_w, dim_h,
                       boxstyle="round,pad=0.06",
                       facecolor=GOLD_BG, edgecolor=ACCENT, linewidth=0.8, zorder=3)
    ax.add_patch(d)
    label(cx + dim_w/2, dim_y + dim_h * 0.62, short, fs=10, color=ACCENT)
    sublabel(cx + dim_w/2, dim_y + dim_h * 0.25, full, fs=6, color="#996B15")

sublabel(3.6, 7.97, "5 dimensions x 11 skills = 5,940 joint states", fs=7, color=MED)

straight_arrow(3.6, 7.9, 3.6, 7.6)

# ── Observation Model A ──
rounded_box(1.1, 7.0, 5.0, 0.55, fill=BLUE_BG, edge=BLUE, lw=1.0)
label(3.6, 7.35, "Observation Model (A)", fs=10.5)
sublabel(3.6, 7.12, "how knowledge produces observable responses", fs=7)

# ── Belief Update (right side) ──
rounded_box(6.8, 9.55, 2.5, 0.85, fill=GOLD_BG, edge=ACCENT, lw=1.3)
label(8.05, 10.1, "Belief Update", fs=11)
sublabel(8.05, 9.77, "Bayesian inference", fs=7.5, color="#996B15")

# ── Policy Selection (right side) ──
rounded_box(6.8, 7.9, 2.5, 1.0, fill=GOLD_BG, edge=ACCENT, lw=1.3)
label(8.05, 8.55, "Policy Selection", fs=11)
sublabel(8.05, 8.18, "min Expected Free Energy", fs=7.5, color="#996B15")
sublabel(8.05, 7.97, "G = epistemic + pragmatic", fs=6.5, color=MED)

# Arrows: Obs Model → Belief Update (curved up-right)
curved(6.1, 7.55, 6.8, 9.7, rad=-0.3, color=ACCENT)
arrow_label(6.9, 8.5, "posterior")

# Belief → Transition (curved right-to-left)
curved(8.5, 10.4, 6.1, 10.45, rad=-0.2, color=ACCENT)
arrow_label(7.3, 10.7, "prior")

# Belief → Policy
straight_arrow(8.05, 9.55, 8.05, 8.9, color=ACCENT)

# ═══════════════════════════════════════
# OBSERVATIONS
# ═══════════════════════════════════════
rounded_box(1.5, 4.8, 7, 1.8, fill=WHITE, edge=DARK, lw=1.1)
label(5, 6.35, "4 Observation Channels", fs=11)

# Table header
for hx, htxt in [(2.6, "Channel"), (4.6, "Values"), (7.0, "Driven By")]:
    ax.text(hx, 6.05, htxt, ha="left", va="center", fontsize=7.5,
            fontweight="bold", color=MED, fontfamily="serif", zorder=5)

# Subtle header line
ax.plot([2.0, 8.0], [5.94, 5.94], color="#DDDDDD", lw=0.6, zorder=5)

obs = [
    ("Accuracy",      "correct / incorrect",     "RS, WM, Affect"),
    ("Response Time",  "fast / normal / slow",    "WM, Affect"),
    ("Explanation",    "low / med / high",        "Schema"),
    ("Confidence",     "1-5 scale",               "RS, Schema"),
]
for i, (ch, vals, driven) in enumerate(obs):
    row_y = 5.7 - i * 0.24
    ax.text(2.6, row_y, ch, ha="left", va="center", fontsize=8.5,
            fontweight="bold", color=DARK, fontfamily="serif", zorder=5)
    ax.text(4.6, row_y, vals, ha="left", va="center", fontsize=8,
            color=MED, fontfamily="serif", zorder=5)
    ax.text(7.0, row_y, driven, ha="left", va="center", fontsize=8,
            color=ACCENT, fontfamily="serif", fontweight="bold", zorder=5)

# Obs Model → Observations
straight_arrow(3.6, 7.0, 3.6, 6.6)

# Policy → Teaching Actions
straight_arrow(8.05, 7.9, 8.05, 3.95, color=ACCENT)
arrow_label(8.35, 5.9, "selects")

# ═══════════════════════════════════════
# TEACHING ACTIONS
# ═══════════════════════════════════════
rounded_box(1.5, 2.2, 7, 1.7, fill=WHITE, edge=DARK, lw=1.1)
label(5, 3.7, "8 Teaching Actions", fs=11)

actions = [
    "Space & Test", "Reteach", "Worked Example", "Faded Example",
    "Interleave", "Inc. Challenge", "Reduce Load", "Diagnostic Probe",
]
for i, action in enumerate(actions):
    col = i % 4
    row = i // 4
    cx = 2.4 + col * 1.6
    cy = 3.2 - row * 0.5
    ax.text(cx, cy, action, ha="center", va="center",
            fontsize=8, color=DARK, fontfamily="serif", zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", facecolor=LIGHT,
                      edgecolor="#C0C0C0", linewidth=0.6))

# ═══════════════════════════════════════
# STUDENT
# ═══════════════════════════════════════
rounded_box(2.8, 0.6, 4.4, 0.85, fill=LIGHT, edge=DARK, lw=1.2)
label(5, 1.12, "Student", fs=13)
sublabel(5, 0.82, "problem + response", fs=8)

# Actions → Student
straight_arrow(3.8, 2.2, 3.8, 1.45)
arrow_label(3.4, 1.85, "presents problem")

# Student → Observations
straight_arrow(6.2, 1.45, 6.2, 4.8)
arrow_label(6.55, 3.1, "responds")

# ═══════════════════════════════════════
# PER-PROBLEM LOOP (left margin)
# ═══════════════════════════════════════
lx = 0.55
# Vertical line down
ax.plot([lx, lx], [11.2, 1.2], color="#CCCCCC", lw=1.0, linestyle="-", zorder=1)
# Arrow at bottom
ax.annotate("", xy=(lx, 1.2), xytext=(lx, 1.8),
            arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1.0), zorder=1)
# Arrow at top
ax.annotate("", xy=(lx, 11.2), xytext=(lx, 10.6),
            arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1.0), zorder=1)
# Label
ax.text(0.25, 6.2, "per-problem loop", ha="center", va="center",
        fontsize=7, color="#AAAAAA", fontfamily="serif",
        rotation=90, style="italic", zorder=1)

# ═══════════════════════════════════════
# SAVE
# ═══════════════════════════════════════
plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.02)
fig.savefig(f"{OUT}/system_diagram.png", dpi=300, bbox_inches="tight", facecolor=WHITE)
print("Saved: poster/system_diagram.png")
