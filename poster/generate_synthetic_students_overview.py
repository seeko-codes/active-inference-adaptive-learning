"""Generate 3 dense infographic panels covering the synthetic student framework."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Style ──
BLACK = "#111111"
DARK = "#222222"
MED = "#555555"
LIGHT = "#999999"
VERY_LIGHT = "#E0E0E0"
BG = "#FAFAFA"
WHITE = "#FFFFFF"
GOLD = "#C17817"
BLUE = "#2E6B8A"
RED = "#B5453A"
GREEN = "#3A7D44"
PURPLE = "#6B4C9A"
TEAL = "#2A9D8F"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "text.color": DARK,
    "axes.facecolor": WHITE,
    "figure.facecolor": BG,
})

OUT = "/Users/aatutor/adaptive-learning/poster"


def rounded_box(ax, x, y, w, h, color=VERY_LIGHT, edge=MED, lw=1.0, alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor=edge, linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(box)
    return box


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: WHAT'S INSIDE A SYNTHETIC STUDENT
# ═══════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(14, 18), dpi=200)
ax1.set_xlim(0, 14)
ax1.set_ylim(0, 18)
ax1.axis("off")
fig1.patch.set_facecolor(BG)

# Title
ax1.text(7, 17.5, "Figure 1: Anatomy of a Synthetic Student", ha="center", va="center",
         fontsize=20, fontweight="bold", color=BLACK, fontfamily="serif")
ax1.text(7, 17.05, "16 cognitive parameters define a student. 5 hidden dimensions evolve. 4 observation channels reveal state.",
         ha="center", va="center", fontsize=10, color=MED, fontfamily="serif", style="italic")

# ── Section A: The 5 Hidden State Dimensions ──
ax1.text(0.5, 16.5, "A", fontsize=16, fontweight="bold", color=GOLD)
ax1.text(1.1, 16.5, "5 Hidden State Dimensions  (never directly observed)", fontsize=13, fontweight="bold", color=BLACK)

dims = [
    ("RS", "Retrievability", "Can they recall it now?", "5 bins  [0, 1]", BLUE,
     ["0.0-0.2: very low", "0.2-0.4: low", "0.4-0.6: moderate", "0.6-0.8: high", "0.8-1.0: very high"]),
    ("SS", "Stability", "How deep is the memory?", "4 bins  [0.1, +inf) days", TEAL,
     ["<1 day: never learned", "1-5 days: fragile", "5-30 days: solid", ">30 days: deep"]),
    ("Sch", "Schema", "Have they organized the idea?", "3 bins  [0, 1]", GREEN,
     ["none: no structure", "partial: some connections", "full: compressed understanding"]),
    ("WM", "Working Memory", "Are they overloaded?", "3 bins  [0, 1]", PURPLE,
     ["low: headroom", "moderate: sweet spot", "high: near capacity"]),
    ("Aff", "Affect", "Frustrated or bored?", "3 bins  {F, E, B}", RED,
     ["frustrated: vicious cycle risk", "engaged: ideal state", "bored: needs challenge"]),
]

for i, (abbr, name, question, bins, color, levels) in enumerate(dims):
    bx = 0.3
    by = 15.8 - i * 1.55
    # Colored left bar
    ax1.fill_between([bx, bx + 0.15], by - 0.55, by + 0.55, color=color, alpha=0.8, zorder=3)
    # Abbreviation
    ax1.text(bx + 0.55, by + 0.3, abbr, fontsize=12, fontweight="bold", color=color, fontfamily="serif")
    # Full name + question
    ax1.text(bx + 1.4, by + 0.3, f"{name}  --  {question}", fontsize=10, fontweight="bold", color=DARK)
    # Bins
    ax1.text(bx + 1.4, by + 0.0, bins, fontsize=8.5, color=MED)
    # Levels
    for j, lvl in enumerate(levels):
        col = j % 3
        row = j // 3
        lx = bx + 1.4 + col * 4.2
        ly = by - 0.3 - row * 0.25
        ax1.text(lx, ly, lvl, fontsize=7.5, color=LIGHT, fontfamily="serif")

# Joint state space callout
rounded_box(ax1, 0.3, 8.0, 13.3, 0.55, color="#FFF8F0", edge=GOLD, lw=1.5)
ax1.text(7, 8.35, "Joint state space:  5 x 4 x 3 x 3 x 3  =  540 states/skill  x  11 skills  =  5,940 total states/student",
         ha="center", fontsize=10, fontweight="bold", color=GOLD)

# ── Section B: 4 Observation Channels ──
ax1.text(0.5, 7.4, "B", fontsize=16, fontweight="bold", color=GOLD)
ax1.text(1.1, 7.4, "4 Observation Channels  (what you actually see)", fontsize=13, fontweight="bold", color=BLACK)

obs_data = [
    ("Accuracy", "correct / incorrect", "RS, Schema, WM, Affect", "#E8F0F4"),
    ("Response Time", "fast / normal / slow", "WM, Affect", "#E8F4EE"),
    ("Explanation Quality", "low / medium / high", "Schema", "#F0E8F4"),
    ("Confidence Report", "1-5 scale", "RS, Schema", "#F4EEE8"),
]

for i, (name, values, drivers, bg) in enumerate(obs_data):
    bx = 0.5 + i * 3.35
    by = 6.4
    rounded_box(ax1, bx, by, 3.0, 0.85, color=bg, edge=MED, lw=0.8)
    ax1.text(bx + 1.5, by + 0.6, name, ha="center", fontsize=9.5, fontweight="bold", color=DARK)
    ax1.text(bx + 1.5, by + 0.35, values, ha="center", fontsize=8, color=MED)
    ax1.text(bx + 1.5, by + 0.1, f"driven by: {drivers}", ha="center", fontsize=7.5, color=LIGHT, style="italic")

# Arrow from hidden states to observations
ax1.annotate("", xy=(7, 7.25), xytext=(7, 7.95),
             arrowprops=dict(arrowstyle="-|>", color=MED, lw=1.5))
ax1.text(7.6, 7.6, "A matrices\n(noisy mapping)", fontsize=8, color=MED, style="italic", ha="left")

# ── Section C: The 16-Parameter Cognitive Profile ──
ax1.text(0.5, 5.8, "C", fontsize=16, fontweight="bold", color=GOLD)
ax1.text(1.1, 5.8, "16-Parameter Cognitive Profile  (fully defines one student)", fontsize=13, fontweight="bold", color=BLACK)

param_groups = [
    ("Prior Knowledge", ["initial_rs", "initial_ss", "initial_schema"], "(per skill)", BLUE),
    ("Working Memory", ["wm_capacity (2-7)", "wm_recovery_rate"], "", PURPLE),
    ("Learning", ["schema_formation_rate", "ss_growth_rate", "rs_recovery_rate"], "", GREEN),
    ("Retention", ["forgetting_rate"], "(power law decay)", TEAL),
    ("Affect", ["frustration_threshold", "boredom_threshold", "affect_inertia", "engagement_baseline"], "", RED),
    ("Response", ["base_response_time", "response_time_variance", "explanation_quality"], "", MED),
    ("Metacognition", ["metacognitive_bias"], "(+ = overconfident)", GOLD),
]

cy = 5.35
for group_name, params, note, color in param_groups:
    ax1.text(0.7, cy, group_name, fontsize=9, fontweight="bold", color=color)
    if note:
        ax1.text(0.7 + len(group_name) * 0.08 + 0.7, cy, note, fontsize=7, color=LIGHT, style="italic")
    param_text = "   ".join(params)
    ax1.text(0.7, cy - 0.22, param_text, fontsize=7.5, color=MED, fontfamily="monospace")
    cy -= 0.6

# ── Section D: How State -> Observation Works ──
ax1.text(0.5, 1.4, "D", fontsize=16, fontweight="bold", color=GOLD)
ax1.text(1.1, 1.4, "Example: How hidden state produces an observation", fontsize=13, fontweight="bold", color=BLACK)

example_text = (
    "Student state:  RS=high (0.75)  +  Schema=full  +  WM=low  +  Affect=engaged  +  EI=medium\n"
    "  --> P(correct)           ~ 0.90       (high RS + full schema + low load = easy)\n"
    "  --> P(fast response)     ~ 0.50       (low WM = not struggling)\n"
    "  --> P(high explanation)  ~ 0.75       (full schema = can articulate)\n"
    "  --> P(high confidence)   ~ 0.70       (high RS + full schema = calibrated)"
)
ax1.text(0.7, 0.95, example_text, fontsize=8.5, fontfamily="monospace", color=DARK,
         verticalalignment="top", linespacing=1.6)

plt.tight_layout(pad=1.0)
fig1.savefig(f"{OUT}/infographic_1_anatomy.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: infographic_1_anatomy.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: WHO ARE THE STUDENTS
# ═══════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(14, 18), dpi=200)
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 18)
ax2.axis("off")
fig2.patch.set_facecolor(BG)

# Title
ax2.text(7, 17.5, "Figure 2: Who Are the Synthetic Students?", ha="center", va="center",
         fontsize=20, fontweight="bold", color=BLACK)
ax2.text(7, 17.05, "8 archetypes cover the failure landscape. Population sampling uses empirically grounded distributions.",
         ha="center", fontsize=10, color=MED, style="italic")

# ── Section A: 8 Archetypes ──
ax2.text(0.5, 16.5, "A", fontsize=16, fontweight="bold", color=GOLD)
ax2.text(1.1, 16.5, "8 Archetypes  (stress-test every failure mode)", fontsize=13, fontweight="bold", color=BLACK)

archetypes = [
    ("Novice",           "Blank slate, slow learner",         4.0, 0.15, "1.3x", 0.60, "#F4E8E8"),
    ("Fast Learner",     "No prior knowledge, high capacity", 6.0, 0.50, "0.7x", 0.80, "#E8F4E8"),
    ("Partial Knowledge","Strong on basics, gaps remain",     5.0, 0.30, "1.0x", 0.70, "#E8ECF4"),
    ("Forgetful",        "Learns in session, rapid decay",    5.0, 0.30, "1.8x", 0.70, "#F4F0E8"),
    ("Low WM",           "Understands but overloads",         3.0, 0.20, "1.1x", 0.55, "#F0E8F4"),
    ("Anxious",          "Adequate knowledge, anxiety eats WM", 4.5, 0.25, "1.2x", 0.50, "#F4E8EC"),
    ("Overconfident",    "Moderate knowledge, resists help",  5.0, 0.25, "1.0x", 0.75, "#ECF4E8"),
    ("Advanced",         "Strong prior, needs maintenance",   6.0, 0.50, "0.6x", 0.85, "#E8F4F0"),
]

# Header
hy = 16.05
ax2.text(0.5, hy, "Archetype", fontsize=8, fontweight="bold", color=MED)
ax2.text(3.8, hy, "Description", fontsize=8, fontweight="bold", color=MED)
ax2.text(8.0, hy, "WM Cap", fontsize=8, fontweight="bold", color=MED)
ax2.text(9.3, hy, "Schema Rate", fontsize=8, fontweight="bold", color=MED)
ax2.text(10.9, hy, "Forget", fontsize=8, fontweight="bold", color=MED)
ax2.text(12.0, hy, "Frust. Thresh", fontsize=8, fontweight="bold", color=MED)
ax2.plot([0.4, 13.6], [hy - 0.1, hy - 0.1], color=VERY_LIGHT, lw=1)

for i, (name, desc, wm, schema, forget, frust, bg) in enumerate(archetypes):
    ry = 15.7 - i * 0.55
    # Background stripe
    if i % 2 == 0:
        ax2.fill_between([0.3, 13.7], ry - 0.22, ry + 0.25, color=bg, alpha=0.5, zorder=0)

    ax2.text(0.5, ry, name, fontsize=9, fontweight="bold", color=DARK)
    ax2.text(3.8, ry, desc, fontsize=8, color=MED)
    # WM capacity bar
    bar_w = wm / 7.0 * 0.8
    ax2.barh(ry, bar_w, height=0.18, left=8.0, color=PURPLE, alpha=0.6, zorder=3)
    ax2.text(8.0 + bar_w + 0.08, ry, f"{wm}", fontsize=8, color=DARK, va="center")
    # Schema rate bar
    bar_s = schema / 0.5 * 0.8
    ax2.barh(ry, bar_s, height=0.18, left=9.3, color=GREEN, alpha=0.6, zorder=3)
    ax2.text(9.3 + bar_s + 0.08, ry, f"{schema}", fontsize=8, color=DARK, va="center")
    # Forget text
    ax2.text(10.9, ry, forget, fontsize=9, color=RED if float(forget.replace("x","")) > 1.0 else DARK, va="center")
    # Frustration threshold bar
    bar_f = frust * 0.8
    ax2.barh(ry, bar_f, height=0.18, left=12.0, color=RED, alpha=0.4, zorder=3)
    ax2.text(12.0 + bar_f + 0.08, ry, f"{frust}", fontsize=8, color=DARK, va="center")

# ── Section B: Population Sampling Distributions ──
ax2.text(0.5, 11.1, "B", fontsize=16, fontweight="bold", color=GOLD)
ax2.text(1.1, 11.1, "Population Sampling  (continuous random students from empirical distributions)",
         fontsize=13, fontweight="bold", color=BLACK)

sampling = [
    ("WM capacity",            "Normal(4.0, 1.5), clipped [2, 7]",  "Cowan 2001"),
    ("Math anxiety prevalence", "Bernoulli(0.20) = 20% chance",     "Ashcraft 2002"),
    ("  if anxious, WM loss",  "Uniform(0.8, 1.5) items",           "Attentional control theory"),
    ("Schema formation rate",  "Normal(0.3 + 0.1 * z_wm, 0.1)",    "Correlated with WM, r~0.5"),
    ("Forgetting rate",        "LogNormal(0, 0.35)",                 "FSRS empirical data"),
    ("Frustration threshold",  "Normal(0.7 - 0.2*anxiety, 0.1)",    "Inversely related to anxiety"),
    ("Metacognitive bias",     "Normal(0, 0.2)",                     "Positive = overconfident"),
]

sy = 10.65
for name, dist, source in sampling:
    ax2.text(0.7, sy, name, fontsize=9, fontweight="bold" if not name.startswith(" ") else "normal", color=DARK)
    ax2.text(5.0, sy, dist, fontsize=8.5, fontfamily="monospace", color=BLUE)
    ax2.text(10.5, sy, source, fontsize=8, color=LIGHT, style="italic")
    sy -= 0.42

# ── Mini distribution plots ──
# WM capacity distribution
ax_wm = fig2.add_axes([0.07, 0.32, 0.25, 0.1])  # [left, bottom, width, height]
x_wm = np.linspace(1, 8, 200)
y_wm = np.exp(-0.5 * ((x_wm - 4.0) / 1.5) ** 2)
y_wm[x_wm < 2] = 0
y_wm[x_wm > 7] = 0
ax_wm.fill_between(x_wm, y_wm, color=PURPLE, alpha=0.3)
ax_wm.plot(x_wm, y_wm, color=PURPLE, lw=1.5)
ax_wm.set_title("WM Capacity", fontsize=8, color=DARK)
ax_wm.set_xlim(1, 8)
ax_wm.set_xticks([2, 3, 4, 5, 6, 7])
ax_wm.tick_params(labelsize=7)
ax_wm.set_yticks([])
ax_wm.spines["top"].set_visible(False)
ax_wm.spines["right"].set_visible(False)
ax_wm.spines["left"].set_visible(False)

# Forgetting rate distribution (log-normal)
ax_fg = fig2.add_axes([0.38, 0.32, 0.25, 0.1])
x_fg = np.linspace(0.1, 4, 200)
y_fg = (1 / (x_fg * 0.35 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (np.log(x_fg) / 0.35) ** 2)
ax_fg.fill_between(x_fg, y_fg, color=RED, alpha=0.3)
ax_fg.plot(x_fg, y_fg, color=RED, lw=1.5)
ax_fg.axvline(1.0, color=MED, lw=0.8, ls="--")
ax_fg.text(1.05, max(y_fg) * 0.7, "1.0x\n(avg)", fontsize=6.5, color=MED)
ax_fg.set_title("Forgetting Rate", fontsize=8, color=DARK)
ax_fg.set_xlim(0, 4)
ax_fg.tick_params(labelsize=7)
ax_fg.set_yticks([])
ax_fg.spines["top"].set_visible(False)
ax_fg.spines["right"].set_visible(False)
ax_fg.spines["left"].set_visible(False)

# Prior knowledge levels (stacked bar)
ax_pk = fig2.add_axes([0.69, 0.32, 0.25, 0.1])
levels = ["Zero", "Exposure", "Fragile", "Solid", "Mastered"]
probs = [0.30, 0.25, 0.20, 0.15, 0.10]
colors_pk = ["#D9534F", "#E8A04C", "#F0C84D", "#7DBA5F", "#4A9D5F"]
bars = ax_pk.barh(range(5), probs, color=colors_pk, edgecolor=WHITE, linewidth=0.5, height=0.7)
ax_pk.set_yticks(range(5))
ax_pk.set_yticklabels(levels, fontsize=7)
ax_pk.set_title("Prior Knowledge (per skill)", fontsize=8, color=DARK)
ax_pk.set_xlim(0, 0.35)
ax_pk.tick_params(labelsize=7)
for bar, p in zip(bars, probs):
    ax_pk.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
               f"{p:.0%}", fontsize=7, va="center", color=DARK)
ax_pk.spines["top"].set_visible(False)
ax_pk.spines["right"].set_visible(False)
ax_pk.invert_yaxis()

# ── Section C: Prior Knowledge Detail Table ──
ax2.text(0.5, 5.3, "C", fontsize=16, fontweight="bold", color=GOLD)
ax2.text(1.1, 5.3, "Initial Knowledge Levels  (what students know before simulation starts)",
         fontsize=13, fontweight="bold", color=BLACK)

pk_header = ["Level", "P(level)", "RS", "SS (days)", "Schema"]
pk_rows = [
    ["Zero",     "30%", "0.05", "0.1",  "0.0"],
    ["Exposure", "25%", "0.20", "1.0",  "0.1"],
    ["Fragile",  "20%", "0.40", "3.0",  "0.5"],
    ["Solid",    "15%", "0.70", "10.0", "0.85"],
    ["Mastered", "10%", "0.90", "30.0", "1.0"],
]

col_x = [0.7, 3.0, 5.0, 6.8, 8.8]
ty = 4.9
for j, h in enumerate(pk_header):
    ax2.text(col_x[j], ty, h, fontsize=9, fontweight="bold", color=MED)
ax2.plot([0.5, 10.5], [ty - 0.12, ty - 0.12], color=VERY_LIGHT, lw=1)

for i, row in enumerate(pk_rows):
    ry = ty - 0.42 - i * 0.38
    bg_alpha = 0.15 if i % 2 == 0 else 0
    if bg_alpha > 0:
        ax2.fill_between([0.5, 10.5], ry - 0.15, ry + 0.2, color=VERY_LIGHT, alpha=0.5)
    for j, val in enumerate(row):
        weight = "bold" if j == 0 else "normal"
        ax2.text(col_x[j], ry, val, fontsize=9, fontweight=weight, color=DARK)

# ── Section D: Key Insight Callout ──
rounded_box(ax2, 0.3, 1.5, 13.3, 1.2, color="#FFF8F0", edge=GOLD, lw=1.5)
ax2.text(7, 2.35, "Key Insight", ha="center", fontsize=11, fontweight="bold", color=GOLD)
ax2.text(7, 2.0, "20% of synthetic students have math anxiety (Ashcraft 2002). Anxiety eats 0.8-1.5 items of WM capacity.",
         ha="center", fontsize=10, color=DARK)
ax2.text(7, 1.7, "A student with WM capacity 4.0 and anxiety effectively has capacity 2.5-3.2 -- pushed into the Low WM danger zone.",
         ha="center", fontsize=9.5, color=MED)

plt.tight_layout(pad=1.0)
fig2.savefig(f"{OUT}/infographic_2_archetypes.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: infographic_2_archetypes.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: HOW SIMULATION WORKS
# ═══════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(figsize=(14, 18), dpi=200)
ax3.set_xlim(0, 14)
ax3.set_ylim(0, 18)
ax3.axis("off")
fig3.patch.set_facecolor(BG)

# Title
ax3.text(7, 17.5, "Figure 3: How the Simulation Works", ha="center", va="center",
         fontsize=20, fontweight="bold", color=BLACK)
ax3.text(7, 17.05, "8 teaching actions with specific cognitive signatures. 5 sessions with forgetting. 200K+ trajectories.",
         ha="center", fontsize=10, color=MED, style="italic")

# ── Section A: 8 Teaching Actions + Their Signatures ──
ax3.text(0.5, 16.5, "A", fontsize=16, fontweight="bold", color=GOLD)
ax3.text(1.1, 16.5, "8 Teaching Actions  (each has a unique effect on the 5 dimensions)",
         fontsize=13, fontweight="bold", color=BLACK)

# Heatmap-style matrix
actions_data = [
    #                        RS     SS    Schema   WM     Affect->Engaged
    ("Space & Test",        [+0.6, +0.45,  0.0,  -0.2,  +0.6]),
    ("Reteach",             [+0.7, +0.15, +0.1,  -0.4,  +0.5]),
    ("Worked Example",      [ 0.0,   0.0, +0.35, -0.5,  +0.4]),
    ("Faded Example",       [+0.3, +0.2,  +0.25, -0.15, +0.7]),
    ("Interleave",          [ 0.0,   0.0,   0.0, +0.3,  +0.3]),
    ("Increase Challenge",  [-0.2, +0.3,    0.0, +0.4,  +0.2]),
    ("Reduce Load",         [+0.3, +0.05,   0.0, -0.6,  +0.6]),
    ("Diagnostic Probe",    [ 0.0,   0.0,    0.0, +0.1,  +0.4]),
]

dim_labels = ["RS", "SS", "Schema", "WM load", "Engage"]
dim_x = [5.5, 7.0, 8.5, 10.0, 11.5]

# Column headers
for j, (label, x) in enumerate(zip(dim_labels, dim_x)):
    ax3.text(x + 0.4, 16.15, label, ha="center", fontsize=8.5, fontweight="bold", color=MED, rotation=0)

# Action rows
for i, (action_name, values) in enumerate(actions_data):
    ry = 15.7 - i * 0.52
    ax3.text(0.7, ry, action_name, fontsize=9, fontweight="bold", color=DARK, va="center")

    for j, (val, x) in enumerate(zip(values, dim_x)):
        # Color code: green=good, red=bad, gray=neutral
        if j == 3:  # WM: negative is good (reducing load)
            if val < -0.1:
                c = GREEN
                symbol = f"{val:+.1f}"
            elif val > 0.1:
                c = RED
                symbol = f"{val:+.1f}"
            else:
                c = LIGHT
                symbol = "0"
        else:  # Other dims: positive is good
            if val > 0.1:
                c = GREEN
                symbol = f"+{val:.2f}"
            elif val < -0.1:
                c = RED
                symbol = f"{val:.1f}"
            else:
                c = LIGHT
                symbol = "--"

        # Cell background
        if val > 0.3 and j != 3:
            bg_c = "#E8F4E8"
        elif val < -0.1 and j != 3:
            bg_c = "#F4E8E8"
        elif j == 3 and val < -0.3:
            bg_c = "#E8F4E8"
        elif j == 3 and val > 0.3:
            bg_c = "#F4E8E8"
        else:
            bg_c = WHITE

        rounded_box(ax3, x, ry - 0.18, 0.85, 0.38, color=bg_c, edge=VERY_LIGHT, lw=0.5)
        ax3.text(x + 0.42, ry, symbol, ha="center", va="center", fontsize=8, fontweight="bold", color=c)

# Legend for the heatmap
ax3.text(5.5, 11.35, "Green = beneficial effect    Red = costly effect    -- = no effect    WM: negative = reducing load (good)",
         fontsize=7.5, color=LIGHT, style="italic")

# Action descriptions (compact)
action_notes = [
    ("Space & Test", "Best for: low RS + solid SS. The desirable difficulty sweet spot."),
    ("Reteach", "Best for: low RS + weak SS. Re-encode, don't test a fragile memory."),
    ("Worked Example", "Best for: no schema + high EI. Main schema-building tool."),
    ("Faded Example", "Best for: partial schema. Balanced -- the all-rounder."),
    ("Interleave", "Best for: confusable skills. Builds discrimination, costs WM."),
    ("Inc. Challenge", "Best for: bored + low WM. Re-engage through difficulty."),
    ("Reduce Load", "Best for: frustrated + high WM. Break the vicious cycle."),
    ("Diagnostic Probe", "Best for: high uncertainty. Pure information, no learning."),
]

ny = 10.9
for name, note in action_notes:
    ax3.text(0.7, ny, name, fontsize=8, fontweight="bold", color=DARK)
    ax3.text(3.6, ny, note, fontsize=8, color=MED)
    ny -= 0.32

# ── Section B: Simulation Protocol ──
ax3.text(0.5, 8.1, "B", fontsize=16, fontweight="bold", color=GOLD)
ax3.text(1.1, 8.1, "Simulation Protocol  (one trajectory = one student's full journey)",
         fontsize=13, fontweight="bold", color=BLACK)

# Timeline diagram
timeline_y = 7.3
session_x = [1.0, 3.5, 6.0, 8.5, 11.0]
forget_x =  [2.25, 4.75, 7.25, 9.75]

for i, sx in enumerate(session_x):
    rounded_box(ax3, sx, timeline_y - 0.25, 1.3, 0.5, color="#E8F0F4", edge=BLUE, lw=1.0)
    ax3.text(sx + 0.65, timeline_y, f"S{i+1}", ha="center", va="center",
             fontsize=10, fontweight="bold", color=BLUE)
    ax3.text(sx + 0.65, timeline_y - 0.18, "20 problems", ha="center", fontsize=7, color=MED)

for i, fx in enumerate(forget_x):
    ax3.text(fx + 0.3, timeline_y, "24h", ha="center", va="center",
             fontsize=8, fontweight="bold", color=RED)
    ax3.text(fx + 0.3, timeline_y - 0.15, "forget", ha="center", fontsize=7, color=RED, style="italic")
    # Arrow
    ax3.annotate("", xy=(fx + 0.6, timeline_y), xytext=(fx, timeline_y),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1))

# Forgetting formula
ax3.text(1.0, 6.55, "Forgetting:  RS(t) = (1 + t / SS)", fontsize=10, color=DARK, fontfamily="serif")
ax3.text(6.9, 6.65, "-forgetting_rate", fontsize=7.5, color=RED, fontfamily="serif")
ax3.text(1.0, 6.2, "High stability (SS) = slower forgetting.  'Forgetful' archetype has 1.8x multiplier.",
         fontsize=9, color=MED)

# Mastery formula
rounded_box(ax3, 0.5, 5.3, 13.0, 0.65, color="#F0F4E8", edge=GREEN, lw=1.0)
ax3.text(7, 5.75, "Final Mastery  =  mean across 11 skills of  ( 0.3 * RS  +  0.4 * SS_normalized  +  0.3 * Schema )",
         ha="center", fontsize=10, fontweight="bold", color=GREEN, fontfamily="serif")
ax3.text(7, 5.45, "Stability (SS) gets the highest weight -- deep encoding matters more than momentary recall",
         ha="center", fontsize=8.5, color=MED, style="italic")

# ── Section C: Scale ──
ax3.text(0.5, 4.7, "C", fontsize=16, fontweight="bold", color=GOLD)
ax3.text(1.1, 4.7, "Simulation Scale", fontsize=13, fontweight="bold", color=BLACK)

scale_data = [
    ("Non-AI policies", "50,000 students", "4 policies", "200,000 trajectories", "~13 min"),
    ("Active Inference", "10,000 students", "1 policy",  "10,000 trajectories",  "~7 hrs (parallel)"),
    ("TOTAL",           "",                "",           "210,000 trajectories", ""),
]

col_sx = [0.7, 4.0, 6.5, 8.3, 11.5]
headers_s = ["Run", "Students", "Policies", "Total Trajectories", "Runtime"]

sy2 = 4.35
for j, h in enumerate(headers_s):
    ax3.text(col_sx[j], sy2, h, fontsize=8.5, fontweight="bold", color=MED)
ax3.plot([0.5, 13.5], [sy2 - 0.1, sy2 - 0.1], color=VERY_LIGHT, lw=1)

for i, row in enumerate(scale_data):
    ry = sy2 - 0.38 - i * 0.35
    weight = "bold" if i == 2 else "normal"
    for j, val in enumerate(row):
        ax3.text(col_sx[j], ry, val, fontsize=9, fontweight=weight, color=DARK if i < 2 else GOLD)

# ── Section D: Key Results from 200K runs ──
ax3.text(0.5, 2.9, "D", fontsize=16, fontweight="bold", color=GOLD)
ax3.text(1.1, 2.9, "Results So Far  (200,000 non-AI trajectories completed)",
         fontsize=13, fontweight="bold", color=BLACK)

results_data = [
    ("Meta-Function",    0.350, 14.2, 85.4, GOLD),
    ("FSRS-Only",        0.339, 54.5, 45.4, BLUE),
    ("Random",           0.319, 33.3, 66.0, MED),
    ("Fixed Curriculum", 0.278, 19.3, 78.2, LIGHT),
]

# Mini bar chart inline
bar_start_x = 5.0
for i, (name, mastery, frust, engage, color) in enumerate(results_data):
    ry = 2.45 - i * 0.42
    ax3.text(0.7, ry, name, fontsize=9, fontweight="bold", color=color)

    # Mastery bar
    bw = mastery * 8
    ax3.barh(ry, bw, height=0.2, left=bar_start_x, color=color, alpha=0.5, zorder=3)
    ax3.text(bar_start_x + bw + 0.1, ry, f"{mastery:.3f}", fontsize=8, color=DARK, va="center")

    # Frustration indicator
    if frust > 30:
        ax3.text(9.5, ry, f"{frust:.0f}% frustrated", fontsize=8, color=RED, fontweight="bold", va="center")
    else:
        ax3.text(9.5, ry, f"{frust:.0f}% frustrated", fontsize=8, color=MED, va="center")

    # Engagement
    ax3.text(12.0, ry, f"{engage:.0f}% engaged", fontsize=8, color=GREEN if engage > 80 else MED, va="center")

# Labels
ax3.text(bar_start_x, 2.75, "Mastery", fontsize=8, fontweight="bold", color=MED)
ax3.text(9.5, 2.75, "Frustrated", fontsize=8, fontweight="bold", color=MED)
ax3.text(12.0, 2.75, "Engaged", fontsize=8, fontweight="bold", color=MED)

# FSRS warning callout
rounded_box(ax3, 0.5, 0.5, 13.0, 0.6, color="#FEF0F0", edge=RED, lw=1.2)
ax3.text(7, 0.88, "FSRS-Only causes 54.5% frustration -- memory-only models ignore affect and WM.",
         ha="center", fontsize=9.5, fontweight="bold", color=RED)
ax3.text(7, 0.6, "It keeps testing a drowning student because their RS looks low. Schema is the most important dimension (ablation: -0.037 mastery when removed).",
         ha="center", fontsize=8.5, color=MED)

plt.tight_layout(pad=1.0)
fig3.savefig(f"{OUT}/infographic_3_simulation.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: infographic_3_simulation.png")

print("\n" + "=" * 60)
print("ALL 3 INFOGRAPHICS GENERATED")
print("=" * 60)
print(f"  {OUT}/infographic_1_anatomy.png")
print(f"  {OUT}/infographic_2_archetypes.png")
print(f"  {OUT}/infographic_3_simulation.png")
