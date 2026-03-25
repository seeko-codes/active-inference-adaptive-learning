"""3 compact infographics for synthetic students. Simple matplotlib only."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colors
GOLD = "#C17817"
BLUE = "#2E6B8A"
RED = "#B5453A"
GREEN = "#3A7D44"
PURPLE = "#6B4C9A"
TEAL = "#2A9D8F"
DARK = "#222222"
MED = "#666666"
LIGHT = "#AAAAAA"
BG = "#FAFAFA"

OUT = "/Users/aatutor/adaptive-learning/poster"


# ═══════════════════════════════════════════
# FIGURE 1: Anatomy of a Synthetic Student
# ═══════════════════════════════════════════

fig1 = plt.figure(figsize=(11, 8), dpi=200, facecolor=BG)

# Title
fig1.suptitle("Anatomy of a Synthetic Student", fontsize=16, fontweight="bold",
              y=0.97, color=DARK)
fig1.text(0.5, 0.94, "16 parameters define a student. 5 hidden dimensions evolve. 4 channels reveal state.",
          ha="center", fontsize=8.5, color=MED, style="italic")

# -- Left panel: 5 Hidden Dimensions --
ax_dims = fig1.add_axes([0.04, 0.28, 0.55, 0.62])
ax_dims.axis("off")
ax_dims.set_xlim(0, 10)
ax_dims.set_ylim(0, 10)

ax_dims.text(0, 9.7, "5 Hidden State Dimensions", fontsize=11, fontweight="bold", color=DARK)
ax_dims.text(4.8, 9.7, "(never directly observed)", fontsize=8, color=LIGHT, style="italic")

dims = [
    ("RS", "Retrievability", "Can they recall it now?", "5 bins [0,1]", BLUE,
     "very_low | low | moderate | high | very_high"),
    ("SS", "Stability", "How deep is the memory?", "4 bins [0.1,inf) days", TEAL,
     "never_learned | fragile | solid | deep"),
    ("Sch", "Schema", "Organized the idea?", "3 bins [0,1]", GREEN,
     "none | partial | full"),
    ("WM", "Working Memory", "Are they overloaded?", "3 bins [0,1]", PURPLE,
     "low (headroom) | moderate | high (near capacity)"),
    ("Aff", "Affect", "Frustrated or bored?", "3 bins", RED,
     "frustrated | engaged | bored"),
]

for i, (abbr, name, q, bins, color, levels) in enumerate(dims):
    y = 8.8 - i * 1.7
    ax_dims.fill_between([0, 0.15], y - 0.4, y + 0.5, color=color, alpha=0.7)
    ax_dims.text(0.4, y + 0.25, abbr, fontsize=11, fontweight="bold", color=color)
    ax_dims.text(1.3, y + 0.25, f"{name} -- {q}", fontsize=9, fontweight="bold", color=DARK)
    ax_dims.text(1.3, y - 0.05, bins, fontsize=7.5, color=MED)
    ax_dims.text(1.3, y - 0.35, levels, fontsize=7, color=LIGHT, fontfamily="monospace")

# Joint state callout
ax_dims.text(0.2, 0.5, "Joint: 5x4x3x3x3 = 540 states/skill x 11 skills = 5,940 total",
             fontsize=8.5, fontweight="bold", color=GOLD,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8F0", edgecolor=GOLD, lw=1))

# -- Right panel: 4 Observation Channels + 16 Parameters --
ax_obs = fig1.add_axes([0.62, 0.52, 0.35, 0.38])
ax_obs.axis("off")
ax_obs.set_xlim(0, 10)
ax_obs.set_ylim(0, 10)

ax_obs.text(0, 9.5, "4 Observation Channels", fontsize=10, fontweight="bold", color=DARK)

obs = [
    ("Accuracy", "correct/incorrect", "RS, Schema, WM, Affect"),
    ("Response Time", "fast/normal/slow", "WM, Affect"),
    ("Explanation", "low/med/high", "Schema"),
    ("Confidence", "1-5 scale", "RS, Schema"),
]

for i, (name, vals, drivers) in enumerate(obs):
    y = 8.2 - i * 2.0
    ax_obs.text(0.3, y, name, fontsize=9, fontweight="bold", color=DARK)
    ax_obs.text(0.3, y - 0.7, vals, fontsize=7.5, color=MED)
    ax_obs.text(0.3, y - 1.3, f"driven by: {drivers}", fontsize=7, color=LIGHT, style="italic")

# -- Bottom: 16 Parameters compact --
ax_params = fig1.add_axes([0.62, 0.28, 0.35, 0.22])
ax_params.axis("off")
ax_params.set_xlim(0, 10)
ax_params.set_ylim(0, 10)

ax_params.text(0, 9.5, "16-Parameter Profile", fontsize=10, fontweight="bold", color=DARK)

params = [
    (BLUE,   "Prior Knowledge", "initial_rs, initial_ss, initial_schema"),
    (PURPLE, "Working Memory",  "wm_capacity, wm_recovery_rate"),
    (GREEN,  "Learning",        "schema_rate, ss_growth, rs_recovery"),
    (TEAL,   "Retention",       "forgetting_rate"),
    (RED,    "Affect",          "frust_thresh, bore_thresh, inertia, baseline"),
    (MED,    "Response",        "base_rt, rt_variance, explanation_quality"),
    (GOLD,   "Metacognition",   "metacognitive_bias"),
]

for i, (color, group, p) in enumerate(params):
    y = 8.0 - i * 1.2
    ax_params.plot([0, 0.15], [y, y], color=color, lw=3, solid_capstyle="round")
    ax_params.text(0.4, y + 0.1, group, fontsize=7.5, fontweight="bold", color=color)
    ax_params.text(0.4, y - 0.5, p, fontsize=6.5, color=MED, fontfamily="monospace")

# -- Bottom strip: Example --
ax_ex = fig1.add_axes([0.04, 0.03, 0.93, 0.2])
ax_ex.axis("off")
ax_ex.set_xlim(0, 10)
ax_ex.set_ylim(0, 10)

ax_ex.text(0, 9, "Example: How hidden state produces observations", fontsize=10, fontweight="bold", color=DARK)
ex = (
    "State: RS=high(0.75)  Schema=full  WM=low  Affect=engaged  EI=medium\n\n"
    "  P(correct)          ~ 0.90    (high RS + full schema + low load)\n"
    "  P(fast response)    ~ 0.50    (low WM = not struggling)\n"
    "  P(high explanation) ~ 0.75    (full schema = can articulate)\n"
    "  P(high confidence)  ~ 0.70    (high RS + full schema = calibrated)"
)
ax_ex.text(0.3, 7.5, ex, fontsize=7.5, fontfamily="monospace", color=DARK,
           va="top", linespacing=1.4)

fig1.savefig(f"{OUT}/overview_1_anatomy.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: overview_1_anatomy.png")


# ═══════════════════════════════════════════
# FIGURE 2: Who Are the Students
# ═══════════════════════════════════════════

fig2 = plt.figure(figsize=(11, 8), dpi=200, facecolor=BG)
fig2.suptitle("Who Are the Synthetic Students?", fontsize=16, fontweight="bold", y=0.98, color=DARK)

# -- Top-left: Archetypes table --
ax_arch = fig2.add_axes([0.03, 0.48, 0.47, 0.45])
ax_arch.axis("off")
ax_arch.set_title("8 Archetypes", fontsize=11, fontweight="bold", color=DARK, loc="left")

names =     ["Novice","Fast Lrn","Partial K","Forgetful","Low WM","Anxious","Overconf.","Advanced"]
wm_caps =   [4.0, 6.0, 5.0, 5.0, 3.0, 4.5, 5.0, 6.0]
schema_r =  [0.15, 0.50, 0.30, 0.30, 0.20, 0.25, 0.25, 0.50]
forget_r =  [1.3, 0.7, 1.0, 1.8, 1.1, 1.2, 1.0, 0.6]
frust_t =   [0.60, 0.80, 0.70, 0.70, 0.55, 0.50, 0.75, 0.85]

cell_text = []
for i in range(8):
    cell_text.append([names[i], f"{wm_caps[i]}", f"{schema_r[i]}", f"{forget_r[i]}x", f"{frust_t[i]}"])

table = ax_arch.table(cellText=cell_text,
                 colLabels=["Type", "WM", "Schema\nRate", "Forget\nRate", "Frust.\nThresh"],
                 cellLoc="center", loc="center",
                 colWidths=[0.28, 0.14, 0.16, 0.16, 0.16])
table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1, 1.3)

for j in range(5):
    table[0, j].set_facecolor("#E8E8E8")
    table[0, j].set_text_props(fontweight="bold", fontsize=7)

for i in range(8):
    if forget_r[i] > 1.0:
        table[i + 1, 3].set_text_props(color=RED, fontweight="bold")
    if frust_t[i] < 0.6:
        table[i + 1, 4].set_text_props(color=RED, fontweight="bold")
    if wm_caps[i] <= 3.0:
        table[i + 1, 1].set_text_props(color=PURPLE, fontweight="bold")

# -- Top-right: Population sampling --
ax_samp = fig2.add_axes([0.53, 0.48, 0.45, 0.45])
ax_samp.axis("off")
ax_samp.set_title("Population Sampling", fontsize=11, fontweight="bold", color=DARK, loc="left")

sampling = [
    ("WM capacity",       "N(4.0, 1.5) [2,7]",     "Cowan 2001"),
    ("Math anxiety",      "Bernoulli(0.20)",         "Ashcraft 2002"),
    ("  anxiety WM loss", "U(0.8, 1.5) items",       "Attn. control"),
    ("Schema rate",       "N(0.3+0.1·z_wm, 0.1)",   "Corr. w/ WM"),
    ("Forget rate",       "LogNormal(0, 0.35)",      "FSRS data"),
    ("Frust. threshold",  "N(0.7-0.2·anx, 0.1)",    "Inv. anxiety"),
    ("Metacog. bias",     "N(0, 0.2)",               "+ = overconf."),
]

cell_samp = [[n, d, s] for n, d, s in sampling]
table2 = ax_samp.table(cellText=cell_samp,
                  colLabels=["Parameter", "Distribution", "Source"],
                  cellLoc="left", loc="center",
                  colWidths=[0.32, 0.38, 0.25])
table2.auto_set_font_size(False)
table2.set_fontsize(7)
table2.scale(1, 1.35)
for j in range(3):
    table2[0, j].set_facecolor("#E8E8E8")
    table2[0, j].set_text_props(fontweight="bold")

# -- Bottom-left: WM Capacity distribution --
ax_wm = fig2.add_axes([0.06, 0.06, 0.26, 0.32])
x = np.linspace(0.5, 8.5, 300)
y_wm = np.exp(-0.5 * ((x - 4.0) / 1.5) ** 2)
y_wm[x < 2] = 0
y_wm[x > 7] = 0
ax_wm.fill_between(x, y_wm, color=PURPLE, alpha=0.25)
ax_wm.plot(x, y_wm, color=PURPLE, lw=2)
ax_wm.axvline(4.0, color=PURPLE, lw=0.8, ls="--", alpha=0.5)
ax_wm.text(4.2, max(y_wm) * 0.85, "mean=4.0", fontsize=7, color=PURPLE)
ax_wm.set_xlabel("WM Capacity (items)", fontsize=8)
ax_wm.set_xlim(0.5, 8.5)
ax_wm.set_yticks([])
ax_wm.set_title("WM Capacity\nNormal(4.0, 1.5)", fontsize=9, fontweight="bold", color=PURPLE)
ax_wm.tick_params(labelsize=7)
for sp in ["top", "right", "left"]:
    ax_wm.spines[sp].set_visible(False)

# -- Bottom-center: Forgetting Rate distribution --
ax_fg = fig2.add_axes([0.38, 0.06, 0.26, 0.32])
x2 = np.linspace(0.1, 4, 300)
y_fg = (1 / (x2 * 0.35 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (np.log(x2) / 0.35) ** 2)
ax_fg.fill_between(x2, y_fg, color=RED, alpha=0.2)
ax_fg.plot(x2, y_fg, color=RED, lw=2)
ax_fg.axvline(1.0, color=RED, lw=0.8, ls="--", alpha=0.5)
ax_fg.axvline(1.8, color=RED, lw=0.8, ls=":", alpha=0.7)
ax_fg.text(1.05, max(y_fg) * 0.85, "1.0x avg", fontsize=7, color=MED)
ax_fg.text(1.85, max(y_fg) * 0.6, "1.8x\nforgetful", fontsize=6.5, color=RED)
ax_fg.set_xlabel("Forgetting Rate Multiplier", fontsize=8)
ax_fg.set_xlim(0, 4)
ax_fg.set_yticks([])
ax_fg.set_title("Forgetting Rate\nLogNormal(0, 0.35)", fontsize=9, fontweight="bold", color=RED)
ax_fg.tick_params(labelsize=7)
for sp in ["top", "right", "left"]:
    ax_fg.spines[sp].set_visible(False)

# -- Bottom-right: Prior knowledge bar chart --
ax_pk = fig2.add_axes([0.72, 0.06, 0.26, 0.32])
levels = ["Zero\n30%", "Expos.\n25%", "Fragile\n20%", "Solid\n15%", "Master\n10%"]
rs_vals = [0.05, 0.20, 0.40, 0.70, 0.90]
ss_vals = [0.1, 1.0, 3.0, 10.0, 30.0]
sch_vals = [0.0, 0.1, 0.5, 0.85, 1.0]

x_pos = np.arange(5)
w = 0.25
ax_pk.bar(x_pos - w, rs_vals, w, label="RS", color=BLUE, alpha=0.7)
ax_pk.bar(x_pos, [s / 30 for s in ss_vals], w, label="SS (norm.)", color=TEAL, alpha=0.7)
ax_pk.bar(x_pos + w, sch_vals, w, label="Schema", color=GREEN, alpha=0.7)

ax_pk.set_xticks(x_pos)
ax_pk.set_xticklabels(levels, fontsize=6.5)
ax_pk.set_ylabel("Value (normalized)", fontsize=7)
ax_pk.set_title("Initial Knowledge\nLevels (per skill)", fontsize=9, fontweight="bold", color=DARK)
ax_pk.legend(fontsize=6.5, loc="upper left")
ax_pk.tick_params(labelsize=7)
for sp in ["top", "right"]:
    ax_pk.spines[sp].set_visible(False)

fig2.savefig(f"{OUT}/overview_2_students.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: overview_2_students.png")


# ═══════════════════════════════════════════
# FIGURE 3: How the Simulation Works
# ═══════════════════════════════════════════

fig3 = plt.figure(figsize=(11, 8), dpi=200, facecolor=BG)
fig3.suptitle("How the Simulation Works", fontsize=16, fontweight="bold", y=0.98, color=DARK)
fig3.text(0.5, 0.95, "8 actions with specific effects. 5 sessions with forgetting. 200K+ trajectories.",
          ha="center", fontsize=8.5, color=MED, style="italic")

# -- Top: Action effect heatmap --
ax_heat = fig3.add_axes([0.06, 0.45, 0.58, 0.47])

actions = ["Space&Test", "Reteach", "Worked Ex.", "Faded Ex.",
           "Interleave", "Inc. Challenge", "Reduce Load", "Diag. Probe"]
dims = ["RS", "SS", "Schema", "WM\n(neg=good)", "Engage"]

data = np.array([
    [+0.6, +0.45,  0.0,  -0.2,  +0.6],
    [+0.7, +0.15, +0.1,  -0.4,  +0.5],
    [ 0.0,   0.0, +0.35, -0.5,  +0.4],
    [+0.3, +0.2,  +0.25, -0.15, +0.7],
    [ 0.0,   0.0,   0.0, +0.3,  +0.3],
    [-0.2, +0.3,    0.0, +0.4,  +0.2],
    [+0.3, +0.05,   0.0, -0.6,  +0.6],
    [ 0.0,   0.0,    0.0, +0.1,  +0.4],
])

# Custom colormap: red for bad, white for neutral, green for good
# For WM column, flip sign for display (negative WM = good)
display_data = data.copy()
display_data[:, 3] = -display_data[:, 3]  # flip WM so positive = good

im = ax_heat.imshow(display_data, cmap="RdYlGn", aspect="auto", vmin=-0.6, vmax=0.7)

# Annotate cells with original values
for i in range(8):
    for j in range(5):
        val = data[i, j]
        if abs(val) < 0.01:
            txt = "--"
            color = LIGHT
        else:
            txt = f"{val:+.2f}"
            color = "white" if abs(display_data[i, j]) > 0.35 else DARK
        ax_heat.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                     fontweight="bold", color=color)

ax_heat.set_xticks(range(5))
ax_heat.set_xticklabels(dims, fontsize=8, fontweight="bold")
ax_heat.set_yticks(range(8))
ax_heat.set_yticklabels(actions, fontsize=8)
ax_heat.set_title("Teaching Action Effects on Hidden State", fontsize=10,
                   fontweight="bold", color=DARK, pad=8)
ax_heat.tick_params(length=0)

# -- Right side: Action descriptions --
ax_desc = fig3.add_axes([0.66, 0.45, 0.32, 0.47])
ax_desc.axis("off")
ax_desc.set_xlim(0, 10)
ax_desc.set_ylim(0, 10)

ax_desc.text(0, 9.7, "When to Use Each", fontsize=10, fontweight="bold", color=DARK)

notes = [
    ("Space & Test",    "low RS + solid SS"),
    ("Reteach",         "low RS + weak SS"),
    ("Worked Example",  "no schema + high EI"),
    ("Faded Example",   "partial schema"),
    ("Interleave",      "confusable skills"),
    ("Inc. Challenge",  "bored + low WM"),
    ("Reduce Load",     "frustrated + high WM"),
    ("Diag. Probe",     "high uncertainty"),
]

for i, (name, when) in enumerate(notes):
    y = 8.8 - i * 1.15
    ax_desc.text(0.2, y, name, fontsize=8, fontweight="bold", color=DARK)
    ax_desc.text(0.2, y - 0.5, when, fontsize=7.5, color=MED)

# -- Bottom-left: Simulation protocol --
ax_proto = fig3.add_axes([0.06, 0.06, 0.55, 0.33])
ax_proto.axis("off")
ax_proto.set_xlim(0, 12)
ax_proto.set_ylim(0, 10)

ax_proto.text(0, 9.5, "Simulation Protocol", fontsize=10, fontweight="bold", color=DARK)

# Timeline
for i in range(5):
    x = 0.5 + i * 2.3
    ax_proto.add_patch(plt.Rectangle((x, 7.2), 1.2, 0.8, facecolor="#E8F0F4",
                                      edgecolor=BLUE, lw=1, zorder=2))
    ax_proto.text(x + 0.6, 7.6, f"S{i+1}", ha="center", fontsize=9, fontweight="bold", color=BLUE)
    ax_proto.text(x + 0.6, 7.3, "20 prob", ha="center", fontsize=6.5, color=MED)
    if i < 4:
        ax_proto.annotate("", xy=(x + 1.8, 7.6), xytext=(x + 1.3, 7.6),
                         arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
        ax_proto.text(x + 1.55, 7.95, "24h", ha="center", fontsize=7, fontweight="bold", color=RED)

# Formulas
ax_proto.text(0.2, 5.8, "Forgetting:", fontsize=9, fontweight="bold", color=DARK)
ax_proto.text(2.5, 5.8, "RS(t) = (1 + t/SS)^(-forget_rate)", fontsize=9, color=RED, fontfamily="monospace")

ax_proto.text(0.2, 4.6, "Mastery:", fontsize=9, fontweight="bold", color=DARK)
ax_proto.text(2.5, 4.6, "mean(0.3*RS + 0.4*SS_norm + 0.3*Schema)", fontsize=9, color=GREEN, fontfamily="monospace")
ax_proto.text(2.5, 3.8, "across 11 skills. SS weighted highest.", fontsize=8, color=MED, style="italic")

ax_proto.text(0.2, 2.5, "Scale:", fontsize=9, fontweight="bold", color=DARK)
ax_proto.text(2.5, 2.5, "50K students x 4 policies = 200K trajectories (~13 min)", fontsize=8, color=MED)
ax_proto.text(2.5, 1.8, "10K students x active inference = 10K trajectories (~7 hrs)", fontsize=8, color=MED)
ax_proto.text(2.5, 1.1, "Total: 210,000 trajectories", fontsize=8.5, fontweight="bold", color=GOLD)

# -- Bottom-right: Results bar chart --
ax_res = fig3.add_axes([0.66, 0.06, 0.31, 0.33])

policies = ["Meta-Func.", "FSRS-Only", "Random", "Fixed Curr."]
masteries = [0.350, 0.339, 0.319, 0.278]
frust = [14.2, 54.5, 33.3, 19.3]
colors_r = [GOLD, BLUE, MED, LIGHT]

bars = ax_res.barh(range(4), masteries, color=colors_r, edgecolor=DARK, lw=0.5, height=0.6)
ax_res.set_yticks(range(4))
ax_res.set_yticklabels(policies, fontsize=8)
ax_res.set_xlabel("Final Mastery", fontsize=8)
ax_res.set_title("Results (200K runs)", fontsize=10, fontweight="bold", color=DARK)
ax_res.set_xlim(0.2, 0.4)
ax_res.tick_params(labelsize=7)
ax_res.invert_yaxis()

for i, (bar, m, f) in enumerate(zip(bars, masteries, frust)):
    ax_res.text(m + 0.002, i, f"{m:.3f}", va="center", fontsize=7.5, fontweight="bold", color=DARK)
    fcolor = RED if f > 30 else MED
    ax_res.text(0.385, i, f"{f:.0f}% frust.", va="center", fontsize=7, color=fcolor, ha="right")

for sp in ["top", "right"]:
    ax_res.spines[sp].set_visible(False)

fig3.savefig(f"{OUT}/overview_3_simulation.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: overview_3_simulation.png")

print("\nDone. Files:")
print(f"  {OUT}/overview_1_anatomy.png")
print(f"  {OUT}/overview_2_students.png")
print(f"  {OUT}/overview_3_simulation.png")
