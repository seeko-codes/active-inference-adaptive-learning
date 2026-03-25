"""3 simple text-based summary cards."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DARK = "#222222"
MED = "#666666"
LIGHT = "#999999"
GOLD = "#C17817"
BG = "#FAFAFA"
OUT = "/Users/aatutor/adaptive-learning/poster"


def text_card(filename, title, body):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200, facecolor=BG)
    ax.axis("off")
    ax.text(0.5, 0.97, title, transform=ax.transAxes, ha="center", va="top",
            fontsize=15, fontweight="bold", color=DARK, fontfamily="serif")
    ax.text(0.05, 0.88, body, transform=ax.transAxes, va="top",
            fontsize=8.5, color=DARK, fontfamily="monospace", linespacing=1.5)
    fig.savefig(f"{OUT}/{filename}", dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {filename}")


# ── Card 1 ──
text_card("simple_1_anatomy.png",
          "1 — Anatomy of a Synthetic Student",
"""
 5 HIDDEN DIMENSIONS (what's inside their head)
 ───────────────────────────────────────────────
  RS    Retrievability    Can they recall it now?          5 bins [0-1]
  SS    Stability         How deep is the memory?          4 bins [0.1-inf days]
  Sch   Schema            Have they organized the idea?    3 bins [0-1]
  WM    Working Memory    Are they overloaded?             3 bins [0-1]
  Aff   Affect            Frustrated / engaged / bored?    3 bins

  Joint state space: 5x4x3x3x3 = 540 states/skill x 11 skills = 5,940 total


 4 OBSERVATION CHANNELS (what you actually see)
 ───────────────────────────────────────────────
  Accuracy            correct/incorrect         driven by RS, Schema, WM, Affect
  Response Time       fast/normal/slow          driven by WM, Affect
  Explanation         low/med/high              driven by Schema
  Confidence          1-5 scale                 driven by RS, Schema


 16-PARAMETER COGNITIVE PROFILE (fully defines one student)
 ───────────────────────────────────────────────
  Prior Knowledge     initial_rs, initial_ss, initial_schema     (per skill)
  Working Memory      wm_capacity (2-7 items), wm_recovery_rate
  Learning            schema_formation_rate, ss_growth_rate, rs_recovery_rate
  Retention           forgetting_rate                            (power law)
  Affect              frustration_thresh, boredom_thresh, inertia, baseline
  Response            base_response_time, rt_variance, explanation_quality
  Metacognition       metacognitive_bias                         (+ = overconfident)
""")


# ── Card 2 ──
text_card("simple_2_students.png",
          "2 — Who Are the Synthetic Students?",
"""
 8 ARCHETYPES (each stress-tests a different failure mode)
 ─────────────────────────────────────────────────────────────────
  Type              WM Cap   Schema Rate   Forget Rate   Frust Thresh
  Novice             4.0       0.15          1.3x          0.60
  Fast Learner       6.0       0.50          0.7x          0.80
  Partial Knowledge  5.0       0.30          1.0x          0.70
  Forgetful          5.0       0.30          1.8x  !!      0.70
  Low WM             3.0 !!    0.20          1.1x          0.55
  Anxious            4.5       0.25          1.2x          0.50  !!
  Overconfident      5.0       0.25          1.0x          0.75
  Advanced           6.0       0.50          0.6x          0.85


 POPULATION SAMPLING (continuous random students from real research)
 ─────────────────────────────────────────────────────────────────
  WM capacity           Normal(4.0, 1.5), clipped [2, 7]      Cowan 2001
  Math anxiety          20% chance (Bernoulli)                 Ashcraft 2002
    -> WM loss          0.8-1.5 items eaten by anxiety
  Schema formation      Normal(0.3 + 0.1*z_wm, 0.1)           correlated w/ WM
  Forgetting rate       LogNormal(0, 0.35)                     FSRS data
  Frustration thresh    Normal(0.7 - 0.2*anxiety, 0.1)         inversely w/ anxiety


 PRIOR KNOWLEDGE PER SKILL (most students start knowing very little)
 ─────────────────────────────────────────────────────────────────
  Zero (30%)       RS=0.05   SS=0.1 days    Schema=0.0
  Exposure (25%)   RS=0.20   SS=1.0 days    Schema=0.1
  Fragile (20%)    RS=0.40   SS=3.0 days    Schema=0.5
  Solid (15%)      RS=0.70   SS=10.0 days   Schema=0.85
  Mastered (10%)   RS=0.90   SS=30.0 days   Schema=1.0
""")


# ── Card 3 ──
text_card("simple_3_simulation.png",
          "3 — How the Simulation Works",
"""
 8 TEACHING ACTIONS (effect on each dimension: + good, - bad, 0 none)
 ─────────────────────────────────────────────────────────────────
  Action              RS      SS     Schema   WM load   Engagement
  Space & Test       +0.60   +0.45    --      -0.20     +0.60    low RS + solid SS
  Reteach            +0.70   +0.15   +0.10    -0.40     +0.50    low RS + weak SS
  Worked Example      --      --     +0.35    -0.50     +0.40    no schema + high EI
  Faded Example      +0.30   +0.20   +0.25    -0.15     +0.70    partial schema
  Interleave          --      --      --      +0.30     +0.30    confusable skills
  Inc. Challenge     -0.20   +0.30    --      +0.40     +0.20    bored + low WM
  Reduce Load        +0.30   +0.05    --      -0.60     +0.60    frustrated + high WM
  Diagnostic Probe    --      --      --      +0.10     +0.40    high uncertainty

  WM load: negative = reducing load (good)


 SIMULATION PROTOCOL
 ─────────────────────────────────────────────────────────────────
  [S1: 20 problems] --24h forget--> [S2] --24h--> [S3] --24h--> [S4] --24h--> [S5]

  Forgetting:   RS(t) = (1 + t/SS) ^ (-forgetting_rate)
  Mastery:      mean across 11 skills of (0.3*RS + 0.4*SS_norm + 0.3*Schema)
  Scale:        50K students x 4 policies + 10K x active inference = 210K trajectories


 RESULTS (200K non-AI computed + active inference projected)
 ─────────────────────────────────────────────────────────────────
  Policy              Mastery    Frustrated   Engaged    Notes
  Active Inference    ~0.360*     ~8-12%*      ~88%*     * projected, not yet run
  Meta-Function        0.350      14%          85%       best computed so far
  FSRS-Only            0.339      55%          45%       dangerous frustration
  Random               0.319      33%          66%       baseline
  Fixed Curriculum     0.278      19%          78%       no adaptation

 WHY ACTIVE INFERENCE SHOULD EDGE AHEAD (~3-6% over meta-function)
 ─────────────────────────────────────────────────────────────────
  Advantage:  EFE balances exploration vs exploitation automatically
              Uses diagnostic_probe when uncertain (meta-function never probes)
              Updates its own A/B matrices online (learns as it goes)
              C vectors penalize bad observations -> prevents frustration early

  Limitation: Must INFER state from noisy observations (meta-function has oracle)
              Needs ~15-20 problems to narrow beliefs (slow start)
              Myopic planning horizon (policy_len=1, only 1 step ahead)

 WHERE AI WINS MOST (projected learning landscape)
 ─────────────────────────────────────────────────────────────────
  Anxious students       largest advantage   detects frustration precursors early
  Low WM students        large advantage     doesn't over-trigger reduce_load
  Forgetful students     large advantage     better reteach vs space_and_test calls
  Novice students        moderate            efficient probe -> teach sequencing
  Advanced students      near zero           ceiling effect, probing is wasted

 TRAJECTORY SHAPE (projected)
 ─────────────────────────────────────────────────────────────────
  Session 1:   slightly below meta-function  (spending time on diagnostic probes)
  Session 2-3: crosses above                 (beliefs narrow, EFE kicks in)
  Session 4-5: pulls ahead                   (cumulative benefit of precise targeting)
  Forgetting:  shallower dips                (better pre-forgetting encoding)

  Key finding: Schema is the most important dimension (ablation: -0.037 when removed)
  Warning: FSRS-Only causes 55% frustration -- memory-only models ignore affect & WM
""")

print("\nDone.")
