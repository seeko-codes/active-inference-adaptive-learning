# Synthetic Student Simulation: Architecture, Data, and Preliminary Results

## 1. The Validation Problem

Adaptive learning systems require thousands of real students to validate. We bypass this constraint through **normative simulation**: synthetic students with known ground-truth cognitive states, governed by empirically grounded dynamics, whose learning trajectories can be compared across instructional policies.

The core question: *Given a student's current cognitive state, which pedagogical action maximizes long-term learning?*

The meta-question: *How do we know the answer without deploying to real classrooms first?*

---

## 2. System Architecture

```
                    ┌──────────────────────────────────────┐
                    │          Knowledge Space              │
                    │   2,519 algebraic forms across        │
                    │   11 skills, 8 complexity tiers       │
                    └──────────────────┬───────────────────┘
                                       │
         ┌─────────────────────────────┼────────────────────────────┐
         │    GENERATIVE MODEL         ▼                            │
         │    ┌──────────────────────────────────────┐              │
         │    │     Transition Model B(a)             │              │
         │    │     P(s_{t+1} | s_t, action)          │              │
         │    └──────────────────┬───────────────────┘              │
         │                       ▼                                  │
         │    ┌──────────────────────────────────────┐              │
         │    │        Hidden Student State           │              │
         │    │                                       │              │
         │    │   ┌─────────────────────────────────┐ │    ┌───────┐│
         │    │   │ RS  │ SS  │ Schema │ WM │ Affect│ │◄───│Belief ││
         │    │   └─────────────────────────────────┘ │    │Update ││
         │    │   5 factors x 11 skills = 5,940       │    │(Bayes)││
         │    │   states per student                  │    └───┬───┘│
         │    └──────────────────┬───────────────────┘        │    │
         │                       ▼                            │    │
         │    ┌──────────────────────────────────────┐  ┌─────┴───┐│
         │    │      Observation Model A              │  │ Policy  ││
         │    │      P(o | s)                         │─►│Selection││
         │    └──────────────────┬───────────────────┘  │ min G(π)││
         │                       │                      └─────┬───┘│
         └───────────────────────┼────────────────────────────┼────┘
                                 ▼                            │
              ┌──────────────────────────────────────┐        │
              │           Observations                │        │
              │  · Accuracy (correct / incorrect)     │        │
              │  · Response time (ms, log-normal)     │        │
              │  · Explanation quality (0-1)           │        │
              │  · Self-reported confidence (1-5)      │        │
              │  · Confusion target (which skill)      │        │
              └──────────────────────────────────────┘        │
                                 ▲                            ▼
                                 │         ┌─────────────────────────────┐
                                 │         │     8 Teaching Actions       │
                                 │         │  Space & Test  · Reteach     │
                                 │         │  Worked Example · Faded Ex.  │
                                 │         │  Interleave · Inc. Challenge │
                                 │         │  Reduce Load · Diag. Probe   │
                                 │         └─────────────┬───────────────┘
                                 │                       │
                                 │  responds   presents  │
                                 └───── Student ◄────────┘
                                     (simulated)
```

### 2.1 Hidden State Space (5 Factors)

Each student's cognitive state is modeled as a 5-dimensional vector, tracked independently per skill:

| Factor | Notation | Bins | Range | Governs |
|--------|----------|------|-------|---------|
| **Retrievability** | RS | 5 | [0, 1] | Can the student recall this now? |
| **Stability** | SS | 4 | [0.1, +inf) days | How resistant is the memory to decay? |
| **Schema** | -- | 3 | [0, 1] continuous | Has the student organized the concept structurally? |
| **Working Memory** | WM | 3 | [0, 1] utilization | How much cognitive capacity is currently consumed? |
| **Affect** | -- | 3 | {frustrated, engaged, bored} | What is the student's emotional state? |

**Joint state space**: 5 x 4 x 3 x 3 x 3 = **540 states per skill**, with mean-field factorization across 11 skills yielding **5,940 total states** per student.

### 2.2 Observation Model

The system never observes hidden states directly. It infers them from four observable channels:

| Observable | Type | Informative About |
|---|---|---|
| Accuracy | Binary | RS, Schema, WM, Affect |
| Response time | Continuous (ms) | WM, Affect, EI |
| Explanation quality | Continuous [0, 1] | Schema depth |
| Confidence self-report | Ordinal 1-5 | Metacognitive calibration |

### 2.3 Policy Selection

The system selects from **8 pedagogical actions** using a priority-ordered meta-function:

| Priority | Condition | Action | Rationale |
|---|---|---|---|
| 1 | Frustrated + high WM | Reduce load | Break overload-frustration vicious cycle |
| 2 | Bored + low WM | Increase challenge | Re-engage through difficulty |
| 3 | High WM + low accuracy | Reduce load | Extraneous cognitive load |
| 4 | Low discriminability + adequate schemas | Interleave | Build skill discrimination |
| 5 | No schema + high EI | Worked example | Scaffolded schema building |
| 5 | Partial schema | Faded example | Transition toward independence |
| 6 | Low RS + solid SS | Space and test | Desirable difficulty zone |
| 6 | Low RS + weak SS | Reteach | Re-encoding, not testing |

An active inference agent (pymdp, EFE minimization) provides an alternative policy that automatically balances exploration (epistemic value: resolving state uncertainty) and exploitation (pragmatic value: driving learning).

---

## 3. Synthetic Student Archetypes

Eight archetypal learner profiles span the parameter space to cover the major failure modes the system must handle. Each is defined by a ground-truth cognitive parameter vector (`CognitiveParams`) with 16 dimensions.

### 3.1 Archetype Definitions

| Archetype | Description | WM Capacity | Schema Formation Rate | Forgetting Rate | Frustration Threshold | Prevalence Weight |
|---|---|---|---|---|---|---|
| **Novice** | No prior knowledge, slow learning | 4.0 | 0.15 | 1.3x | 0.60 | 1.5 |
| **Fast Learner** | No prior knowledge, high capacity | 6.0 | 0.50 | 0.7x | 0.80 | 0.8 |
| **Partial Knowledge** | Strong on basics, gaps in distribution/factoring | 5.0 | 0.30 | 1.0x | 0.70 | 1.5 |
| **Forgetful** | Learns in-session, rapid between-session decay | 5.0 | 0.30 | 1.8x | 0.70 | 1.0 |
| **Low WM** | Understands concepts, overloads on complex problems | 3.0 | 0.20 | 1.1x | 0.55 | 1.0 |
| **Anxious** | Adequate knowledge, anxiety consumes WM | 4.5 | 0.25 | 1.2x | 0.50 | 1.0 |
| **Overconfident** | Moderate knowledge, resists scaffolding | 5.0 | 0.25 | 1.0x | 0.75 | 0.8 |
| **Advanced** | Strong prior knowledge, needs maintenance | 6.0 | 0.50 | 0.6x | 0.85 | 0.5 |

### 3.2 Cognitive Parameter Vector

Each archetype is fully specified by 16 parameters:

```
CognitiveParams:
  Prior knowledge (per-skill):    initial_rs, initial_ss, initial_schema
  Working memory:                 wm_capacity, wm_recovery_rate
  Learning dynamics:              schema_formation_rate, ss_growth_rate, rs_recovery_rate
  Retention:                      forgetting_rate
  Affect dynamics:                frustration_threshold, boredom_threshold,
                                  affect_inertia, engagement_baseline
  Response characteristics:       base_response_time, response_time_variance,
                                  explanation_quality
  Metacognition:                  metacognitive_bias
```

### 3.3 Continuous Population Sampling

Beyond fixed archetypes, the system supports **continuous random sampling** from empirically grounded distributions for population-level Monte Carlo:

| Parameter | Distribution | Source |
|---|---|---|
| WM capacity | N(4.0, 1.5), clipped [2, 7] | Cowan (2001) |
| Math anxiety prevalence | Bernoulli(0.20) | Ashcraft (2002) |
| Anxiety WM reduction | U(0.8, 1.5) items | Attentional control theory |
| Schema formation rate | N(0.3 + 0.1z_wm, 0.1) | Correlated with WM, r ~ 0.5 |
| Forgetting rate | LogNormal(0, 0.35) | FSRS empirical data |
| Frustration threshold | N(0.7 - 0.2 * anxiety, 0.1) | Inversely related to anxiety |
| Metacognitive bias | N(0, 0.2) | Positive = overconfident |
| Prior knowledge per skill | Categorical(5 levels) | Skewed toward lower knowledge |

**Knowledge level priors** (per-skill):

| Level | P(level) | RS | SS (days) | Schema |
|---|---|---|---|---|
| Zero | 0.30 | 0.05 | 0.1 | 0.0 |
| Exposure | 0.25 | 0.20 | 1.0 | 0.1 |
| Fragile | 0.20 | 0.40 | 3.0 | 0.5 |
| Solid | 0.15 | 0.70 | 10.0 | 0.85 |
| Mastered | 0.10 | 0.90 | 30.0 | 1.0 |

---

## 4. Simulation Protocol

### 4.1 Monte Carlo Design

Each simulation run follows a **N seeds x M learner types x K policies** factorial design:

- **Trajectory**: 5 sessions of 20 problems each (100 total problems per student)
- **Between-session forgetting**: 24 hours, power-law decay RS(t) = (1 + t/SS)^(-forgetting_rate)
- **Outcome measure**: Mastery score = mean across skills of (0.3 * RS + 0.4 * normalized_SS + 0.3 * Schema)

### 4.2 Policies Compared

| Policy | Description | Uses |
|---|---|---|
| **Meta-function** | Rule-based priority ordering from 5-state belief (oracle access) | RS, SS, Schema, WM, Affect |
| **Active inference** | pymdp EFE minimization over POMDP | All 5 states via belief update |
| **FSRS-only** | Spacing based on RS/SS only | RS, SS |
| **Fixed curriculum** | Textbook order, blocked practice | None (predetermined) |
| **Random** | Uniform random action selection | None |

### 4.3 Planned Scale

| Run | Students | Policies | Total Trajectories | Est. Runtime |
|---|---|---|---|---|
| Non-AI policies | 50,000 (sampled) | 4 | 200,000 | ~13 min |
| Active inference | 10,000 (sampled) | 1 | 10,000 | ~7 hrs (parallelized) |
| **Total** | -- | -- | **210,000** | -- |

---

## 5. Preliminary Results

From initial Monte Carlo runs (8 archetypes x 5 policies x 10 seeds = 400 trajectories):

### 5.1 Policy Comparison

The meta-function policy achieves the highest final mastery for **4 of 8** learner types:
- Novice, Fast Learner, Overconfident, Partial Knowledge

### 5.2 Key Findings

| Finding | Implication |
|---|---|
| FSRS-only causes frustration spirals in anxious, low-WM, and novice learners | Memory-only models are insufficient; affect and WM state are necessary for safety |
| Schema is the most important state dimension (ablation: -0.037 mastery) | Schema-level inference is the highest-leverage improvement target |
| WM ablation reveals over-triggering of `reduce_load` action | Meta-function calibration needed: current WM thresholds are too conservative |
| Frustration spirals are the primary failure mode for vulnerable learners | Priority 1 (break vicious cycles) is correctly positioned as highest priority |

### 5.3 State Ablation Summary

Dropping each state dimension from the meta-function's inputs while keeping the simulated student dynamics unchanged:

| Condition | Effect on Mastery | Interpretation |
|---|---|---|
| Full (baseline) | -- | All 5 states informing action selection |
| No Schema | -0.037 | Largest drop. System cannot match instruction to knowledge organization level |
| No Affect | -0.02 (est.) | Cannot detect or break frustration spirals |
| No WM | +/- small | Over-triggering reduce_load may offset WM-awareness benefit |
| No Memory (RS, SS) | -0.02 (est.) | Cannot schedule spacing or detect forgetting |
| No Discriminability | Small | Interleaving is infrequent; smaller effect |

---

## 6. Data Inventory

### 6.1 Completed (Available Now)

| Component | Location | Status |
|---|---|---|
| 8 archetype definitions | `simulation/learner_types.py` | Complete |
| Continuous population sampler | `simulation/learner_types.py:sample_learner_params()` | Complete |
| Simulated agent (ground-truth dynamics) | `simulation/simulated_agent.py` | Complete |
| Monte Carlo engine | `simulation/monte_carlo.py` | Complete |
| Analysis pipeline (comparison, recovery, ablation, landscape) | `simulation/analysis.py` | Complete |
| Meta-function (rule-based policy) | `meta_function.py` | Complete |
| Active inference agent (pymdp/EFE) | `active_inference/pomdp.py` | Complete |
| Precompute script (50K + 10K students) | `poster/precompute.py` | Written, not yet executed |
| Architecture diagram generator | `poster/generate_architecture.py` | Written, not yet executed |

### 6.2 In Progress / Remaining

| Component | Status | Notes |
|---|---|---|
| Large-scale precomputed results (210K trajectories) | Not yet run | `poster/precompute.py` ready to execute |
| Architecture diagram PNG | Not yet generated | `poster/generate_architecture.py` ready |
| Statistical significance tests | Not yet computed | Need large-N results first |
| Parameter recovery analysis (full inference engine) | Simplified version only | Current version uses naive proxy, not full Bayesian inference |
| Frontend visualization (Phase 5) | Delegated | Outline at `PHASE_5_FRONTEND_OUTLINE.md` |
| Validation against real student data (Phase 6) | Not started | Requires classroom deployment |
