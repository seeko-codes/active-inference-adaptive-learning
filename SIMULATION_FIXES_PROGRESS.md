# Simulation Fixes — Implementation Progress

**Status: ALL FIXES APPLIED**

All 10 fixes plus continuous schema updates are complete.

---

## COMPLETED

### Fix 5 — WM Overcorrection (meta_function.py)
- Added `recent_accuracy: float = 1.0` parameter to `select_action()`
- Priority 3 now requires `wm_label == "high" AND recent_accuracy < 0.4` (extraneous load only)

### Fix 3 — Empirical Distributions (learner_types.py)
- Added `metacognitive_bias: float = 0.0` to `CognitiveParams`
- Changed `get_initial_schema()` return type from `int` to `float`
- Replaced `sample_learner_params()` with empirically grounded distributions:
  - WM: Normal(4.0, 1.5) — Cowan (2001)
  - Math anxiety: 20% prevalence — Ashcraft (2002)
  - Learning rate correlated with WM (r ~ 0.5)
  - Forgetting: log-normal (FSRS empirical)
  - Frustration threshold inversely related to anxiety
- Updated `KNOWLEDGE_LEVELS` schema values from int to float (0->0.0, 1->0.5, 2->1.0)
- Updated all 8 archetypes with:
  - Float schema values (0->0.0, 1->0.5, 2->1.0)
  - Appropriate `metacognitive_bias` values (anxious=-0.2, overconfident=0.25, etc.)

### Fix 7 — Continuous Schema (simulated_agent.py, taxonomy.py, learner_types.py)
- `GroundTruthState.schema` is now float [0.0, 1.0]
- `_determine_correct()`: continuous schema bonus = `schema * 0.3`
- `_generate_explanation()`: interpolated quality and word counts for continuous schema
- `_update_schema()`: continuous increment with expertise reversal (`gain *= (1 - schema)^0.5`)
- `mastery_score()`: uses schema directly (already [0, 1])
- `taxonomy.py effective_ei()`: continuous reduction proportional to schema level

### Fix 4 — Confidence Observation Modality (pomdp.py, simulated_agent.py)
- `_generate_confidence()` now uses `metacognitive_bias` directly instead of boredom_threshold proxy
- Added `_build_confidence_A()`: P(confidence_bin | RS, Schema) with 3 bins
- `NUM_OBS` updated to `[2, 3, 3, 3]` (added confidence)
- `A_DEPENDENCIES` updated: confidence depends on [0, 2] (RS, Schema)
- `_discretize_obs()` now handles confidence bins
- `step()` accepts `confidence` parameter
- C vectors updated with confidence preferences

### Fix 6 — Ashcraft Affect Mechanism (pomdp.py)
- Replaced flat affect penalty `[-0.1, 0.0, -0.05]` with Ashcraft mechanism
- Frustration now adds ~1 effective WM bin (anxiety consumes WM capacity)
- Penalty hits harder at moderate/high WM (matches empirical finding)
- Boredom retained as separate small penalty (-0.05)

### Fix 2 — EI in A Matrix (pomdp.py, state_space.py)
- Added `EI_BINS` to `state_space.py` with 3 levels (low/medium/high)
- Added `discretize_ei()` function
- EI added as 6th state factor in POMDP (identity transition — controlled externally)
- Accuracy A matrix now shaped `(2, 5, 3, 3, 3, 3)` — conditioned on EI
- EI x Schema interaction encodes expertise reversal effect
- `build_B_matrices()` includes identity B for EI factor
- `build_D_vectors()` includes uniform prior for EI
- `set_ei_belief()` method sets EI to known one-hot delta
- `step()` accepts `ei_value` parameter and sets EI belief before inference
- `B_action_dependencies` updated for 6 factors

### Fix 10 — Parameter Learning (pomdp.py)
- Added `_model_lr = 0.1` to `__init__`
- Added `update_model()` method: increments Dirichlet concentration params for A matrix
- Learning rate decays: `lr / (1 + step * 0.05)`
- Called automatically from `step()` after inference

---

## COMPLETED (session 2)

### Fix 1 — Add Active Inference Policy to Monte Carlo (monte_carlo.py)
- Created `_ActiveInferencePolicy` class with `__init__`, `__call__`, `observe` methods
- Registered as class in `POLICIES` dict (`"active_inference": _ActiveInferencePolicy`)
- `run_trajectory()` detects class-based policies and instantiates per-trajectory
- `run_session()` calls `policy_fn.observe(obs)` for stateful policies
- `prompt_confidence=True` set in `agent.present_problem()` call

### Fix 8 — Population-Level Monte Carlo (monte_carlo.py)
- Added `run_population_monte_carlo()` function
- Samples n_students (default 500) from empirical distributions
- Runs each student through all policies, prints progress every 50 students

### Fix 5 tracking — Rolling Accuracy in Meta Function Policy (monte_carlo.py)
- Added `_MetaFunctionState` class tracking per-skill last-5 rolling accuracy
- Attached to agent instance for persistence across sessions
- `run_session()` records observations into rolling tracker
- `recent_accuracy` passed to `select_action()` in `_policy_meta_function`

### Fix 9 — B Matrix Citations (transition_model.py)
- Added citation blocks to `space_and_test`, `reteach`, `worked_example`, `interleave`
- Bumped `space_and_test` SS shift from 0.4 to 0.45 (FSRS calibrated)

### Continuous Schema Updates (analysis.py + monte_carlo.py)
- `_policy_meta_function`: schema_label discretized from continuous float
- `_policy_meta_function`: `schema_adequate >= 0.33` (was `>= 1`)
- `_policy_meta_function`: tier uses `int(schema * 4 + 1)` (was `schema * 2 + 1`)
- `_policy_fixed_curriculum`: mastery check `< 0.85` (was `< 2`)
- `analysis.py run_ablation()`: schema_label discretized from continuous float
- `analysis.py run_ablation()`: tier uses `int(schema * 4 + 1)`
- `analysis.py parameter_recovery()`: discretizes both true and inferred schema for comparison

---

## Validation (run after all fixes)
```bash
cd /Users/aatutor/adaptive-learning
python3 -c "
from simulation.monte_carlo import run_monte_carlo, run_population_monte_carlo
from simulation.analysis import run_full_analysis

results = run_monte_carlo(n_seeds=10, n_sessions=5, problems_per_session=20)
run_full_analysis(results, run_ablation_study=True)

pop_results = run_population_monte_carlo(n_students=500, n_sessions=5)
run_full_analysis(pop_results)
"
```
