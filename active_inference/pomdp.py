"""
pymdp integration: active inference outer loop for action selection.

Replaces the rule-based meta function with expected free energy (EFE)
minimization. EFE naturally decomposes into:

  G(pi) = epistemic_value + pragmatic_value

- High state uncertainty → epistemic dominates → system probes (diagnostic_probe)
- Low uncertainty → pragmatic dominates → system teaches optimally
- Surprise → posterior widens → epistemic rises → re-diagnosis

Architecture:
  Observations → State Estimators → Belief → pymdp Agent → Action

The 5 state dimensions are modeled as separate factors in the POMDP
(mean-field factorization). Each factor has its own A and B matrices.

Observation modalities:
  0: Accuracy (correct/incorrect → 2 levels)
  1: Response time regime (fast/normal/slow → 3 levels)
  2: Explanation quality (low/medium/high → 3 levels)

State factors (per skill):
  0: RS (5 bins)
  1: SS (4 bins)
  2: Schema (3 bins)
  3: WM (3 bins)
  4: Affect (3 bins)

Actions: 8 pedagogical actions from transition_model.py
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.special import softmax as _scipy_softmax

from active_inference.state_space import (
    RS_BINS, SS_BINS, SCHEMA_BINS, WM_BINS, AFFECT_BINS, EI_BINS,
    FACTORIZED_DIMS, discretize_ei,
)
from active_inference.transition_model import (
    ACTIONS, NUM_ACTIONS, get_transition_matrices,
)


# ==========================================
# Observation Model (A matrices)
# ==========================================
# A[m] maps hidden states to expected observations.
# Shape: (num_obs_m, num_states_f1, num_states_f2, ...) for factors
# that modality m depends on.
#
# With mean-field factorization and A_dependencies, each modality
# only depends on specific factors:
#   accuracy → RS, Schema, WM, Affect, EI_bin
#   response_time → WM, Affect
#   explanation_quality → Schema
#   confidence → RS, Schema (metacognitive calibration signal)

NUM_OBS = [2, 3, 3, 3]  # accuracy, response_time, explanation_quality, confidence

# A_dependencies: which state factors each obs modality depends on
# Factor indices: 0=RS, 1=SS, 2=Schema, 3=WM, 4=Affect, 5=EI_bin
A_DEPENDENCIES = [
    [0, 2, 3, 4, 5],  # accuracy depends on RS, Schema, WM, Affect, EI_bin
    [3, 4],            # response_time depends on WM, Affect
    [2],               # explanation_quality depends on Schema
    [0, 2],            # confidence depends on RS, Schema
]


def _build_accuracy_A():
    """
    P(correct | RS, Schema, WM, Affect, EI_bin).

    Shape: (2, 5, 3, 3, 3, 3) — (accuracy, RS, Schema, WM, Affect, EI_bin)

    Includes:
    - Ashcraft mechanism: frustration increases effective WM load (Fix 6)
    - Expertise reversal: high EI penalizes low-schema students,
      but schema compresses elements so full-schema students are unaffected (Fix 2)
    """
    n_rs, n_schema, n_wm, n_affect, n_ei = 5, 3, 3, 3, 3
    A = np.zeros((2, n_rs, n_schema, n_wm, n_affect, n_ei))

    rs_base = [0.1, 0.3, 0.5, 0.7, 0.9]

    for rs_i in range(n_rs):
        for schema_i in range(n_schema):
            for wm_i in range(n_wm):
                for affect_i in range(n_affect):
                    for ei_i in range(n_ei):
                        p = rs_base[rs_i]

                        # Schema bonus
                        p += [0.0, 0.1, 0.2][schema_i]

                        # Ashcraft mechanism (Fix 6): frustration adds ~1 effective WM bin
                        # instead of flat affect penalty
                        effective_wm = wm_i + (1 if affect_i == 0 else 0)
                        effective_wm = min(effective_wm, n_wm - 1)

                        # WM penalty using effective load
                        p -= [0.0, 0.05, 0.2][effective_wm]

                        # Boredom: slight penalty (disengagement)
                        if affect_i == 2:
                            p -= 0.05

                        # EI × Schema interaction (expertise reversal, Fix 2)
                        # No schema: high EI devastating
                        # Full schema: EI penalty eliminated (schema compresses elements)
                        ei_schema_penalty = [
                            [0.0, -0.1, -0.25],   # none: EI hurts
                            [0.0, -0.05, -0.1],    # partial: moderate
                            [0.05, 0.0, 0.0],      # full: no penalty (compression)
                        ]
                        p += ei_schema_penalty[schema_i][ei_i]

                        p = np.clip(p, 0.05, 0.95)
                        A[1, rs_i, schema_i, wm_i, affect_i, ei_i] = p
                        A[0, rs_i, schema_i, wm_i, affect_i, ei_i] = 1 - p

    return A


def _build_response_time_A():
    """
    P(response_time_regime | WM, Affect).

    Shape: (3, 3, 3) — (rt_regime, WM, Affect)
    rt_regime: 0=fast, 1=normal, 2=slow
    """
    n_wm, n_affect = 3, 3
    A = np.zeros((3, n_wm, n_affect))

    for wm_i in range(n_wm):
        for affect_i in range(n_affect):
            # High WM → slow, low WM → fast
            # Frustrated → slow, bored → fast (disengaged)
            if wm_i == 0:  # low WM
                base = [0.5, 0.4, 0.1]
            elif wm_i == 1:  # moderate
                base = [0.2, 0.6, 0.2]
            else:  # high
                base = [0.05, 0.3, 0.65]

            # Affect shifts
            if affect_i == 0:  # frustrated → slower
                shift = [-0.1, -0.05, 0.15]
            elif affect_i == 2:  # bored → faster (less effort)
                shift = [0.1, 0.0, -0.1]
            else:
                shift = [0, 0, 0]

            probs = np.clip(np.array(base) + np.array(shift), 0.01, 1.0)
            probs /= probs.sum()
            A[:, wm_i, affect_i] = probs

    return A


def _build_explanation_A():
    """
    P(explanation_quality | Schema).

    Shape: (3, 3) — (quality_level, Schema)
    quality_level: 0=low, 1=medium, 2=high
    """
    # Direct mapping: schema level strongly predicts explanation quality
    A = np.array([
        [0.8, 0.15, 0.05],  # P(low_quality | schema=none/partial/full)
        [0.15, 0.7, 0.2],   # P(medium_quality | ...)
        [0.05, 0.15, 0.75], # P(high_quality | ...)
    ])
    return A


def _build_confidence_A():
    """
    P(confidence_bin | RS, Schema).

    Shape: (3, 5, 3) — (confidence_bin, RS, Schema)
    confidence_bin: 0=low (1-2), 1=medium (3), 2=high (4-5)

    High RS + full schema → high confidence.
    Low RS + no schema → low confidence.
    Metacognitive bias shifts reports (handled in simulated agent, not here).
    """
    n_rs, n_schema = 5, 3
    A = np.zeros((3, n_rs, n_schema))

    for rs_i in range(n_rs):
        for schema_i in range(n_schema):
            # Competence drives confidence (assuming calibrated observer)
            competence = (rs_i / 4.0) * 0.5 + (schema_i / 2.0) * 0.5

            if competence > 0.65:
                probs = [0.05, 0.25, 0.70]
            elif competence > 0.35:
                probs = [0.20, 0.60, 0.20]
            else:
                probs = [0.65, 0.25, 0.10]

            A[:, rs_i, schema_i] = probs

    return A


def build_A_matrices():
    """Build all A matrices as numpy arrays."""
    return [
        _build_accuracy_A().astype(np.float64),
        _build_response_time_A().astype(np.float64),
        _build_explanation_A().astype(np.float64),
        _build_confidence_A().astype(np.float64),
    ]


# ==========================================
# Transition Model (B matrices)
# ==========================================
# B[f] shape: (num_states_f, num_states_f, num_actions)
# B[f][:, s, a] = P(s' | s, a) — column-normalized (axis 0)
#
# We reuse the transition matrices from transition_model.py,
# but transpose them to match pymdp convention.

def build_B_matrices():
    """
    Build B matrices for all 6 state factors across all 8 actions.

    Returns list of 6 JAX arrays, one per factor.
    Factors 0-4: rs, ss, schema, wm, affect (from transition_model.py)
    Factor 5: ei_bin (identity — EI is controlled externally by problem selection)
    """
    dim_names = ["rs", "ss", "schema", "wm", "affect"]
    dim_sizes = [RS_BINS["n"], SS_BINS["n"], SCHEMA_BINS["n"], WM_BINS["n"], AFFECT_BINS["n"]]

    B_matrices = []
    for dim_idx, (dim_name, dim_size) in enumerate(zip(dim_names, dim_sizes)):
        B_f = np.zeros((dim_size, dim_size, NUM_ACTIONS))

        for action_idx, action in enumerate(ACTIONS):
            T = get_transition_matrices(action)
            T_dim = T[dim_name]  # shape (n, n): T[i, j] = P(j | i)

            # pymdp convention: B[:, s, a] = P(s' | s, a)
            # Our T[i, j] = P(next=j | current=i), so T transposed gives
            # T.T[j, i] = P(next=j | current=i) — which is the column format
            B_f[:, :, action_idx] = T_dim.T

        # Ensure column normalization
        col_sums = B_f.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1, col_sums)
        B_f /= col_sums

        B_matrices.append(B_f.astype(np.float64))

    # Factor 5: EI_bin — identity transition (controlled externally)
    n_ei = EI_BINS["n"]
    B_ei = np.zeros((n_ei, n_ei, NUM_ACTIONS))
    for a in range(NUM_ACTIONS):
        B_ei[:, :, a] = np.eye(n_ei)
    B_matrices.append(B_ei.astype(np.float64))

    return B_matrices


# ==========================================
# Preferences (C vector)
# ==========================================
# C[m] encodes log-preferences over observations.
# Higher value = more preferred observation.

def build_C_vectors():
    """
    Build preference vectors for each observation modality.

    Preferences encode the system's goals:
    - Strongly prefer correct answers (pragmatic: student learning)
    - Mildly prefer normal response times (not too slow = overload)
    - Prefer high explanation quality (schema development)
    """
    return [
        np.array([-1.0, 2.0]),       # accuracy: strongly prefer correct
        np.array([0.5, 1.0, -0.5]),   # rt: prefer normal, avoid slow
        np.array([-0.5, 0.5, 1.5]),   # explanation: prefer high quality
        np.array([-0.5, 0.5, 1.0]),   # confidence: prefer high confidence
    ]


# ==========================================
# Initial Priors (D vectors)
# ==========================================

def build_D_vectors():
    """Uniform priors over all state factors (maximum uncertainty)."""
    dim_sizes = [
        RS_BINS["n"], SS_BINS["n"], SCHEMA_BINS["n"],
        WM_BINS["n"], AFFECT_BINS["n"], EI_BINS["n"],
    ]
    return [np.ones(d) / d for d in dim_sizes]


# ==========================================
# POMDP Agent Wrapper
# ==========================================

@dataclass
class POMDPConfig:
    """Configuration for the active inference agent."""
    policy_len: int = 1                    # Planning horizon (1 = myopic)
    use_utility: bool = True               # Use pragmatic value (preferences)
    use_states_info_gain: bool = True      # Use epistemic value (info gain)
    action_selection: str = "stochastic"   # "deterministic" or "stochastic"
    gamma: float = 8.0                     # Precision of action selection (higher = more deterministic)
    alpha: float = 16.0                    # Precision of beliefs


def _log_stable(x):
    """Numerically stable log."""
    return np.log(np.maximum(x, 1e-16))


def _factor_dot(A_m, beliefs, dep_factors):
    """
    Contract A matrix with belief vectors over dependent factors.

    A_m has shape (num_obs, dim_f1, dim_f2, ...) where f1, f2, ... are the
    dependent state factors. Contract each factor dimension with the
    corresponding belief vector to get predicted observation distribution.

    Returns: 1D array of shape (num_obs,)
    """
    result = A_m
    # Contract from the last factor backward to keep axis indices stable
    for i in reversed(range(len(dep_factors))):
        # Axis 0 is the obs dimension; factor i is at axis i+1
        result = np.tensordot(result, beliefs[dep_factors[i]], axes=([i + 1], [0]))
    return result


class ActiveInferenceAgent:
    """
    Pure-numpy active inference agent for the adaptive learning system.

    Implements expected free energy (EFE) minimization without JAX/XLA
    to avoid CUDA kernel compilation hangs. The state space is small
    enough that numpy is fast and the computation is transparent.

    Handles:
    - Converting inference engine outputs to POMDP observations
    - Running belief updates via fixed-point iteration (FPI)
    - Computing EFE for policy evaluation
    - Converting selected actions back to pedagogical actions
    - Tracking belief state across problems

    Usage:
        agent = ActiveInferenceAgent()
        action, info = agent.step(
            correct=True,
            response_time_ms=8000,
            explanation_quality=0.7,
        )
        # action is one of ACTIONS: "space_and_test", "worked_example", etc.
    """

    def __init__(self, config: POMDPConfig = None):
        if config is None:
            config = POMDPConfig()
        self.config = config

        self.A = build_A_matrices()
        self.B = build_B_matrices()
        self.C = build_C_vectors()
        self.D = build_D_vectors()

        self._beliefs = [d.copy() for d in self.D]
        self._step_count = 0
        self._rng = np.random.default_rng(0)
        self._model_lr = 0.1
        self._fpi_num_iter = 16

    def _discretize_obs(
        self,
        correct: bool,
        response_time_ms: float,
        explanation_quality: float,
        confidence: int = 0,
    ) -> list[int]:
        """Convert raw observables to discrete observation indices."""
        obs_accuracy = 1 if correct else 0

        if response_time_ms < 5000:
            obs_rt = 0
        elif response_time_ms < 15000:
            obs_rt = 1
        else:
            obs_rt = 2

        if explanation_quality < 0.35:
            obs_eq = 0
        elif explanation_quality < 0.65:
            obs_eq = 1
        else:
            obs_eq = 2

        if confidence <= 2:
            obs_conf = 0
        elif confidence <= 3:
            obs_conf = 1
        else:
            obs_conf = 2

        return [obs_accuracy, obs_rt, obs_eq, obs_conf]

    def set_ei_belief(self, ei_value: float):
        """Set the EI factor belief to a known value (one-hot delta)."""
        ei_bin = discretize_ei(ei_value)
        ei_belief = np.zeros(EI_BINS["n"])
        ei_belief[ei_bin] = 1.0
        self._beliefs[5] = ei_belief

    def _infer_states(self, obs_indices: list[int], prior: list[np.ndarray]) -> list[np.ndarray]:
        """
        Fixed-point iteration (FPI) for state inference.

        Given observations and priors, update beliefs over each state factor
        by iteratively incorporating likelihood evidence from all observation
        modalities that depend on that factor.
        """
        num_factors = len(prior)
        qs = [p.copy() for p in prior]

        for _iteration in range(self._fpi_num_iter):
            qs_old = [q.copy() for q in qs]

            for f in range(num_factors):
                # Accumulate log-likelihood from all modalities depending on factor f
                log_likelihood = np.zeros_like(qs[f])

                for m, obs_idx in enumerate(obs_indices):
                    dep_factors = A_DEPENDENCIES[m]
                    if f not in dep_factors:
                        continue

                    # Get A_m slice at the observed value
                    A_m = self.A[m]
                    # A_m[obs_idx] has shape (dim_f1, dim_f2, ...) for dep_factors
                    A_obs = A_m[obs_idx]

                    # Contract with beliefs of all OTHER dependent factors
                    result = A_obs
                    dep_list = list(dep_factors)
                    f_pos = dep_list.index(f)

                    # Contract from last to first, skipping factor f
                    for i in reversed(range(len(dep_list))):
                        if i == f_pos:
                            continue
                        result = np.tensordot(result, qs[dep_list[i]], axes=([i], [0]))

                    # result is now 1D with shape (dim_f,)
                    log_likelihood += _log_stable(np.maximum(result, 1e-16))

                # Update: q(s_f) ∝ prior(s_f) * exp(log_likelihood)
                log_q = _log_stable(prior[f]) + log_likelihood
                log_q -= log_q.max()
                qs[f] = np.exp(log_q)
                qs[f] /= qs[f].sum()

            # Check convergence
            max_delta = max(np.max(np.abs(qs[f] - qs_old[f])) for f in range(num_factors))
            if max_delta < 1e-6:
                break

        return qs

    def _compute_efe(self, beliefs: list[np.ndarray]) -> np.ndarray:
        """
        Compute negative expected free energy for each action.

        For each action a:
        1. Predict next state: q(s'_f) = B_f[:,:,a] @ q(s_f)
        2. Predict observations: q(o_m) = A_m contracted with q(s'_{deps_m})
        3. Pragmatic value: sum_m q(o_m) . C_m  (preference satisfaction)
        4. Epistemic value: H[q(o_m)] - E_q(o)[H[P(o|s')]]  (info gain)

        Returns: array of shape (NUM_ACTIONS,) with neg_efe per action
        """
        neg_efe = np.zeros(NUM_ACTIONS)

        for a in range(NUM_ACTIONS):
            # 1. Predict next states
            qs_next = []
            for f in range(len(beliefs)):
                T = self.B[f][:, :, a]  # (n_states, n_states)
                q_next_f = T @ beliefs[f]
                q_next_f /= q_next_f.sum() + 1e-16
                qs_next.append(q_next_f)

            # 2-4. Accumulate EFE terms across observation modalities
            pragmatic = 0.0
            epistemic = 0.0

            for m in range(len(self.A)):
                dep_factors = A_DEPENDENCIES[m]

                # Predicted observation: q(o_m) = A_m contracted with predicted beliefs
                qo_m = _factor_dot(self.A[m], qs_next, dep_factors)
                qo_m = np.maximum(qo_m, 1e-16)
                qo_m /= qo_m.sum()

                # Pragmatic value: q(o) . C
                if self.config.use_utility:
                    pragmatic += np.dot(qo_m, self.C[m])

                # Epistemic value (state info gain):
                # H[q(o)] - E_q(s')[H[P(o|s')]]
                if self.config.use_states_info_gain:
                    # H[q(o)] = entropy of predicted observations
                    H_qo = -np.sum(qo_m * _log_stable(qo_m))

                    # E_q(s')[H[P(o|s')]] = expected conditional entropy
                    # For each state config, compute H[P(o|s)] weighted by q(s)
                    A_m = self.A[m]
                    # We need to compute: sum over states of q(states) * H[A_m[:, states]]
                    # Use the outer product of beliefs for the dependent factors
                    dep_beliefs = [qs_next[f] for f in dep_factors]

                    # Compute entropy of each column of A
                    # A_m has shape (num_obs, d1, d2, ...) for dep factors
                    H_A = -np.sum(A_m * _log_stable(A_m), axis=0)
                    # H_A has shape (d1, d2, ...) — entropy for each state combination

                    # Weight by joint belief (outer product of marginals)
                    expected_H = H_A
                    for i in reversed(range(len(dep_factors))):
                        expected_H = np.tensordot(expected_H, dep_beliefs[i], axes=([i], [0]))
                    # expected_H is now scalar

                    epistemic += H_qo - float(expected_H)

            neg_efe[a] = pragmatic + epistemic

        return neg_efe

    def step(
        self,
        correct: bool,
        response_time_ms: float,
        explanation_quality: float,
        confidence: int = 3,
        ei_value: float = 5.0,
    ) -> tuple[str, dict]:
        """
        Process one observation and select the next action.

        Args:
            correct: Whether the student answered correctly
            response_time_ms: Response time in milliseconds
            explanation_quality: Explanation quality score [0, 1]
            confidence: Self-reported confidence (1-5)
            ei_value: Element interactivity of the problem presented

        Returns:
            (action_name, info_dict) where info contains beliefs, EFE, etc.
        """
        obs_indices = self._discretize_obs(correct, response_time_ms, explanation_quality, confidence)

        # Set EI belief to known value (we chose the problem)
        self.set_ei_belief(ei_value)

        # Compute empirical prior
        prior = [d.copy() for d in self.D] if self._step_count == 0 else self._beliefs

        # Infer posterior beliefs
        self._beliefs = self._infer_states(obs_indices, prior)

        # Compute EFE for each action
        neg_efe = self._compute_efe(self._beliefs)

        # Policy posterior via softmax
        q_pi = _scipy_softmax(self.config.gamma * neg_efe)

        # Select action
        if self.config.action_selection == "deterministic":
            action_int = int(np.argmax(q_pi))
        else:
            action_int = int(self._rng.choice(NUM_ACTIONS, p=q_pi))
        action_name = ACTIONS[action_int]

        # Build info dict
        info = self._build_info(q_pi, neg_efe, action_int)

        # Update generative model from observation (Fix 10)
        self.update_model(obs_indices, action_int)

        # Update prior for next step using transition model
        self._update_prior(action_int)

        self._step_count += 1
        return action_name, info

    def _update_prior(self, action_idx: int):
        """Update beliefs through the transition model for the next step."""
        new_beliefs = []
        for f in range(len(self._beliefs)):
            T = self.B[f][:, :, action_idx]
            new_prior = T @ self._beliefs[f]
            new_prior /= new_prior.sum() + 1e-16
            new_beliefs.append(new_prior)
        self._beliefs = new_beliefs

    def _build_info(self, q_pi, neg_efe, action_idx: int) -> dict:
        """Build interpretable info dict from inference results."""
        dim_names = ["rs", "ss", "schema", "wm", "affect", "ei_bin"]
        dim_labels = {
            "rs": RS_BINS["labels"],
            "ss": SS_BINS["labels"],
            "schema": SCHEMA_BINS["labels"],
            "wm": WM_BINS["labels"],
            "affect": AFFECT_BINS["labels"],
            "ei_bin": EI_BINS["labels"],
        }

        beliefs_map = {}
        beliefs_entropy = {}
        for i, (name, belief) in enumerate(zip(dim_names, self._beliefs)):
            b = belief
            map_idx = int(np.argmax(b))
            beliefs_map[name] = dim_labels[name][map_idx]
            b_safe = np.clip(b, 1e-10, 1.0)
            entropy = -np.sum(b_safe * np.log(b_safe))
            beliefs_entropy[name] = float(entropy)

        policy_probs = {ACTIONS[i]: float(q_pi[i]) for i in range(NUM_ACTIONS)}
        efe_values = {ACTIONS[i]: float(neg_efe[i]) for i in range(NUM_ACTIONS)}

        total_entropy = sum(beliefs_entropy.values())
        max_entropy = sum(np.log(d) for d in [5, 4, 3, 3, 3, 3])

        return {
            "action": ACTIONS[action_idx],
            "beliefs": beliefs_map,
            "entropy": beliefs_entropy,
            "total_entropy": total_entropy,
            "uncertainty_ratio": total_entropy / max_entropy,
            "policy_probs": policy_probs,
            "efe": efe_values,
            "step": self._step_count,
        }

    def get_beliefs(self) -> dict:
        """Get current belief state as readable dict."""
        dim_names = ["rs", "ss", "schema", "wm", "affect", "ei_bin"]
        return {
            name: belief.tolist()
            for name, belief in zip(dim_names, self._beliefs)
        }

    def update_model(self, obs_indices: list, action_idx: int):
        """
        Update A Dirichlet concentration parameters from observation.

        Implements online model learning (Friston et al., 2017):
        - A update: increment (inferred_state, observation) co-occurrences
        - Learning rate decays with experience (early obs have larger updates)
        """
        lr = self._model_lr / (1.0 + self._step_count * 0.05)

        for m, obs_idx in enumerate(obs_indices):
            dep_factors = A_DEPENDENCIES[m]
            # Build index into A matrix using MAP states of dependent factors
            idx = [obs_idx] + [int(np.argmax(self._beliefs[f])) for f in dep_factors]
            self.A[m][tuple(idx)] += lr
            # Renormalize along observation axis
            norm_idx = [slice(None)] + [int(np.argmax(self._beliefs[f])) for f in dep_factors]
            col = self.A[m][tuple(norm_idx)]
            self.A[m][tuple(norm_idx)] = col / col.sum()

    def reset(self):
        """Reset beliefs to uniform prior."""
        self._beliefs = [d.copy() for d in self.D]
        self._step_count = 0
