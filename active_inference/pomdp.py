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
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field

from pymdp.agent import Agent as PyMDPAgent

from active_inference.state_space import (
    RS_BINS, SS_BINS, SCHEMA_BINS, WM_BINS, AFFECT_BINS,
    FACTORIZED_DIMS,
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
#   accuracy → RS, Schema, WM, Affect
#   response_time → WM, Affect
#   explanation_quality → Schema

NUM_OBS = [2, 3, 3]  # accuracy, response_time, explanation_quality

# A_dependencies: which state factors each obs modality depends on
# Factor indices: 0=RS, 1=SS, 2=Schema, 3=WM, 4=Affect
A_DEPENDENCIES = [
    [0, 2, 3, 4],  # accuracy depends on RS, Schema, WM, Affect
    [3, 4],         # response_time depends on WM, Affect
    [2],            # explanation_quality depends on Schema
]


def _build_accuracy_A():
    """
    P(correct | RS, Schema, WM, Affect).

    Shape: (2, 5, 3, 3, 3) — (accuracy, RS, Schema, WM, Affect)

    Higher RS → more correct. Full schema → more correct.
    High WM → less correct. Frustrated → less correct.
    """
    n_rs, n_schema, n_wm, n_affect = 5, 3, 3, 3
    A = np.zeros((2, n_rs, n_schema, n_wm, n_affect))

    # Base P(correct) from RS
    rs_base = [0.1, 0.3, 0.5, 0.7, 0.9]

    for rs_i in range(n_rs):
        for schema_i in range(n_schema):
            for wm_i in range(n_wm):
                for affect_i in range(n_affect):
                    p = rs_base[rs_i]

                    # Schema bonus
                    p += [0.0, 0.1, 0.2][schema_i]

                    # WM penalty
                    p -= [0.0, 0.05, 0.2][wm_i]

                    # Affect modifier
                    p += [-0.1, 0.0, -0.05][affect_i]  # frustrated, engaged, bored

                    p = np.clip(p, 0.05, 0.95)
                    A[1, rs_i, schema_i, wm_i, affect_i] = p      # P(correct)
                    A[0, rs_i, schema_i, wm_i, affect_i] = 1 - p  # P(incorrect)

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


def build_A_matrices():
    """Build all A matrices as JAX arrays."""
    return [
        jnp.array(_build_accuracy_A(), dtype=jnp.float32),
        jnp.array(_build_response_time_A(), dtype=jnp.float32),
        jnp.array(_build_explanation_A(), dtype=jnp.float32),
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
    Build B matrices for all 5 state factors across all 8 actions.

    Returns list of 5 JAX arrays, one per factor.
    Each has shape (num_states_f, num_states_f, 8).
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

        B_matrices.append(jnp.array(B_f, dtype=jnp.float32))

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
        jnp.array([-1.0, 2.0], dtype=jnp.float32),     # accuracy: strongly prefer correct
        jnp.array([0.5, 1.0, -0.5], dtype=jnp.float32), # rt: prefer normal, avoid slow
        jnp.array([-0.5, 0.5, 1.5], dtype=jnp.float32), # explanation: prefer high quality
    ]


# ==========================================
# Initial Priors (D vectors)
# ==========================================

def build_D_vectors():
    """Uniform priors over all state factors (maximum uncertainty)."""
    dim_sizes = [RS_BINS["n"], SS_BINS["n"], SCHEMA_BINS["n"], WM_BINS["n"], AFFECT_BINS["n"]]
    return [jnp.ones(d, dtype=jnp.float32) / d for d in dim_sizes]


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


class ActiveInferenceAgent:
    """
    Wrapper around pymdp Agent for the adaptive learning system.

    Handles:
    - Converting inference engine outputs to POMDP observations
    - Running belief updates and policy selection
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

        # pymdp expects all actions across all factors
        # Only factor 0 (RS) is "directly controlled" — but actually
        # all factors are affected by the same single action.
        # We model this as: all factors share the same action (8 actions).
        num_controls = [NUM_ACTIONS] * 5  # Each factor has 8 possible transitions

        # B_action_dependencies: which action factors each state factor depends on
        # All state factors depend on the same single action choice
        # We use a single control factor with 8 options
        # Actually with pymdp, if all factors share the same action,
        # we need one control factor with NUM_ACTIONS options
        # and all B factors depend on it.

        self._agent = PyMDPAgent(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            A_dependencies=A_DEPENDENCIES,
            num_controls=[NUM_ACTIONS],
            B_action_dependencies=[[0], [0], [0], [0], [0]],
            policy_len=config.policy_len,
            use_utility=config.use_utility,
            use_states_info_gain=config.use_states_info_gain,
            action_selection=config.action_selection,
            gamma=config.gamma,
            alpha=config.alpha,
            inference_algo="fpi",
            batch_size=1,
        )

        # Current beliefs — start with priors
        self._beliefs = [d[None, :] for d in self.D]  # Add batch dim
        self._step_count = 0
        self._rng_key = jax.random.PRNGKey(0)

    def _discretize_obs(
        self,
        correct: bool,
        response_time_ms: float,
        explanation_quality: float,
    ) -> list:
        """Convert raw observables to discrete observation indices."""
        # Accuracy: 0=incorrect, 1=correct
        obs_accuracy = 1 if correct else 0

        # Response time regime: 0=fast (<5s), 1=normal (5-15s), 2=slow (>15s)
        if response_time_ms < 5000:
            obs_rt = 0
        elif response_time_ms < 15000:
            obs_rt = 1
        else:
            obs_rt = 2

        # Explanation quality: 0=low (<0.35), 1=medium (0.35-0.65), 2=high (>0.65)
        if explanation_quality < 0.35:
            obs_eq = 0
        elif explanation_quality < 0.65:
            obs_eq = 1
        else:
            obs_eq = 2

        return [
            jnp.array([obs_accuracy]),
            jnp.array([obs_rt]),
            jnp.array([obs_eq]),
        ]

    def step(
        self,
        correct: bool,
        response_time_ms: float,
        explanation_quality: float,
    ) -> tuple[str, dict]:
        """
        Process one observation and select the next action.

        This is the main active inference loop:
        1. Discretize observations
        2. Update beliefs (posterior inference)
        3. Evaluate policies via expected free energy
        4. Select action

        Args:
            correct: Whether the student answered correctly
            response_time_ms: Response time in milliseconds
            explanation_quality: Explanation quality score [0, 1]

        Returns:
            (action_name, info_dict) where info contains beliefs, EFE, etc.
        """
        obs = self._discretize_obs(correct, response_time_ms, explanation_quality)

        # Compute empirical prior from beliefs and previous action
        if self._step_count == 0:
            prior = [d[None, :] for d in self.D]
        else:
            prior = self._beliefs

        # Infer posterior beliefs
        qs = self._agent.infer_states(obs, empirical_prior=prior)
        # qs is list of arrays with shape (batch, time, num_states_f)
        # Extract the latest timestep
        self._beliefs = [q[:, -1, :] for q in qs]

        # Evaluate policies and compute EFE
        q_pi, G = self._agent.infer_policies(qs)

        # Select action (rng_key needs batch dim for vmapped agent)
        self._rng_key, subkey = jax.random.split(self._rng_key)
        action_idx = self._agent.sample_action(q_pi, rng_key=subkey[None, :])
        action_int = int(action_idx[0, 0])
        action_name = ACTIONS[action_int]

        # Compute epistemic vs pragmatic decomposition for interpretability
        info = self._build_info(q_pi, G, action_int)

        # Update prior for next step using transition model
        self._update_prior(action_int)

        self._step_count += 1
        return action_name, info

    def _update_prior(self, action_idx: int):
        """Update beliefs through the transition model for the next step."""
        new_beliefs = []
        for f, (belief_f, B_f) in enumerate(zip(self._beliefs, self.B)):
            # B_f[:, :, action] @ belief = new prior
            T = B_f[:, :, action_idx]  # (n_states, n_states)
            # belief_f shape: (batch, n_states)
            new_prior = jnp.matmul(belief_f, T.T)  # (batch, n_states)
            # Normalize
            new_prior = new_prior / jnp.sum(new_prior, axis=-1, keepdims=True)
            new_beliefs.append(new_prior)
        self._beliefs = new_beliefs

    def _build_info(self, q_pi, G, action_idx: int) -> dict:
        """Build interpretable info dict from inference results."""
        dim_names = ["rs", "ss", "schema", "wm", "affect"]
        dim_labels = {
            "rs": RS_BINS["labels"],
            "ss": SS_BINS["labels"],
            "schema": SCHEMA_BINS["labels"],
            "wm": WM_BINS["labels"],
            "affect": AFFECT_BINS["labels"],
        }

        # Extract MAP beliefs for each factor
        beliefs_map = {}
        beliefs_entropy = {}
        for i, (name, belief) in enumerate(zip(dim_names, self._beliefs)):
            b = np.array(belief[0])  # Remove batch dim
            map_idx = int(np.argmax(b))
            beliefs_map[name] = dim_labels[name][map_idx]
            # Entropy: -sum(p * log(p))
            b_safe = np.clip(b, 1e-10, 1.0)
            entropy = -np.sum(b_safe * np.log(b_safe))
            beliefs_entropy[name] = float(entropy)

        # Policy probabilities
        q_pi_np = np.array(q_pi[0])
        policy_probs = {ACTIONS[i]: float(q_pi_np[i]) for i in range(NUM_ACTIONS)}

        # EFE values
        G_np = np.array(G[0])
        efe_values = {ACTIONS[i]: float(G_np[i]) for i in range(NUM_ACTIONS)}

        # Total uncertainty
        total_entropy = sum(beliefs_entropy.values())
        max_entropy = sum(np.log(d) for d in [5, 4, 3, 3, 3])

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
        dim_names = ["rs", "ss", "schema", "wm", "affect"]
        return {
            name: np.array(belief[0]).tolist()
            for name, belief in zip(dim_names, self._beliefs)
        }

    def reset(self):
        """Reset beliefs to uniform prior."""
        self._beliefs = [d[None, :] for d in self.D]
        self._step_count = 0
