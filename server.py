"""
FastAPI server bridging the adaptive learning backend to the frontend.

Endpoints:
  POST /session/start    → create session, get first problem
  POST /session/respond  → submit answer, get feedback + next problem
  GET  /session/{id}/state → current student state
"""

import json
import uuid
import time
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

try:
    from active_inference.pomdp import ActiveInferenceAgent
    HAS_POMDP = True
except ImportError:
    HAS_POMDP = False
from active_inference.state_space import rs_label, ss_label
from domain.taxonomy import SKILLS, TIER_BASE_EI, CONFUSABLE_PAIRS, effective_ei
from meta_function import select_action

app = FastAPI(title="Adaptive Learning System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load problem bank
DATA_DIR = Path(__file__).parent / "data"
_problems_cache = None


def _load_problems():
    global _problems_cache
    if _problems_cache is not None:
        return _problems_cache

    ks_path = DATA_DIR / "knowledge_space.json"
    if not ks_path.exists():
        _problems_cache = []
        return _problems_cache

    with open(ks_path) as f:
        dataset = json.load(f)

    from domain.render import render_problems, render_supplementary_problems
    from domain.knowledge_space import TIERS
    from domain.ast_nodes import Leaf, Op

    ast_registry = {}
    for tier_def in TIERS:
        for name, node in tier_def["seeds"].items():
            ast_registry[name] = node

    problems = render_problems(dataset, ast_registry, num_numeric_variants=3)
    problems.extend(render_supplementary_problems())
    _problems_cache = problems
    return _problems_cache


# Session storage (in-memory for prototype)
sessions: dict = {}


class SessionState:
    def __init__(self, student_id: str):
        self.session_id = str(uuid.uuid4())
        self.student_id = student_id
        self.ai_agent = ActiveInferenceAgent() if HAS_POMDP else None
        self.created_at = time.time()
        self.problems_completed = 0
        self.current_problem = None
        self.current_action = "diagnostic_probe"
        self.current_skill = SKILLS[0]
        self.history = []
        self.skill_accuracy = {s: [] for s in SKILLS}
        self.skill_schema_est = {s: 0 for s in SKILLS}
        self.rng = random.Random(int(time.time()))

        # Rolling stats
        self.total_correct = 0
        self.recent_correct = []
        self.recent_response_times = []
        self.affect_estimate = "engaged"
        self.wm_estimate = "moderate"
        self.mastery_estimates = {s: 0.0 for s in SKILLS}

    def select_problem(self, action: str, skill: str) -> dict:
        """Select a problem from the bank matching the action and skill."""
        problems = _load_problems()
        if not problems:
            return self._fallback_problem(action, skill)

        # Determine target tier from skill and current estimates
        schema = self.skill_schema_est.get(skill, 0)
        tier = min(7, max(0, schema * 2 + 1))

        # Filter by matching criteria
        candidates = [
            p for p in problems
            if p.get("tier", 0) <= tier + 1
            and p.get("tier", 0) >= max(0, tier - 1)
        ]

        # Prefer problems matching the action's question type
        action_to_qtypes = {
            "worked_example": ["worked_example"],
            "faded_example": ["simplify", "expand", "inverse_rewrite", "fill_in_blank"],
            "space_and_test": ["simplify", "evaluate", "expand", "strategic_compute", "fill_in_blank"],
            "reteach": ["worked_example", "identify_property", "inverse_rewrite", "fill_in_blank"],
            "interleave": ["identify_property", "equivalent", "boundary_test", "custom_operation"],
            "increase_challenge": ["simplify", "find_error", "expand", "strategic_compute", "custom_operation", "proof_disproof", "parentheses_placement"],
            "reduce_load": ["evaluate", "identify_property", "equivalent", "order_of_ops", "fill_in_blank"],
            "diagnostic_probe": ["simplify", "identify_property", "evaluate", "boundary_test", "order_of_ops", "fill_in_blank"],
        }
        preferred_qtypes = action_to_qtypes.get(action, ["simplify"])

        preferred = [p for p in candidates if p.get("question_type") in preferred_qtypes]
        if preferred:
            candidates = preferred

        # Filter by skill if possible
        skill_matches = [
            p for p in candidates
            if skill in p.get("skills_tested", [])
            or skill in p.get("derivation_path", [])
        ]
        if skill_matches:
            candidates = skill_matches

        if not candidates:
            candidates = problems[:50]

        problem = self.rng.choice(candidates)

        # Add scaffolding for worked/faded examples
        scaffolding = None
        if action == "faded_example" and problem.get("question_type") == "simplify":
            scaffolding = {
                "hint": f"Try applying the {skill.replace('-', ' ').lower()} property.",
            }
        elif action == "worked_example" and problem.get("question_type") != "worked_example":
            scaffolding = {
                "hint": f"Study the pattern and apply {skill}.",
                "worked_step": problem.get("expected_answer", ""),
            }

        result = {
            "problem_id": problem.get("problem_id", str(uuid.uuid4())),
            "question_type": problem.get("question_type", "simplify"),
            "prompt": problem.get("prompt", "Simplify:"),
            "student_sees": problem.get("student_sees", problem.get("pretty", "")),
            "tier": problem.get("tier", tier),
            "skill": skill,
            "action": action,
            "skills_tested": problem.get("skills_tested", [skill]),
        }

        # Add type-specific fields
        if problem.get("question_type") == "identify_property":
            result["expression_before"] = problem.get("expression_before", "")
            result["expression_after"] = problem.get("expression_after", "")
            result["choices"] = SKILLS
        elif problem.get("question_type") == "equivalent":
            result["expression_a"] = problem.get("expression_a", "")
            result["expression_b"] = problem.get("expression_b", "")
        elif problem.get("question_type") == "non_equivalent":
            result["expression_a"] = problem.get("expression_a", "")
            result["expression_b"] = problem.get("expression_b", "")
        elif problem.get("question_type") == "worked_example":
            result["example_before"] = problem.get("example_before", "")
            result["example_after"] = problem.get("example_after", "")
            result["rule_demonstrated"] = problem.get("rule_demonstrated", "")
            result["rule_description"] = problem.get("rule_description", "")
        elif problem.get("question_type") == "find_error":
            result["original_expression"] = problem.get("original_expression", "")
            result["claimed_answer"] = problem.get("claimed_answer", "")
        elif problem.get("question_type") == "evaluate":
            result["substitution"] = problem.get("substitution", {})
        elif problem.get("question_type") == "boundary_test":
            result["expression_a"] = problem.get("expression_a", "")
            result["expression_b"] = problem.get("expression_b", "")
        elif problem.get("question_type") == "custom_operation":
            result["expression_a"] = problem.get("student_sees", "").split("  vs  ")[0] if "  vs  " in problem.get("student_sees", "") else ""
            result["expression_b"] = problem.get("student_sees", "").split("  vs  ")[1] if "  vs  " in problem.get("student_sees", "") else ""
        elif problem.get("question_type") == "proof_disproof":
            pass  # Uses student_sees + prompt, no extra fields
        elif problem.get("question_type") == "fill_in_blank":
            pass  # Uses student_sees + prompt
        elif problem.get("question_type") == "parentheses_placement":
            pass  # Uses student_sees + prompt

        if scaffolding:
            result["scaffolding"] = scaffolding
        elif problem.get("scaffolding"):
            result["scaffolding"] = problem["scaffolding"]

        # Store expected answer internally
        self.current_problem = {**result, "_expected": problem.get("expected_answer")}
        return result

    def _fallback_problem(self, action: str, skill: str) -> dict:
        """Generate a simple problem when no problem bank is available."""
        a, b = self.rng.randint(2, 9), self.rng.randint(2, 9)
        problem = {
            "problem_id": f"fb_{self.problems_completed}",
            "question_type": "simplify",
            "prompt": "Simplify the following expression:",
            "student_sees": f"{a}(x + {b})",
            "tier": 3,
            "skill": skill,
            "action": action,
            "skills_tested": [skill],
        }
        self.current_problem = {
            **problem,
            "_expected": f"{a}x + {a * b}",
        }
        return problem

    def check_answer(self, answer: str) -> tuple[bool, str]:
        """Check if the student's answer is correct."""
        if not self.current_problem:
            return False, ""

        expected = self.current_problem.get("_expected")
        qtype = self.current_problem.get("question_type", "")

        if expected is None:
            return True, "No expected answer available"

        # Normalize for comparison
        def normalize(s):
            if s is None:
                return ""
            s = str(s).strip().lower()
            s = s.replace(" ", "").replace("×", "*").replace("·", "*")
            return s

        answer_norm = normalize(answer)
        expected_norm = normalize(expected)

        if qtype in ("equivalent", "non_equivalent", "boundary_test", "custom_operation"):
            # Boolean answer
            answer_bool = answer_norm in ("true", "yes", "1")
            expected_bool = expected_norm in ("true", "yes", "1")
            correct = answer_bool == expected_bool
        elif qtype == "proof_disproof":
            # Explanation-graded: accept any answer, quality comes from explanation
            answer_bool = answer_norm in ("true", "yes", "1")
            expected_bool = expected_norm in ("true", "yes", "1")
            correct = answer_bool == expected_bool
        elif qtype == "identify_property":
            correct = answer_norm == normalize(expected)
        else:
            # Try string comparison first, then numeric
            correct = answer_norm == expected_norm
            if not correct:
                try:
                    correct = abs(float(answer_norm) - float(expected_norm)) < 0.01
                except (ValueError, TypeError):
                    pass

        return correct, str(expected)

    def update_estimates(self, skill: str, correct: bool, response_time_ms: float,
                         explanation_quality: float):
        """Update internal estimates after a response."""
        # Accuracy tracking
        self.skill_accuracy[skill].append(correct)
        if len(self.skill_accuracy[skill]) > 20:
            self.skill_accuracy[skill] = self.skill_accuracy[skill][-20:]

        self.recent_correct.append(correct)
        if len(self.recent_correct) > 10:
            self.recent_correct = self.recent_correct[-10:]

        self.recent_response_times.append(response_time_ms)
        if len(self.recent_response_times) > 10:
            self.recent_response_times = self.recent_response_times[-10:]

        if correct:
            self.total_correct += 1

        # Schema estimate from explanation quality
        if explanation_quality > 0.65:
            self.skill_schema_est[skill] = min(2, self.skill_schema_est[skill] + 1)
        elif explanation_quality < 0.25 and self.skill_schema_est[skill] > 0:
            self.skill_schema_est[skill] = max(0, self.skill_schema_est[skill] - 1)

        # WM estimate from response time
        if response_time_ms > 15000:
            self.wm_estimate = "high"
        elif response_time_ms < 5000:
            self.wm_estimate = "low"
        else:
            self.wm_estimate = "moderate"

        # Affect estimate from patterns
        recent_acc = sum(self.recent_correct) / max(1, len(self.recent_correct))
        if recent_acc < 0.3 and self.wm_estimate == "high":
            self.affect_estimate = "frustrated"
        elif recent_acc > 0.8 and self.wm_estimate == "low":
            self.affect_estimate = "bored"
        else:
            self.affect_estimate = "engaged"

        # Mastery estimate per skill
        skill_acc = self.skill_accuracy.get(skill, [])
        if skill_acc:
            acc = sum(skill_acc[-5:]) / len(skill_acc[-5:])
            schema_bonus = self.skill_schema_est[skill] / 2.0
            self.mastery_estimates[skill] = round(acc * 0.6 + schema_bonus * 0.4, 3)

    def select_next_action(self) -> tuple[str, str, str]:
        """Use AI agent + meta function to select next action and skill."""
        # Find the skill most in need of attention
        weakest = min(SKILLS, key=lambda s: self.mastery_estimates.get(s, 0))

        # Use meta function for action selection
        schema = self.skill_schema_est.get(weakest, 0)
        skill_acc = self.skill_accuracy.get(weakest, [])
        recent_acc = sum(skill_acc[-5:]) / max(1, len(skill_acc[-5:])) if skill_acc else 0.5

        rs = "moderate" if recent_acc > 0.5 else "low" if recent_acc > 0.2 else "very_low"
        ss = "moderate" if len(skill_acc) > 10 else "low" if len(skill_acc) > 3 else "very_low"

        action_sel = select_action(
            skill=weakest,
            rs_label=rs,
            ss_label=ss,
            schema_label=["none", "partial", "full"][schema],
            wm_label=self.wm_estimate,
            affect=self.affect_estimate,
            effective_ei=TIER_BASE_EI.get(min(7, schema * 2 + 1), 5),
        )

        return action_sel.action, weakest, action_sel.reason


# ==========================================
# Request/Response Models
# ==========================================

class StartRequest(BaseModel):
    student_id: str = "student_1"


class RespondRequest(BaseModel):
    session_id: str
    problem_id: str
    answer: str
    explanation: str = ""
    response_time_ms: float
    confidence: int = 0


class SessionStateResponse(BaseModel):
    session_id: str
    problems_completed: int
    mastery_estimates: dict
    affect: str
    wm_load: str
    recent_accuracy: float
    skills: list


# ==========================================
# Endpoints
# ==========================================

@app.post("/session/start")
def start_session(req: StartRequest):
    state = SessionState(req.student_id)
    sessions[state.session_id] = state

    # First problem: diagnostic probe on weakest skill
    problem = state.select_problem("diagnostic_probe", SKILLS[0])

    return {
        "session_id": state.session_id,
        "first_problem": problem,
        "action": "diagnostic_probe",
        "action_reason": "Starting session: diagnosing initial state",
    }


@app.post("/session/respond")
def submit_response(req: RespondRequest):
    state = sessions.get(req.session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    # Check answer
    correct, expected = state.check_answer(req.answer)
    state.problems_completed += 1

    # Estimate explanation quality from length
    word_count = len(req.explanation.split()) if req.explanation else 0
    explanation_quality = min(1.0, word_count / 30.0) * 0.7 + (0.3 if correct else 0.0)

    # Feed to AI agent
    skill = state.current_problem.get("skill", SKILLS[0]) if state.current_problem else SKILLS[0]
    try:
        if state.ai_agent is not None:
            ai_action, ai_info = state.ai_agent.step(
                correct=correct,
                response_time_ms=req.response_time_ms,
                explanation_quality=explanation_quality,
            )
        else:
            raise RuntimeError("No POMDP agent")
    except Exception:
        ai_action = "faded_example"
        ai_info = {"beliefs": {}, "uncertainty_ratio": 0.5}

    # Update internal estimates
    state.update_estimates(skill, correct, req.response_time_ms, explanation_quality)

    # Select next action
    next_action, next_skill, action_reason = state.select_next_action()

    # Record history
    state.history.append({
        "problem_id": req.problem_id,
        "skill": skill,
        "correct": correct,
        "response_time_ms": req.response_time_ms,
        "confidence": req.confidence,
    })

    # Select next problem
    next_problem = state.select_problem(next_action, next_skill)

    # Feedback text
    if correct:
        feedback = "Correct!"
    else:
        feedback = f"Not quite. The answer is: {expected}"

    recent_acc = sum(state.recent_correct) / max(1, len(state.recent_correct))

    return {
        "correct": correct,
        "expected_answer": expected,
        "feedback": feedback,
        "next_problem": next_problem,
        "next_action": next_action,
        "next_skill": next_skill,
        "action_reason": action_reason,
        "state_summary": {
            "beliefs": ai_info.get("beliefs", {}),
            "uncertainty": round(ai_info.get("uncertainty_ratio", 0.5), 3),
            "affect": state.affect_estimate,
            "wm_load": state.wm_estimate,
            "problems_completed": state.problems_completed,
            "session_accuracy": round(state.total_correct / max(1, state.problems_completed), 3),
            "recent_accuracy": round(recent_acc, 3),
            "mastery_estimates": state.mastery_estimates,
        },
    }


@app.get("/session/{session_id}/state")
def get_session_state(session_id: str):
    state = sessions.get(session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    recent_acc = sum(state.recent_correct) / max(1, len(state.recent_correct))

    return {
        "session_id": state.session_id,
        "student_id": state.student_id,
        "problems_completed": state.problems_completed,
        "affect": state.affect_estimate,
        "wm_load": state.wm_estimate,
        "recent_accuracy": round(recent_acc, 3),
        "session_accuracy": round(state.total_correct / max(1, state.problems_completed), 3),
        "mastery_estimates": state.mastery_estimates,
        "skill_schemas": state.skill_schema_est,
        "history": state.history[-20:],
    }
