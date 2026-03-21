"""
LLM-based explanation analyzer.

Single Claude API call per student explanation, returning:
- schema_level: none / partial / full (State 2)
- coherence: 0.0-1.0 (State 3 input)
- affect: frustrated / engaged / bored (State 5)
- confusion_flag: null or {confused_with, evidence} (State 4)

This serves States 2, 3, 4, and 5 from one call.
Cost scales with problems answered, not states tracked.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import anthropic

from domain.taxonomy import SCHEMA_RUBRICS


@dataclass
class AnalysisResult:
    schema_level: str           # "none", "partial", "full"
    schema_evidence: str        # LLM's reasoning for the classification
    coherence: float            # 0.0-1.0, how coherent the explanation is
    affect: str                 # "frustrated", "engaged", "bored"
    confusion_flag: Optional[dict] = None  # {confused_with: str, evidence: str} or None

    @property
    def schema_level_int(self) -> int:
        return {"none": 0, "partial": 1, "full": 2}.get(self.schema_level, 0)


def _build_rubric_text(skill: str) -> str:
    """Build the rubric text for a given skill."""
    rubric = SCHEMA_RUBRICS.get(skill)
    if not rubric:
        return f"No specific rubric for {skill}. Classify based on general understanding."

    lines = [
        f"Skill: {rubric['skill_name']} ({rubric['description']})",
        "",
        "Schema levels:",
        f"  NONE: {rubric['levels']['none']}",
        f"  PARTIAL: {rubric['levels']['partial']}",
        f"  FULL: {rubric['levels']['full']}",
    ]
    return "\n".join(lines)


def _build_prompt(problem: str, explanation: str, skill: str,
                   recent_context: str = "") -> str:
    """Build the analysis prompt for Claude."""
    rubric = _build_rubric_text(skill)

    prompt = f"""Analyze this student's explanation of their work on an algebra problem.

PROBLEM: {problem}
STUDENT'S EXPLANATION: {explanation}
SKILL BEING ASSESSED: {skill}

RUBRIC:
{rubric}

{f"RECENT CONTEXT (last few interactions): {recent_context}" if recent_context else ""}

Respond with a JSON object (no markdown formatting, just raw JSON):
{{
  "schema_level": "none" | "partial" | "full",
  "schema_evidence": "Brief explanation of why you chose this level",
  "coherence": 0.0-1.0 (how clear and well-structured is the explanation? 1.0 = perfectly clear, 0.0 = incoherent),
  "affect": "frustrated" | "engaged" | "bored",
  "confusion_flag": null or {{"confused_with": "skill_name", "evidence": "what in the explanation suggests confusion with another skill"}}
}}

Guidelines:
- schema_level: Match the student's reasoning to the rubric levels above. Focus on HOW they think, not just whether they got the right answer.
- coherence: Rate the clarity and completeness of the explanation itself. Fragmented, incomplete, or contradictory explanations score low.
- affect: Infer from tone, effort, and language. Short, dismissive answers suggest boredom. Expressions of frustration or confusion suggest frustration. Detailed, engaged responses suggest engagement.
- confusion_flag: Only flag if the explanation reveals confusion between THIS skill and another specific skill. For example, if the student explains distribution but calls it factoring, flag it."""

    return prompt


class ExplanationAnalyzer:
    """
    Analyzes student explanations using Claude API.

    Usage:
        analyzer = ExplanationAnalyzer()
        result = analyzer.analyze(
            problem="Simplify: a(b + c)",
            explanation="I multiplied a by each term inside the parentheses",
            skill="Dist-Right"
        )
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str = None):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def analyze(self, problem: str, explanation: str, skill: str,
                recent_context: str = "") -> AnalysisResult:
        """
        Analyze a student's explanation.

        Args:
            problem: The problem text shown to the student
            explanation: The student's free-text explanation
            skill: The primary skill being assessed (from SKILLS)
            recent_context: Optional string summarizing recent interactions

        Returns:
            AnalysisResult with schema_level, coherence, affect, confusion_flag
        """
        prompt = _build_prompt(problem, explanation, skill, recent_context)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse_response(message.content[0].text)

    def _parse_response(self, text: str) -> AnalysisResult:
        """Parse the LLM's JSON response into an AnalysisResult."""
        try:
            # Strip any markdown formatting if present
            clean = text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            data = json.loads(clean)

            schema_level = data.get("schema_level", "none")
            if schema_level not in ("none", "partial", "full"):
                schema_level = "none"

            affect = data.get("affect", "engaged")
            if affect not in ("frustrated", "engaged", "bored"):
                affect = "engaged"

            coherence = float(data.get("coherence", 0.5))
            coherence = max(0.0, min(1.0, coherence))

            confusion_flag = data.get("confusion_flag")
            if confusion_flag and not isinstance(confusion_flag, dict):
                confusion_flag = None

            return AnalysisResult(
                schema_level=schema_level,
                schema_evidence=data.get("schema_evidence", ""),
                coherence=coherence,
                affect=affect,
                confusion_flag=confusion_flag,
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: return defaults if parsing fails
            return AnalysisResult(
                schema_level="none",
                schema_evidence=f"Parse error. Raw response: {text[:200]}",
                coherence=0.5,
                affect="engaged",
                confusion_flag=None,
            )


class MockAnalyzer:
    """
    Mock analyzer for testing without API calls.
    Returns deterministic results based on explanation length and keywords.
    """

    def analyze(self, problem: str, explanation: str, skill: str,
                recent_context: str = "") -> AnalysisResult:
        words = explanation.lower().split()
        word_count = len(words)

        # Schema level from explanation quality
        if word_count < 3 or "idk" in explanation.lower() or "don't know" in explanation.lower():
            schema_level = "none"
        elif any(w in words for w in ["because", "since", "means", "property", "always"]):
            schema_level = "full"
        elif word_count > 5:
            schema_level = "partial"
        else:
            schema_level = "none"

        # Coherence from word count
        coherence = min(1.0, word_count / 20.0)

        # Affect from keywords
        if any(w in words for w in ["confused", "hard", "frustrated", "help", "stuck"]):
            affect = "frustrated"
        elif word_count < 3:
            affect = "bored"
        else:
            affect = "engaged"

        # Confusion flag: check for skill name misuse
        confusion_flag = None
        if "factor" in words and skill in ("Dist-Right", "Dist-Left"):
            confusion_flag = {"confused_with": "Factor", "evidence": "Used 'factor' while doing distribution"}
        elif "distribute" in words and skill == "Factor":
            confusion_flag = {"confused_with": "Dist-Right", "evidence": "Used 'distribute' while factoring"}

        return AnalysisResult(
            schema_level=schema_level,
            schema_evidence=f"Mock analysis: {word_count} words, keywords detected",
            coherence=coherence,
            affect=affect,
            confusion_flag=confusion_flag,
        )
