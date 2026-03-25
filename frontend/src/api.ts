const BASE = '';

export interface Problem {
  problem_id: string;
  question_type: string;
  prompt: string;
  student_sees?: string;
  tier: number;
  skill: string;
  action: string;
  skills_tested: string[];
  expression_before?: string;
  expression_after?: string;
  expression_a?: string;
  expression_b?: string;
  example_before?: string;
  example_after?: string;
  rule_demonstrated?: string;
  rule_description?: string;
  original_expression?: string;
  claimed_answer?: string;
  substitution?: Record<string, number>;
  choices?: string[];
  scaffolding?: { hint?: string; worked_step?: string };
}

export interface StateSummary {
  beliefs: Record<string, string>;
  uncertainty: number;
  affect: string;
  wm_load: string;
  problems_completed: number;
  session_accuracy: number;
  recent_accuracy: number;
  mastery_estimates: Record<string, number>;
}

export interface StartResponse {
  session_id: string;
  first_problem: Problem;
  action: string;
  action_reason: string;
}

export interface RespondResponse {
  correct: boolean;
  expected_answer: string;
  feedback: string;
  next_problem: Problem;
  next_action: string;
  next_skill: string;
  action_reason: string;
  state_summary: StateSummary;
}

export async function startSession(studentId: string): Promise<StartResponse> {
  const res = await fetch(`${BASE}/session/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ student_id: studentId }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Start session failed (${res.status}): ${text}`);
  }
  return res.json();
}

export async function submitResponse(
  sessionId: string,
  problemId: string,
  answer: string,
  explanation: string,
  responseTimeMs: number,
  confidence: number,
): Promise<RespondResponse> {
  const res = await fetch(`${BASE}/session/respond`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      problem_id: problemId,
      answer,
      explanation,
      response_time_ms: responseTimeMs,
      confidence,
    }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Submit response failed (${res.status}): ${text}`);
  }
  return res.json();
}
