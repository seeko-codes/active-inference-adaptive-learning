import { useState, useRef, useCallback, useEffect } from 'react';
import { motion } from 'motion/react';
import { startSession, submitResponse } from '@/api';
import type { Problem, RespondResponse } from '@/api';
import {
  Square, CheckCircle2, XCircle, AlertTriangle,
  ArrowLeft, ChevronUp, ChevronDown,
} from 'lucide-react';
import { ActiveInferenceBackground } from './ActiveInferenceBackground';

const PHASE_LABELS: Record<BehaviorPhase, string> = {
  0: 'struggling',
  1: 'building',
  2: 'mastering',
  3: 'mixed',
};

interface SimLog {
  index: number;
  problemId: string;
  questionType: string;
  action: string;
  skill: string;
  tier: number;
  answerSent: string;
  correct: boolean;
  expectedAnswer: string;
  feedback: string;
  simPhase: string;
  error?: string;
}

interface TypeCoverage {
  count: number;
  correct: number;
  errors: number;
  actions: Set<string>;
}

const ALL_QUESTION_TYPES = [
  'simplify', 'identify_property', 'equivalent', 'non_equivalent',
  'worked_example', 'find_error', 'evaluate', 'boundary_test',
  'custom_operation', 'proof_disproof', 'fill_in_blank',
  'parentheses_placement', 'expand', 'inverse_rewrite',
  'strategic_compute', 'order_of_ops',
];

const ACHIEVABLE_ACTIONS = [
  'diagnostic_probe', 'worked_example', 'faded_example',
  'space_and_test', 'reteach',
  'increase_challenge', 'reduce_load',
];

const UNREACHABLE_ACTIONS = [
  'interleave',
  'boundary_test',
  'strategic_compute',
  'inverse_rewrite',
  'order_of_ops',
];

/*
 * Behavioral phases:
 *   0 "struggling": wrong + slow → high WM + low accuracy → frustrated
 *   1 "building":   correct + long explanations → builds schema
 *   2 "mastering":  correct + fast + long explanations → bored + low WM
 *   3 "mixed":      alternating correct/wrong + medium speed
 */
type BehaviorPhase = 0 | 1 | 2 | 3;

function getPhase(index: number, total: number): BehaviorPhase {
  const pct = index / total;
  if (pct < 0.2) return 0;
  if (pct < 0.45) return 1;
  if (pct < 0.7) return 2;
  return 3;
}

function getResponseTimeMs(phase: BehaviorPhase): number {
  switch (phase) {
    case 0: return 16000 + Math.random() * 5000;
    case 1: return 6000 + Math.random() * 8000;
    case 2: return 1000 + Math.random() * 3000;
    case 3: return 4000 + Math.random() * 10000;
  }
}

function getConfidence(phase: BehaviorPhase): number {
  switch (phase) {
    case 0: return Math.floor(Math.random() * 2) + 1;
    case 1: return Math.floor(Math.random() * 2) + 3;
    case 2: return Math.floor(Math.random() * 2) + 4;
    case 3: return Math.floor(Math.random() * 5) + 1;
  }
}

function shouldAnswerCorrectly(phase: BehaviorPhase, index: number): boolean {
  switch (phase) {
    case 0: return Math.random() < 0.1;
    case 1: return Math.random() < 0.8;
    case 2: return Math.random() < 0.95;
    case 3: return index % 2 === 0;
  }
}

const LONG_EXPLANATION = 'I carefully analyzed the algebraic structure of this expression by identifying the relevant properties and rules that apply. First I looked at the terms and their coefficients, then I applied the appropriate algebraic transformation step by step, verifying each intermediate result to make sure my reasoning was sound and consistent with the mathematical principles involved.';
const SHORT_EXPLANATION = 'I guessed';

function generateAnswer(problem: Problem, wantCorrect: boolean): string {
  const qt = problem.question_type;

  if (['equivalent', 'non_equivalent', 'boundary_test', 'custom_operation', 'proof_disproof'].includes(qt)) {
    if (!wantCorrect) return Math.random() > 0.5 ? 'true' : 'false';
    return Math.random() > 0.5 ? 'true' : 'false';
  }

  if (qt === 'identify_property' && problem.choices?.length) {
    return problem.choices[Math.floor(Math.random() * problem.choices.length)];
  }

  if (qt === 'worked_example') {
    return problem.example_after || 'understood';
  }

  if (qt === 'find_error') {
    if (!wantCorrect) return 'wrong answer';
    return problem.original_expression || '2x + 1';
  }

  if (qt === 'evaluate') {
    if (!wantCorrect) return '999';
    return String(Math.floor(Math.random() * 20));
  }

  if (!wantCorrect) return 'definitely wrong answer 999';

  const expr = problem.student_sees || '';
  const nums = expr.match(/\d+/g);
  if (nums && nums.length >= 2) {
    return String(Number(nums[0]) + Number(nums[1]));
  }
  return 'x + 1';
}

function generateExplanation(phase: BehaviorPhase): string {
  switch (phase) {
    case 0: return SHORT_EXPLANATION;
    case 1: return LONG_EXPLANATION;
    case 2: return LONG_EXPLANATION;
    case 3: return Math.random() > 0.5 ? LONG_EXPLANATION : SHORT_EXPLANATION;
  }
}

type Archetype = 'adaptive' | 'fast-learner' | 'struggling' | 'overconfident';

interface Props {
  onBack: () => void;
}

export function SimulationRunner({ onBack }: Props) {
  const [numProblems, setNumProblems] = useState(30);
  const [simStudentId, setSimStudentId] = useState('');
  const [archetype, setArchetype] = useState<Archetype>('adaptive');
  const [simSpeed, setSimSpeed] = useState<'fast' | 'normal'>('fast');
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [logs, setLogs] = useState<SimLog[]>([]);
  const [progress, setProgress] = useState(0);
  const [typeCoverage, setTypeCoverage] = useState<Record<string, TypeCoverage>>({});
  const [actionCoverage, setActionCoverage] = useState<Set<string>>(new Set());
  const [statusMsg, setStatusMsg] = useState('');
  const abortRef = useRef(false);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Never auto-scroll the page during live simulation updates.
  // Only the internal log container may stick to bottom, and only when
  // the user is already near the bottom (within 80px).
  useEffect(() => {
    const el = logContainerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom < 80) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs.length]);

  const runSimulation = useCallback(async () => {
    abortRef.current = false;
    setRunning(true);
    setDone(false);
    setLogs([]);
    setProgress(0);
    setTypeCoverage({});
    setActionCoverage(new Set());

    const coverage: Record<string, TypeCoverage> = {};
    const actions = new Set<string>();
    const delay = simSpeed === 'fast' ? 30 : 100;

    try {
      setStatusMsg('Starting session...');
      const session = await startSession(simStudentId.trim() || `sim_${Date.now()}`);
      let currentProblem: Problem = session.first_problem;
      actions.add('diagnostic_probe');
      setActionCoverage(new Set(actions));

      for (let i = 0; i < numProblems; i++) {
        if (abortRef.current) {
          setStatusMsg('Simulation stopped');
          break;
        }

        const qt = currentProblem.question_type;
        const act = currentProblem.action;
        const phase = getPhase(i, numProblems);
        setStatusMsg(`[${PHASE_LABELS[phase]}] Problem ${i + 1}/${numProblems}: ${qt} (${act})`);

        if (!coverage[qt]) {
          coverage[qt] = { count: 0, correct: 0, errors: 0, actions: new Set() };
        }
        coverage[qt].count++;
        coverage[qt].actions.add(act);
        actions.add(act);

        const wantCorrect = shouldAnswerCorrectly(phase, i);
        const answer = generateAnswer(currentProblem, wantCorrect);
        const explanation = generateExplanation(phase);
        const responseTimeMs = getResponseTimeMs(phase);
        const confidence = getConfidence(phase);

        let log: SimLog;

        try {
          const res: RespondResponse = await submitResponse(
            session.session_id,
            currentProblem.problem_id,
            answer,
            explanation,
            responseTimeMs,
            confidence,
          );

          log = {
            index: i + 1,
            problemId: currentProblem.problem_id,
            questionType: qt,
            action: act,
            skill: currentProblem.skill,
            tier: currentProblem.tier,
            answerSent: answer,
            correct: res.correct,
            expectedAnswer: res.expected_answer,
            feedback: res.feedback,
            simPhase: PHASE_LABELS[phase],
          };

          if (res.correct) coverage[qt].correct++;

          const missing: string[] = [];
          if (res.correct === undefined) missing.push('correct');
          if (!res.expected_answer && res.expected_answer !== '') missing.push('expected_answer');
          if (!res.feedback) missing.push('feedback');
          if (!res.next_problem) missing.push('next_problem');
          if (!res.action_reason) missing.push('action_reason');
          if (!res.state_summary) missing.push('state_summary');

          if (missing.length > 0) {
            log.error = `Missing fields: ${missing.join(', ')}`;
            coverage[qt].errors++;
          }

          if (res.next_problem) {
            const np = res.next_problem;
            const npmissing: string[] = [];
            if (!np.problem_id) npmissing.push('problem_id');
            if (!np.question_type) npmissing.push('question_type');
            if (!np.prompt) npmissing.push('prompt');
            if (np.tier === undefined) npmissing.push('tier');

            if (npmissing.length > 0) {
              log.error = (log.error ? log.error + '; ' : '') +
                `next_problem missing: ${npmissing.join(', ')}`;
              coverage[qt].errors++;
            }
          }

          currentProblem = res.next_problem;
        } catch (err) {
          log = {
            index: i + 1,
            problemId: currentProblem.problem_id,
            questionType: qt,
            action: act,
            skill: currentProblem.skill,
            tier: currentProblem.tier,
            answerSent: answer,
            correct: false,
            expectedAnswer: '',
            feedback: '',
            simPhase: PHASE_LABELS[phase],
            error: err instanceof Error ? err.message : String(err),
          };
          coverage[qt].errors++;
          setStatusMsg(`Fatal error at problem ${i + 1}: ${log.error}`);
          setLogs(prev => [...prev, log]);
          setProgress(((i + 1) / numProblems) * 100);
          setTypeCoverage({ ...coverage });
          setActionCoverage(new Set(actions));
          break;
        }

        setLogs(prev => [...prev, log]);
        setProgress(((i + 1) / numProblems) * 100);
        setTypeCoverage({ ...coverage });
        setActionCoverage(new Set(actions));

        await new Promise(r => setTimeout(r, delay));
      }

      if (!abortRef.current) setStatusMsg('Simulation complete');
    } catch (err) {
      setStatusMsg(`Failed to start session: ${err instanceof Error ? err.message : String(err)}`);
    }

    setRunning(false);
    setDone(true);
  }, [numProblems, simStudentId, simSpeed]);

  const totalCorrect = logs.filter(l => l.correct).length;
  const totalErrors = logs.filter(l => l.error).length;
  const unseenTypes = ALL_QUESTION_TYPES.filter(t => !typeCoverage[t]);
  const unseenAchievable = ACHIEVABLE_ACTIONS.filter(a => !actionCoverage.has(a));
  const surpriseActions = [...actionCoverage].filter(
    a => !ACHIEVABLE_ACTIONS.includes(a) && !UNREACHABLE_ACTIONS.includes(a)
  );

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-[#f0f0f4] antialiased relative">
      <ActiveInferenceBackground />

      {/* Sticky header */}
      <header className="relative z-20 bg-[#0a0a0a]/90 backdrop-blur-md border-b border-[#1c1c24] sticky top-0">
        <div className="max-w-6xl mx-auto px-6 h-[52px] flex items-center gap-4">
          <button
            onClick={onBack}
            disabled={running}
            className="flex items-center gap-1.5 text-zinc-500 hover:text-zinc-200 disabled:opacity-30
                       transition-colors cursor-pointer text-[12px] font-medium"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Back
          </button>
          <div className="w-px h-3 bg-[#1c1c24]" />
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-[#4a7c6f]" />
            <span className="text-[12px] font-semibold text-zinc-200">Simulation Runner</span>
          </div>
          {logs.length > 0 && (
            <div className="ml-auto flex items-center gap-4 font-mono text-[11px]">
              <span className="text-zinc-600">
                <span className="text-zinc-300">{logs.length}</span> problems
              </span>
              <span className="text-zinc-600">
                <span className="text-[#4a7c6f]">{totalCorrect}</span> correct
              </span>
              {totalErrors > 0 && (
                <span className="text-zinc-600">
                  <span className="text-red-400">{totalErrors}</span> errors
                </span>
              )}
            </div>
          )}
        </div>
      </header>

      {/* Two-column layout */}
      <div className="relative z-10 max-w-6xl mx-auto px-6 py-8">
        <div className="flex gap-8 items-start">

          {/* Left column — controls (38%) */}
          <div className="w-[38%] shrink-0 space-y-5">
            <div>
              <h1 className="text-[1.5rem] font-bold text-zinc-50 tracking-tight">
                Simulation Runner
              </h1>
              <p className="text-[12px] text-zinc-500 mt-1 leading-relaxed">
                Run synthetic learner sessions through the active inference engine.
              </p>
            </div>

            <div className="h-px bg-[#1c1c24]" />

            {/* Parameters */}
            <div className="space-y-4">
              {/* Learner ID */}
              <div>
                <label className="block text-[10px] font-semibold text-zinc-300 tracking-[0.08em] uppercase mb-1.5">
                  Learner ID
                </label>
                <input
                  type="text"
                  value={simStudentId}
                  onChange={e => setSimStudentId(e.target.value)}
                  disabled={running}
                  placeholder="sim_learner"
                  className="w-full h-9 bg-[#111116] border border-[#1c1c24] rounded-md px-3
                             text-[13px] text-zinc-100 placeholder:text-zinc-700 font-mono
                             focus:border-zinc-600 focus:outline-none disabled:opacity-40
                             hover:border-zinc-700 transition-colors"
                />
              </div>

              {/* Problem Count */}
              <div>
                <label className="block text-[10px] font-semibold text-zinc-300 tracking-[0.08em] uppercase mb-1.5">
                  Problem Count
                </label>
                <div className="flex items-stretch h-9">
                  <input
                    type="number"
                    min={1}
                    max={200}
                    value={numProblems}
                    onChange={e => setNumProblems(Math.max(1, parseInt(e.target.value) || 1))}
                    disabled={running}
                    className="flex-1 bg-[#111116] border border-[#1c1c24] border-r-0 rounded-l-md px-3
                               text-[13px] text-zinc-100 font-mono
                               focus:outline-none focus:border-zinc-600 disabled:opacity-40
                               hover:border-zinc-700 transition-colors
                               [appearance:textfield]
                               [&::-webkit-outer-spin-button]:appearance-none
                               [&::-webkit-inner-spin-button]:appearance-none"
                  />
                  <div className="flex flex-col border border-[#1c1c24] rounded-r-md overflow-hidden">
                    <motion.button
                      onClick={() => !running && setNumProblems(p => Math.min(200, p + 1))}
                      disabled={running}
                      whileTap={{ scale: 0.88 }}
                      transition={{ duration: 0.08 }}
                      className="group flex-1 w-7 flex items-center justify-center
                                 bg-[#111116] hover:bg-[#1c1c24] border-b border-[#1c1c24]
                                 disabled:opacity-40 cursor-pointer transition-colors duration-100
                                 disabled:cursor-not-allowed"
                    >
                      <ChevronUp className="w-2.5 h-2.5 text-zinc-600 group-hover:text-zinc-300 transition-colors" strokeWidth={2.5} />
                    </motion.button>
                    <motion.button
                      onClick={() => !running && setNumProblems(p => Math.max(1, p - 1))}
                      disabled={running}
                      whileTap={{ scale: 0.88 }}
                      transition={{ duration: 0.08 }}
                      className="group flex-1 w-7 flex items-center justify-center
                                 bg-[#111116] hover:bg-[#1c1c24]
                                 disabled:opacity-40 cursor-pointer transition-colors duration-100
                                 disabled:cursor-not-allowed"
                    >
                      <ChevronDown className="w-2.5 h-2.5 text-zinc-600 group-hover:text-zinc-300 transition-colors" strokeWidth={2.5} />
                    </motion.button>
                  </div>
                </div>
              </div>

              {/* Learner Archetype */}
              <div>
                <label className="block text-[10px] font-semibold text-zinc-300 tracking-[0.08em] uppercase mb-1.5">
                  Learner Archetype
                </label>
                <div className="relative">
                  <select
                    value={archetype}
                    onChange={e => setArchetype(e.target.value as Archetype)}
                    disabled={running}
                    className="w-full h-9 bg-[#111116] border border-[#1c1c24] rounded-md px-3 pr-8
                               text-[13px] text-zinc-300
                               focus:border-zinc-600 focus:outline-none disabled:opacity-40
                               hover:border-zinc-700 transition-colors cursor-pointer
                               appearance-none"
                  >
                    <option value="adaptive">Adaptive</option>
                    <option value="fast-learner">Fast Learner</option>
                    <option value="struggling">Struggling</option>
                    <option value="overconfident">Overconfident</option>
                  </select>
                  <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3 h-3 text-zinc-600 pointer-events-none" />
                </div>
              </div>

              {/* Simulation Speed */}
              <div>
                <label className="block text-[10px] font-semibold text-zinc-300 tracking-[0.08em] uppercase mb-1.5">
                  Simulation Speed
                </label>
                <div className="flex gap-2">
                  {(['fast', 'normal'] as const).map(s => (
                    <button
                      key={s}
                      onClick={() => !running && setSimSpeed(s)}
                      disabled={running}
                      className={`flex-1 h-9 rounded-md text-[12px] font-medium border transition-all duration-150 cursor-pointer
                        ${simSpeed === s
                          ? 'border-[#4a7c6f]/50 bg-[#4a7c6f]/10 text-[#4a7c6f]'
                          : 'border-[#1c1c24] text-zinc-600 hover:border-zinc-700 hover:text-zinc-400 bg-[#111116]'
                        } disabled:opacity-40`}
                    >
                      {s.charAt(0).toUpperCase() + s.slice(1)}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="h-px bg-[#1c1c24]" />

            {/* Run / Stop */}
            {!running ? (
              <motion.button
                onClick={runSimulation}
                initial="idle"
                whileHover="hover"
                whileTap={{ scale: 0.97 }}
                variants={{
                  idle: { backgroundColor: '#4a7c6f', boxShadow: '0 0 0px rgba(74,124,111,0)' },
                  hover: { backgroundColor: '#3d6b5e', boxShadow: '0 0 6px rgba(74,124,111,0.3)' },
                }}
                transition={{ duration: 0.15 }}
                className="group w-full h-11 rounded-md text-white text-[13px] font-semibold
                           flex items-center justify-between px-5 cursor-pointer"
              >
                <span>Run Simulation</span>
                <motion.span
                  variants={{ idle: { x: 0 }, hover: { x: 3 } }}
                  transition={{ duration: 0.15 }}
                  className="text-white/60 text-[14px] inline-block"
                >→</motion.span>
              </motion.button>
            ) : (
              <motion.button
                onClick={() => { abortRef.current = true; }}
                whileHover={{ opacity: 0.8 }}
                whileTap={{ scale: 0.97 }}
                className="w-full h-11 rounded-md bg-[#1c1c24] border border-[#27272a] text-red-400
                           text-[13px] font-medium flex items-center justify-between px-5 cursor-pointer"
              >
                <span>Stop Simulation</span>
                <Square className="w-3.5 h-3.5 text-red-400/60" />
              </motion.button>
            )}

            {/* Status */}
            {statusMsg && (
              <p className="font-mono text-[11px] text-zinc-600 leading-relaxed">{statusMsg}</p>
            )}

            {/* Progress bar */}
            {(running || (done && progress > 0)) && (
              <div className="h-px bg-[#1c1c24] overflow-hidden rounded-full">
                <motion.div
                  className="h-full bg-[#4a7c6f] rounded-full"
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                />
              </div>
            )}
          </div>

          {/* Right column — results */}
          <div className="flex-1 min-w-0 flex flex-col gap-4">
            {/* Terminal log panel */}
            <div className="bg-[#0d0d0d] border border-[#1c1c24] rounded-lg overflow-hidden flex flex-col min-h-[480px]">
              <div className="px-4 py-2.5 border-b border-[#1c1c24] flex items-center gap-2 shrink-0">
                <div className="w-1.5 h-1.5 rounded-full bg-[#4a7c6f]" />
                <span className="font-mono text-[10px] text-zinc-600 uppercase tracking-[0.12em]">
                  Session Log
                </span>
                {logs.length > 0 && (
                  <span className="ml-auto font-mono text-[10px] text-zinc-700">
                    {logs.length} entries
                  </span>
                )}
              </div>

              {logs.length === 0 ? (
                <div className="flex-1 flex flex-col items-center justify-center gap-1.5 py-16">
                  <p className="text-[13px] text-zinc-600">No simulation data yet.</p>
                  <p className="text-[11px] text-zinc-700">Run a session to observe the inference engine.</p>
                </div>
              ) : (
                <div ref={logContainerRef} className="flex-1 overflow-y-auto p-4 space-y-[2px]">
                  {logs.map(log => (
                    <motion.div
                      key={log.index}
                      initial={{ opacity: 0, x: -6 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.15 }}
                      className="flex items-center gap-3 font-mono text-[11px] py-[2px]"
                    >
                      <span className="text-zinc-700 tabular-nums w-6 text-right shrink-0">
                        {log.index}
                      </span>
                      <span className={`shrink-0 w-3 ${log.correct ? 'text-[#4a7c6f]' : 'text-zinc-600'}`}>
                        {log.correct ? '✓' : '✗'}
                      </span>
                      <span className="text-zinc-500 shrink-0">[{log.action}]</span>
                      <span className="text-zinc-400 shrink-0">{log.questionType}</span>
                      <span className="text-zinc-600 shrink-0 truncate max-w-[100px]">{log.skill}</span>
                      <span className="text-zinc-700 shrink-0">T{log.tier}</span>
                      <span className="text-zinc-700 shrink-0">[{log.simPhase}]</span>
                      {log.error && (
                        <span className="text-red-400 truncate text-[10px]">ERR: {log.error}</span>
                      )}
                    </motion.div>
                  ))}
                </div>
              )}
            </div>

            {/* Summary stats grid — visible when done */}
            {done && logs.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="grid grid-cols-2 gap-3"
              >
                <StatCard label="Problems Completed" value={String(logs.length)} />
                <StatCard label="Accuracy" value={`${Math.round((totalCorrect / logs.length) * 100)}%`} />
                <StatCard label="Actions Seen" value={String(actionCoverage.size)} />
                <StatCard label="Errors" value={String(totalErrors)} accent={totalErrors > 0} />
              </motion.div>
            )}
          </div>
        </div>

        {/* Coverage tables — full width, below two columns */}
        {(done || logs.length > 0) && (
          <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Question Type Coverage */}
            <div className="bg-[#111116] border border-[#1c1c24] rounded-lg overflow-hidden">
              <div className="px-4 py-3 border-b border-[#1c1c24]">
                <p className="text-[11px] font-semibold text-zinc-300">Question Type Coverage</p>
              </div>
              <div className="p-4 space-y-1.5">
                {ALL_QUESTION_TYPES.map(qt => {
                  const cov = typeCoverage[qt];
                  const seen = !!cov;
                  return (
                    <div key={qt} className={`flex items-center justify-between text-[11px] py-1 px-2 rounded ${
                      seen ? (cov.errors > 0 ? 'bg-red-500/8' : 'bg-[#4a7c6f]/8') : 'bg-[#1a1a1a]/30'
                    }`}>
                      <div className="flex items-center gap-2">
                        {seen ? (
                          cov.errors > 0 ? (
                            <AlertTriangle className="w-3 h-3 text-red-400" />
                          ) : (
                            <CheckCircle2 className="w-3 h-3 text-[#4a7c6f]" />
                          )
                        ) : (
                          <XCircle className="w-3 h-3 text-zinc-700" />
                        )}
                        <span className={`font-mono ${seen ? 'text-zinc-300' : 'text-zinc-700'}`}>
                          {qt}
                        </span>
                      </div>
                      {seen && (
                        <div className="flex items-center gap-2 font-mono">
                          <span className="text-zinc-600">{cov.count}x</span>
                          <span className="text-[#4a7c6f]">{cov.correct} ok</span>
                          {cov.errors > 0 && (
                            <span className="text-red-400">{cov.errors} err</span>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
                {unseenTypes.length > 0 && done && (
                  <p className="text-[11px] text-yellow-600 mt-3 font-mono">
                    {unseenTypes.length} type(s) not seen. Try more problems.
                  </p>
                )}
              </div>
            </div>

            {/* Action Coverage */}
            <div className="bg-[#111116] border border-[#1c1c24] rounded-lg overflow-hidden">
              <div className="px-4 py-3 border-b border-[#1c1c24]">
                <p className="text-[11px] font-semibold text-zinc-300">Action Coverage</p>
              </div>
              <div className="p-4">
                <p className="text-[10px] text-zinc-600 uppercase tracking-[0.1em] mb-2 font-medium">
                  Achievable Actions
                </p>
                <div className="space-y-1.5">
                  {ACHIEVABLE_ACTIONS.map(act => {
                    const seen = actionCoverage.has(act);
                    return (
                      <div key={act} className={`flex items-center gap-2 text-[11px] py-1 px-2 rounded ${
                        seen ? 'bg-[#4a7c6f]/8' : 'bg-[#1a1a1a]/30'
                      }`}>
                        {seen ? (
                          <CheckCircle2 className="w-3 h-3 text-[#4a7c6f]" />
                        ) : (
                          <XCircle className="w-3 h-3 text-zinc-700" />
                        )}
                        <span className={`font-mono ${seen ? 'text-zinc-300' : 'text-zinc-700'}`}>
                          {act}
                        </span>
                      </div>
                    );
                  })}
                </div>
                {unseenAchievable.length > 0 && done && (
                  <p className="text-[11px] text-yellow-600 mt-3 font-mono">
                    {unseenAchievable.length} achievable action(s) not triggered.
                  </p>
                )}
                {unseenAchievable.length === 0 && done && (
                  <p className="text-[11px] text-[#4a7c6f] mt-3 font-mono">
                    All achievable actions triggered.
                  </p>
                )}

                <div className="h-px bg-[#1c1c24] my-3" />

                <p className="text-[10px] text-zinc-600 uppercase tracking-[0.1em] mb-2 font-medium">
                  Backend-Limited
                </p>
                <div className="space-y-1.5">
                  {UNREACHABLE_ACTIONS.map(act => {
                    const seen = actionCoverage.has(act);
                    return (
                      <div key={act} className={`flex items-center gap-2 text-[11px] py-1 px-2 rounded ${
                        seen ? 'bg-yellow-500/10' : 'bg-[#1a1a1a]/20'
                      }`}>
                        {seen ? (
                          <AlertTriangle className="w-3 h-3 text-yellow-500" />
                        ) : (
                          <span className="w-3 h-3 text-center text-[10px] text-zinc-700 inline-flex items-center justify-center">—</span>
                        )}
                        <span className="font-mono text-zinc-700">{act}</span>
                        {seen && <span className="text-yellow-500 text-[10px]">unexpected</span>}
                      </div>
                    );
                  })}
                </div>

                {surpriseActions.length > 0 && (
                  <>
                    <div className="h-px bg-[#1c1c24] my-3" />
                    <p className="text-[10px] text-yellow-600 uppercase tracking-[0.1em] mb-2 font-medium">
                      Unexpected Actions
                    </p>
                    <div className="space-y-1.5">
                      {surpriseActions.map(act => (
                        <div key={act} className="flex items-center gap-2 text-[11px] py-1 px-2 rounded bg-yellow-500/10">
                          <AlertTriangle className="w-3 h-3 text-yellow-500" />
                          <span className="font-mono text-yellow-500">{act}</span>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="bg-[#111116] border border-[#1c1c24] rounded-lg px-4 py-3">
      <p className="text-[10px] text-zinc-600 uppercase tracking-[0.1em] mb-1.5 font-medium">{label}</p>
      <p className={`font-mono text-[2rem] font-semibold leading-none tabular-nums ${
        accent ? 'text-red-400' : 'text-zinc-100'
      }`}>{value}</p>
    </div>
  );
}
