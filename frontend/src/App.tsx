import { useState, useRef, useCallback } from 'react';
import type React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Toaster, toast } from 'sonner';
import { ProblemCard } from '@/components/ProblemCard';
import { FeedbackView } from '@/components/FeedbackView';
import { ConfidencePrompt } from '@/components/ConfidencePrompt';
import { SessionDashboard } from '@/components/SessionDashboard';
import { SimulationRunner } from '@/components/SimulationRunner';
import { ActiveInferenceBackground } from '@/components/ActiveInferenceBackground';
import { Input } from '@/components/ui/input';
import { startSession, submitResponse } from '@/api';
import type { Problem, StateSummary, RespondResponse } from '@/api';
import { Loader2, Activity } from 'lucide-react';

type Phase = 'login' | 'problem' | 'confidence' | 'feedback';

export default function App() {
  const [phase, setPhase] = useState<Phase>('login');
  const [showSimulation, setShowSimulation] = useState(false);
  const [studentId, setStudentId] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [problem, setProblem] = useState<Problem | null>(null);
  const [actionReason, setActionReason] = useState('');
  const [stateSummary, setStateSummary] = useState<StateSummary | null>(null);
  const [lastResponse, setLastResponse] = useState<RespondResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const pendingAnswer = useRef('');
  const pendingExplanation = useRef('');
  const problemStartTime = useRef(Date.now());

  const handleReset = useCallback(() => {
    setPhase('login');
    setShowSimulation(false);
    setSessionId('');
    setProblem(null);
    setStateSummary(null);
    setLastResponse(null);
    setActionReason('');
    setLoading(false);
  }, []);

  const handleStart = useCallback(async () => {
    const id = studentId.trim() || 'student_1';
    setLoading(true);
    try {
      const res = await startSession(id);
      setSessionId(res.session_id);
      setProblem(res.first_problem);
      setActionReason(res.action_reason);
      setPhase('problem');
      problemStartTime.current = Date.now();
    } catch {
      toast.error('Failed to connect. Is the inference server running?');
    } finally {
      setLoading(false);
    }
  }, [studentId]);

  const handleSubmitAnswer = useCallback((answer: string, explanation: string) => {
    pendingAnswer.current = answer;
    pendingExplanation.current = explanation;
    setPhase('confidence');
  }, []);

  const handleConfidence = useCallback(async (confidence: number) => {
    if (!problem || !sessionId) return;
    setLoading(true);
    const responseTimeMs = Date.now() - problemStartTime.current;
    try {
      const res = await submitResponse(
        sessionId, problem.problem_id,
        pendingAnswer.current, pendingExplanation.current,
        responseTimeMs, confidence,
      );
      setLastResponse(res);
      setStateSummary(res.state_summary);
      setActionReason(res.action_reason);
      setPhase('feedback');
    } catch {
      toast.error('Failed to submit response');
      setPhase('problem');
    } finally {
      setLoading(false);
    }
  }, [problem, sessionId]);

  const handleContinue = useCallback(() => {
    if (!lastResponse) return;
    setProblem(lastResponse.next_problem);
    setLastResponse(null);
    setPhase('problem');
    problemStartTime.current = Date.now();
  }, [lastResponse]);

  const completed = stateSummary?.problems_completed ?? 0;

  if (showSimulation) {
    return <SimulationRunner onBack={() => setShowSimulation(false)} />;
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-[#f0f0f4] antialiased relative">
      <ActiveInferenceBackground />

      <Toaster
        theme="dark"
        position="top-center"
        toastOptions={{
          style: {
            background: '#18181e',
            border: '1px solid #27272a',
            color: '#f0f0f4',
            fontSize: '13px',
          },
        }}
      />

      {/* ── Persistent Navbar ── */}
      <header className="relative z-20 w-full bg-[#0a0a0a]/85 backdrop-blur-md border-b border-[#27272a]/70 sticky top-0">
        <div className="w-full px-6 lg:px-10 h-[52px] flex items-center justify-between">
          <button
            onClick={handleReset}
            className="flex items-center gap-2.5 group cursor-pointer"
          >
            <div className="w-1.5 h-1.5 rounded-full bg-[#4a7c6f] transition-opacity group-hover:opacity-70" />
            <span className="text-[13px] font-semibold text-[#f0f0f4] tracking-tight">
              Active Inference
            </span>
            {phase !== 'login' && (
              <>
                <span className="text-[#3f3f46] text-[13px] select-none">·</span>
                <span className="text-[12px] text-[#3f3f46] hidden sm:inline">
                  Learner State Estimation
                </span>
              </>
            )}
          </button>

          <div className="flex items-center gap-3.5">
            {phase !== 'login' && stateSummary && (
              <>
                <span className="font-mono text-[11px] text-[#52525b]">
                  <span className="text-[#a0a0ab] tabular-nums">{completed}</span>
                  {' '}obs
                </span>
                <div className="w-px h-3 bg-[#27272a]" />
                <span className="font-mono text-[11px] text-[#52525b]">
                  <span className="text-[#a0a0ab] tabular-nums">
                    {Math.round(stateSummary.session_accuracy * 100)}%
                  </span>
                  {' '}acc
                </span>
                <div className="w-px h-3 bg-[#27272a]" />
              </>
            )}
            <button
              onClick={() => setShowSimulation(true)}
              className="flex items-center gap-1.5 text-[11px] text-[#3f3f46] hover:text-[#71717a] transition-colors cursor-pointer"
            >
              <Activity className="w-3 h-3" />
              Simulation
            </button>
          </div>
        </div>

        {/* Session progress indicator — only during active session */}
        {phase !== 'login' && (
          <div className="h-px bg-[#1c1c24]">
            <motion.div
              className="h-full bg-[#4a7c6f]/60"
              animate={{ width: `${Math.min((completed / Math.max(completed + 4, 10)) * 100, 96)}%` }}
              transition={{ duration: 0.7, ease: 'easeOut' }}
            />
          </div>
        )}
      </header>

      {/* ── Login / Entry ── */}
      {phase === 'login' && (
        <div className="relative z-10 flex items-start justify-center min-h-[calc(100vh-52px)] pt-[12vh] pb-16 px-6">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            className="flex gap-14 max-w-[800px] w-full items-start"
          >
            {/* ── Left: initialization form ── */}
            <div className="w-[320px] shrink-0">
              {/* Brand mark */}
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.06, duration: 0.4 }}
                className="flex items-center gap-2 mb-9"
              >
                <div className="w-1.5 h-1.5 rounded-full bg-[#4a7c6f] shadow-[0_0_5px_rgba(74,124,111,0.45)]" />
                <span className="text-[10px] font-semibold tracking-[0.24em] text-[#4a7c6f] uppercase">
                  Active Inference
                </span>
              </motion.div>

              {/* Heading block */}
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.12, duration: 0.4 }}
                className="mb-8"
              >
                <h1 className="text-[1.75rem] font-semibold leading-tight tracking-tight text-zinc-50 mb-3">
                  Adaptive Learning Lab
                </h1>
                <p className="text-[13px] text-zinc-400 leading-[1.65]">
                  Launch an adaptive learning session powered<br />
                  by real-time inference.
                </p>
              </motion.div>

              {/* Divider */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2, duration: 0.3 }}
                className="h-px bg-[#1c1c24] mb-7"
              />

              {/* Form */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.24, duration: 0.4 }}
                className="space-y-3"
              >
                <div>
                  <label className="block text-[10px] font-semibold text-zinc-200 tracking-[0.08em] uppercase mb-1.5">
                    Learner ID
                  </label>
                  <Input
                    value={studentId}
                    onChange={(e) => setStudentId(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !loading && handleStart()}
                    placeholder="Enter learner ID..."
                    className="h-10 bg-[#0d0d11] border-[#27272a] hover:border-zinc-600
                               focus:border-[#4a7c6f]/60 focus:ring-1 focus:ring-[#4a7c6f]/15
                               rounded-md px-3 text-[13px] text-zinc-100
                               placeholder:text-zinc-600 caret-[#4a7c6f]
                               transition-all duration-200 font-mono"
                  />
                </div>

                <motion.button
                  onClick={handleStart}
                  disabled={loading}
                  initial="idle"
                  whileHover={!loading ? 'hover' : 'idle'}
                  whileTap={!loading ? { scale: 0.98 } : {}}
                  variants={{
                    idle: {
                      backgroundColor: '#4a7c6f',
                      boxShadow: '0 0 0px rgba(74,124,111,0)',
                    },
                    hover: {
                      backgroundColor: '#3d6b5e',
                      boxShadow: '0 0 6px rgba(74,124,111,0.3)',
                    },
                  }}
                  transition={{ duration: 0.18 }}
                  className="group w-full h-12 rounded-md text-white text-[14px] font-semibold
                             disabled:opacity-30 relative overflow-hidden
                             flex items-center justify-between px-5 cursor-pointer"
                >
                  {/* Radial highlight sweep on hover */}
                  <span className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300
                                   bg-[radial-gradient(ellipse_at_50%_0%,rgba(255,255,255,0.08),transparent_65%)]
                                   pointer-events-none" />
                  {loading ? (
                    <>
                      <span className="text-white/50">Connecting</span>
                      <Loader2 className="w-4 h-4 animate-spin text-white/50" />
                    </>
                  ) : (
                    <>
                      <span>Start Adaptive Session</span>
                      <motion.span
                        variants={{ idle: { x: 0 }, hover: { x: 4 } }}
                        transition={{ duration: 0.18 }}
                        className="text-white/60 text-[15px] inline-block"
                      >
                        →
                      </motion.span>
                    </>
                  )}
                </motion.button>
              </motion.div>

              {/* Secondary action */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.38, duration: 0.35 }}
                className="mt-7"
              >
                <button
                  onClick={() => setShowSimulation(true)}
                  className="group flex items-center gap-2 text-[11px] text-zinc-500 hover:text-zinc-300 transition-colors duration-200 cursor-pointer"
                >
                  <div className="w-1 h-1 rounded-full bg-[#4a7c6f]/50 group-hover:bg-[#4a7c6f] transition-colors duration-200 shrink-0" />
                  <span>Run synthetic learner simulation</span>
                  <span className="inline-block transition-transform duration-200 group-hover:translate-x-1 text-zinc-600 group-hover:text-zinc-300">
                    →
                  </span>
                </button>
              </motion.div>
            </div>

            {/* ── Right: system state panel ── */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.5 }}
              className="flex-1 pt-[4.5rem] hidden md:block opacity-70"
            >
              <div className="space-y-6">
                {/* System status */}
                <SystemSection label="System">
                  <StatusRow label="Inference Engine" status="Ready" />
                  <StatusRow label="Policy Engine" status="Active" />
                  <StatusRow label="Belief Tracker" status="Online" />
                </SystemSection>

                <div className="h-px bg-[#1c1c24]" />

                {/* Domain scope */}
                <SystemSection label="Domain Scope">
                  <MetaRow k="Subject" v="Algebra" />
                  <MetaRow k="Skills" v="8 Concepts" />
                  <MetaRow k="Tiers" v="1 – 4" />
                  <MetaRow k="Question Types" v="16" />
                </SystemSection>

                <div className="h-px bg-[#1c1c24]" />

                {/* Framework */}
                <SystemSection label="Framework">
                  <MetaRow k="Method" v="Active Inference" />
                  <MetaRow k="Model" v="POMDP" />
                  <MetaRow k="Belief Updates" v="Bayesian" />
                  <MetaRow k="Action Selection" v="EFE Minimization" />
                </SystemSection>

                <div className="h-px bg-[#1c1c24]" />

                <p className="font-mono text-[10px] text-[#3f3f46]">
                  v0.1 · research build
                </p>
              </div>
            </motion.div>
          </motion.div>
        </div>
      )}

      {/* ── Session workspace ── */}
      {phase !== 'login' && (
        <div className="relative z-10 w-full py-10 px-6">
          {/* Fixed-width wrapper centered as a unit — card + sidebar never move */}
          <div className="mx-auto flex items-start gap-6" style={{ width: 'min(100%, 940px)' }}>

            {/* Problem card */}
            <div className="flex-1 min-w-0">
              <AnimatePresence mode="wait">
                {phase === 'problem' && problem && (
                  <motion.div
                    key={`p-${problem.problem_id}`}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
                  >
                    <ProblemCard
                      problem={problem}
                      actionReason={actionReason}
                      onSubmit={handleSubmitAnswer}
                      disabled={loading}
                    />
                  </motion.div>
                )}

                {phase === 'confidence' && (
                  <motion.div
                    key="conf"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
                  >
                    <ConfidencePrompt onSelect={handleConfidence} />
                  </motion.div>
                )}

                {phase === 'feedback' && lastResponse && (
                  <motion.div
                    key="fb"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
                  >
                    <FeedbackView
                      correct={lastResponse.correct}
                      expectedAnswer={lastResponse.expected_answer}
                      feedback={lastResponse.feedback}
                      onContinue={handleContinue}
                    />
                  </motion.div>
                )}
              </AnimatePresence>

              {loading && phase !== 'problem' && (
                <div className="flex items-center justify-center py-16">
                  <Loader2 className="w-4 h-4 text-[#3f3f46] animate-spin" />
                </div>
              )}
            </div>

            {/* Learner model panel */}
            <aside className="hidden lg:block w-[280px] shrink-0">
              <div className="sticky top-[68px]">
                <SessionDashboard
                  state={stateSummary}
                  actionReason={actionReason}
                  studentId={sessionId ? (studentId.trim() || 'student_1') : undefined}
                />
              </div>
            </aside>

          </div>
        </div>
      )}
    </div>
  );
}

// ── Login screen primitives ────────────────────────────────────────────────

function SystemSection({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2.5">
      <p className="text-[10px] font-semibold tracking-[0.1em] text-zinc-500 uppercase">
        {label}
      </p>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

function StatusRow({ label, status }: { label: string; status: string }) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="text-[11px] text-zinc-500">{label}</span>
      <div className="flex items-center gap-2 shrink-0">
        {/* Pulsing dot */}
        <span className="relative flex h-1.5 w-1.5 shrink-0">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#4a7c6f] opacity-40" />
          <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[#4a7c6f]" />
        </span>
        <span className="font-mono text-[11px] font-semibold text-zinc-300 tracking-[0.06em] uppercase w-[46px]">
          {status}
        </span>
      </div>
    </div>
  );
}

function MetaRow({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-0.5">
      <span className="text-[11px] text-zinc-600">{k}</span>
      <span className="font-mono text-[10px] text-zinc-400 text-right">{v}</span>
    </div>
  );
}
