import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Toaster, toast } from 'sonner';
import { ProblemCard } from '@/components/ProblemCard';
import { FeedbackView } from '@/components/FeedbackView';
import { ConfidencePrompt } from '@/components/ConfidencePrompt';
import { SessionDashboard } from '@/components/SessionDashboard';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import { startSession, submitResponse } from '@/api';
import type { Problem, StateSummary, RespondResponse } from '@/api';
import { Brain, Play, Loader2 } from 'lucide-react';

type Phase = 'login' | 'problem' | 'confidence' | 'feedback';

export default function App() {
  const [phase, setPhase] = useState<Phase>('login');
  const [studentId, setStudentId] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [problem, setProblem] = useState<Problem | null>(null);
  const [actionReason, setActionReason] = useState('');
  const [stateSummary, setStateSummary] = useState<StateSummary | null>(null);
  const [lastResponse, setLastResponse] = useState<RespondResponse | null>(null);
  const [loading, setLoading] = useState(false);

  // Track response for confidence step
  const pendingAnswer = useRef('');
  const pendingExplanation = useRef('');
  const problemStartTime = useRef(Date.now());

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
      toast.success('Session started');
    } catch (err) {
      toast.error('Failed to start session. Is the server running?');
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
        sessionId,
        problem.problem_id,
        pendingAnswer.current,
        pendingExplanation.current,
        responseTimeMs,
        confidence,
      );
      setLastResponse(res);
      setStateSummary(res.state_summary);
      setActionReason(res.action_reason);
      setPhase('feedback');
    } catch (err) {
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

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Toaster
        theme="dark"
        position="top-center"
        toastOptions={{
          style: {
            background: 'var(--card)',
            border: '1px solid var(--border)',
            color: 'var(--foreground)',
          },
        }}
      />

      {/* Header */}
      <header className="border-b border-border/30 bg-card/40 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 h-12 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-primary" />
            <span className="font-semibold text-sm tracking-tight">Adaptive Learning</span>
          </div>
          {sessionId && stateSummary && (
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span>{stateSummary.problems_completed} problems</span>
              <Separator orientation="vertical" className="h-4" />
              <span>{Math.round(stateSummary.session_accuracy * 100)}% accuracy</span>
            </div>
          )}
        </div>
      </header>

      {/* Login */}
      {phase === 'login' && (
        <div className="flex items-center justify-center min-h-[calc(100vh-48px)]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-sm space-y-6 px-4"
          >
            <div className="text-center space-y-2">
              <motion.div
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 200, damping: 20 }}
              >
                <Brain className="w-12 h-12 text-primary mx-auto" />
              </motion.div>
              <h1 className="text-xl font-semibold">Adaptive Learning</h1>
              <p className="text-sm text-muted-foreground">
                Algebra practice powered by active inference
              </p>
            </div>

            <div className="space-y-3">
              <Input
                value={studentId}
                onChange={(e) => setStudentId(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleStart()}
                placeholder="Student ID (optional)"
                className="bg-card/50 border-border/50"
              />
              <Button
                onClick={handleStart}
                disabled={loading}
                className="w-full"
                size="lg"
              >
                {loading ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Play className="w-4 h-4 mr-2" />
                )}
                Start Session
              </Button>
            </div>
          </motion.div>
        </div>
      )}

      {/* Main session layout */}
      {phase !== 'login' && (
        <div className="max-w-5xl mx-auto px-4 py-6">
          <div className="flex gap-6">
            {/* Problem area */}
            <div className="flex-1 min-w-0 space-y-4">
              <AnimatePresence mode="wait">
                {phase === 'problem' && problem && (
                  <motion.div
                    key={`problem-${problem.problem_id}`}
                    initial={{ opacity: 0, x: 30 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -30 }}
                    transition={{ duration: 0.25 }}
                  >
                    <ProblemCard
                      problem={problem}
                      onSubmit={handleSubmitAnswer}
                      disabled={loading}
                    />
                  </motion.div>
                )}

                {phase === 'confidence' && (
                  <motion.div
                    key="confidence"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    <ConfidencePrompt onSelect={handleConfidence} />
                  </motion.div>
                )}

                {phase === 'feedback' && lastResponse && (
                  <motion.div
                    key="feedback"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
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
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 text-primary animate-spin" />
                </div>
              )}
            </div>

            {/* Sidebar dashboard */}
            <aside className="w-72 shrink-0 hidden lg:block">
              <div className="sticky top-16">
                <SessionDashboard state={stateSummary} actionReason={actionReason} />
              </div>
            </aside>
          </div>
        </div>
      )}
    </div>
  );
}
