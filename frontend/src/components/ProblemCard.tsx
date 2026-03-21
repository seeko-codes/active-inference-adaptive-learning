import { useState, useRef, useEffect } from 'react';
import { motion } from 'motion/react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { MathDisplay } from './MathDisplay';
import type { Problem } from '@/api';
import { Send, Lightbulb, BookOpen } from 'lucide-react';

interface Props {
  problem: Problem;
  onSubmit: (answer: string, explanation: string) => void;
  disabled?: boolean;
}

export function ProblemCard({ problem, onSubmit, disabled }: Props) {
  const [answer, setAnswer] = useState('');
  const [explanation, setExplanation] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setAnswer('');
    setExplanation('');
    inputRef.current?.focus();
  }, [problem.problem_id]);

  const wordCount = explanation.trim().split(/\s+/).filter(Boolean).length;
  const canSubmit = answer.trim().length > 0 && wordCount >= 3 && !disabled;

  const handleSubmit = () => {
    if (canSubmit) onSubmit(answer.trim(), explanation.trim());
  };

  const actionLabel: Record<string, string> = {
    worked_example: 'Study Mode',
    faded_example: 'Guided Practice',
    space_and_test: 'Retrieval Test',
    reteach: 'Review',
    interleave: 'Mixed Practice',
    increase_challenge: 'Challenge',
    reduce_load: 'Simplified',
    diagnostic_probe: 'Diagnostic',
    boundary_test: 'Boundary Test',
    strategic_compute: 'Strategic',
    inverse_rewrite: 'PEMA Rewrite',
    order_of_ops: 'Order of Ops',
  };

  const actionIcon: Record<string, typeof BookOpen> = {
    worked_example: BookOpen,
    diagnostic_probe: Lightbulb,
  };
  const Icon = actionIcon[problem.action] || Lightbulb;

  return (
    <Card className="border-border/50 bg-card/80 backdrop-blur-sm">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-primary border-primary/30 text-xs">
              <Icon className="w-3 h-3 mr-1" />
              {actionLabel[problem.action] || problem.action}
            </Badge>
            <Badge variant="secondary" className="text-xs">
              {problem.skill}
            </Badge>
          </div>
          <span className="text-xs text-muted-foreground">Tier {problem.tier}</span>
        </div>
      </CardHeader>

      <CardContent className="space-y-5">
        {/* Worked example display */}
        {problem.question_type === 'worked_example' && problem.example_before && (
          <div className="rounded-lg bg-primary/5 border border-primary/10 p-4 space-y-2">
            <p className="text-xs font-medium text-primary uppercase tracking-wider">
              {problem.rule_demonstrated}
            </p>
            <div className="flex items-center gap-3 text-lg">
              <MathDisplay expr={problem.example_before} display />
              <span className="text-muted-foreground">=</span>
              <MathDisplay expr={problem.example_after!} display />
            </div>
            {problem.rule_description && (
              <p className="text-sm text-muted-foreground">{problem.rule_description}</p>
            )}
          </div>
        )}

        {/* Identify property: show before/after */}
        {problem.question_type === 'identify_property' && (
          <div className="space-y-3">
            <div className="flex items-center gap-4 justify-center text-xl py-2">
              <MathDisplay expr={problem.expression_before!} display />
              <span className="text-primary font-bold">&rarr;</span>
              <MathDisplay expr={problem.expression_after!} display />
            </div>
          </div>
        )}

        {/* Equivalent / non-equivalent / custom operation */}
        {(problem.question_type === 'equivalent' || problem.question_type === 'non_equivalent' || problem.question_type === 'boundary_test' || problem.question_type === 'custom_operation') && problem.expression_a && (
          <div className="flex items-center gap-4 justify-center text-xl py-2">
            <div className="text-center">
              <MathDisplay expr={problem.expression_a!} display />
            </div>
            <span className="text-muted-foreground text-sm">vs</span>
            <div className="text-center">
              <MathDisplay expr={problem.expression_b!} display />
            </div>
          </div>
        )}

        {/* Main expression */}
        {problem.student_sees && problem.question_type !== 'identify_property' &&
         problem.question_type !== 'equivalent' && problem.question_type !== 'non_equivalent' &&
         problem.question_type !== 'boundary_test' && problem.question_type !== 'custom_operation' && (
          <div className="text-center py-4">
            <div className="text-2xl">
              <MathDisplay expr={problem.student_sees} display />
            </div>
          </div>
        )}

        {/* Prompt */}
        <p className="text-sm text-foreground/80">{problem.prompt}</p>

        {/* Scaffolding hint */}
        {problem.scaffolding?.hint && (
          <div className="rounded-md bg-primary/5 border border-primary/10 px-3 py-2 text-sm text-primary/80">
            <Lightbulb className="w-3.5 h-3.5 inline mr-1.5 -mt-0.5" />
            {problem.scaffolding.hint}
          </div>
        )}

        {/* Answer input */}
        {problem.question_type === 'identify_property' && problem.choices ? (
          <div className="grid grid-cols-2 gap-2">
            {problem.choices.map((choice) => (
              <button
                key={choice}
                onClick={() => setAnswer(choice)}
                disabled={disabled}
                className={`px-3 py-2 rounded-md text-sm border transition-all
                  ${answer === choice
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border hover:border-primary/30 text-foreground/70'
                  }`}
              >
                {choice}
              </button>
            ))}
          </div>
        ) : (problem.question_type === 'equivalent' || problem.question_type === 'non_equivalent' || problem.question_type === 'boundary_test' || problem.question_type === 'custom_operation' || problem.question_type === 'proof_disproof') ? (
          <div className="flex gap-3 justify-center">
            {['Yes', 'No'].map((opt) => (
              <button
                key={opt}
                onClick={() => setAnswer(opt === 'Yes' ? 'true' : 'false')}
                disabled={disabled}
                className={`px-6 py-2 rounded-md text-sm border transition-all
                  ${(answer === 'true' && opt === 'Yes') || (answer === 'false' && opt === 'No')
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border hover:border-primary/30 text-foreground/70'
                  }`}
              >
                {opt}
              </button>
            ))}
          </div>
        ) : (
          <Input
            ref={inputRef}
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && canSubmit && handleSubmit()}
            placeholder="Your answer..."
            disabled={disabled}
            className="bg-background/50 border-border/50 focus:border-primary/50"
          />
        )}

        {/* Explanation */}
        <div className="space-y-1.5">
          <Textarea
            value={explanation}
            onChange={(e) => setExplanation(e.target.value)}
            placeholder="Explain your reasoning..."
            disabled={disabled}
            rows={2}
            className="bg-background/50 border-border/50 focus:border-primary/50 resize-none text-sm"
          />
          <div className="flex justify-between items-center">
            <span className={`text-xs ${wordCount < 3 ? 'text-destructive/70' : 'text-muted-foreground'}`}>
              {wordCount} word{wordCount !== 1 ? 's' : ''} {wordCount < 3 && '(min 3)'}
            </span>
          </div>
        </div>

        {/* Submit */}
        <motion.div whileTap={{ scale: 0.98 }}>
          <Button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="w-full"
            size="lg"
          >
            <Send className="w-4 h-4 mr-2" />
            Submit
          </Button>
        </motion.div>
      </CardContent>
    </Card>
  );
}
