import { useState, useRef, useEffect } from 'react';
import { motion } from 'motion/react';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { MathDisplay } from './MathDisplay';
import type { Problem } from '@/api';
import { Lightbulb } from 'lucide-react';

interface Props {
  problem: Problem;
  actionReason?: string;
  onSubmit: (answer: string, explanation: string) => void;
  disabled?: boolean;
}

const actionLabel: Record<string, string> = {
  worked_example: 'Study',
  faded_example: 'Guided',
  space_and_test: 'Retrieval',
  reteach: 'Review',
  interleave: 'Mixed',
  increase_challenge: 'Challenge',
  reduce_load: 'Scaffold',
  diagnostic_probe: 'Diagnostic',
  boundary_test: 'Boundary',
  strategic_compute: 'Compute',
  inverse_rewrite: 'Rewrite',
  order_of_ops: 'Order of Ops',
};

export function ProblemCard({ problem, actionReason, onSubmit, disabled }: Props) {
  const [answer, setAnswer] = useState('');
  const [explanation, setExplanation] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setAnswer('');
    setExplanation('');
    // preventScroll: true — never let focus cause the page to jump
    inputRef.current?.focus({ preventScroll: true });
  }, [problem.problem_id]);

  const wordCount = explanation.trim().split(/\s+/).filter(Boolean).length;
  const canSubmit = answer.trim().length > 0 && wordCount >= 3 && !disabled;

  const handleSubmit = () => {
    if (canSubmit) onSubmit(answer.trim(), explanation.trim());
  };

  return (
    <div className="bg-[#111116] border border-[#27272a] rounded-lg overflow-hidden">
      {/* Policy context header */}
      <div className="px-6 pt-5 pb-4 border-b border-[#1c1c24]">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[10px] font-semibold tracking-[0.14em] uppercase text-[#4a7c6f]">
                {actionLabel[problem.action] || problem.action}
              </span>
              <span className="text-[#27272a] text-[10px]">·</span>
              <span className="font-mono text-[10px] text-[#52525b] tracking-wide">
                {problem.skill}
              </span>
              <span className="text-[#27272a] text-[10px]">·</span>
              <span className="font-mono text-[10px] text-[#3f3f46]">
                tier {problem.tier}
              </span>
            </div>
            {actionReason && (
              <p className="text-[11px] text-[#3f3f46] leading-relaxed line-clamp-2">
                {actionReason}
              </p>
            )}
          </div>
        </div>
      </div>

      <div className="px-6 pb-6 pt-5 space-y-5">
        {/* Worked example display */}
        {problem.question_type === 'worked_example' && problem.example_before && (
          <div className="bg-[#0d0d11] rounded-md p-4 space-y-2.5 border border-[#1c1c24]">
            {problem.rule_demonstrated && (
              <p className="text-[10px] font-semibold text-[#4a7c6f] uppercase tracking-[0.14em]">
                {problem.rule_demonstrated}
              </p>
            )}
            <div className="flex items-center gap-5 text-base text-[#f0f0f4]">
              <MathDisplay expr={problem.example_before} display />
              <span className="text-[#52525b] text-sm">=</span>
              <MathDisplay expr={problem.example_after!} display />
            </div>
            {problem.rule_description && (
              <p className="text-[12px] text-[#71717a] leading-relaxed">
                {problem.rule_description}
              </p>
            )}
          </div>
        )}

        {/* Identify property — before → after */}
        {problem.question_type === 'identify_property' && (
          <div className="flex items-center gap-6 justify-center py-6 text-lg text-[#f0f0f4]">
            <MathDisplay expr={problem.expression_before!} display />
            <span className="text-[#3f3f46] text-xs tracking-widest uppercase">to</span>
            <MathDisplay expr={problem.expression_after!} display />
          </div>
        )}

        {/* Comparison types */}
        {(problem.question_type === 'equivalent' ||
          problem.question_type === 'non_equivalent' ||
          problem.question_type === 'boundary_test' ||
          problem.question_type === 'custom_operation') &&
          problem.expression_a && (
            <div className="flex items-center gap-7 justify-center py-6 text-lg text-[#f0f0f4]">
              <MathDisplay expr={problem.expression_a!} display />
              <span className="text-[10px] font-mono tracking-[0.18em] text-[#3f3f46] uppercase">vs</span>
              <MathDisplay expr={problem.expression_b!} display />
            </div>
          )}

        {/* Main expression */}
        {problem.student_sees &&
          problem.question_type !== 'identify_property' &&
          problem.question_type !== 'equivalent' &&
          problem.question_type !== 'non_equivalent' &&
          problem.question_type !== 'boundary_test' &&
          problem.question_type !== 'custom_operation' && (
            <div className="text-center py-7">
              <div className="text-[2rem] leading-tight text-[#f0f0f4]">
                <MathDisplay expr={problem.student_sees} display />
              </div>
            </div>
          )}

        {/* Prompt */}
        <p className="text-[13px] text-[#a0a0ab] leading-[1.7]">{problem.prompt}</p>

        {/* Scaffold hint */}
        {problem.scaffolding?.hint && (
          <div className="flex items-start gap-2.5 bg-[#0d0d11] rounded-md px-3.5 py-2.5 border border-[#1c1c24]">
            <Lightbulb className="w-3 h-3 text-[#52525b] mt-0.5 shrink-0" strokeWidth={1.5} />
            <p className="text-[11px] text-[#52525b] leading-relaxed">{problem.scaffolding.hint}</p>
          </div>
        )}

        {/* Answer controls */}
        {problem.question_type === 'identify_property' && problem.choices ? (
          <div className="grid grid-cols-2 gap-1.5">
            {problem.choices.map((choice) => (
              <button
                key={choice}
                onClick={() => setAnswer(choice)}
                disabled={disabled}
                className={`px-3 py-2 rounded-md text-[12px] border transition-all duration-120 cursor-pointer
                  ${answer === choice
                    ? 'border-[#4a7c6f]/40 bg-[#4a7c6f]/8 text-[#4a7c6f] font-medium'
                    : 'border-[#27272a] text-[#71717a] hover:border-[#3f3f46] hover:text-[#a0a0ab]'
                  }`}
              >
                {choice}
              </button>
            ))}
          </div>
        ) : (
          problem.question_type === 'equivalent' ||
          problem.question_type === 'non_equivalent' ||
          problem.question_type === 'boundary_test' ||
          problem.question_type === 'custom_operation' ||
          problem.question_type === 'proof_disproof'
        ) ? (
          <div className="flex gap-2 justify-center">
            {['Yes', 'No'].map((opt) => (
              <button
                key={opt}
                onClick={() => setAnswer(opt === 'Yes' ? 'true' : 'false')}
                disabled={disabled}
                className={`w-20 py-2.5 rounded-md text-[13px] font-medium border transition-all duration-120 cursor-pointer
                  ${(answer === 'true' && opt === 'Yes') || (answer === 'false' && opt === 'No')
                    ? 'border-[#4a7c6f]/40 bg-[#4a7c6f]/8 text-[#4a7c6f]'
                    : 'border-[#27272a] text-[#71717a] hover:border-[#3f3f46] hover:text-[#a0a0ab]'
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
            placeholder="Answer"
            disabled={disabled}
            className="h-10 bg-[#0d0d11] border-[#27272a] hover:border-[#3f3f46] focus:border-[#52525b] focus:ring-0 rounded-md px-3 text-[13px] text-[#f0f0f4] placeholder:text-[#3f3f46] transition-colors"
          />
        )}

        {/* Reasoning trace */}
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label className="text-[10px] font-medium text-[#3f3f46] tracking-[0.08em] uppercase">
              Reasoning trace
            </label>
            <span className={`text-[10px] font-mono tabular-nums transition-colors ${wordCount >= 3 ? 'text-[#52525b]' : 'text-[#3f3f46]'}`}>
              {wordCount} / 3 min
            </span>
          </div>
          <Textarea
            value={explanation}
            onChange={(e) => setExplanation(e.target.value)}
            placeholder="Describe your reasoning…"
            disabled={disabled}
            rows={2}
            className="bg-[#0d0d11] border-[#27272a] hover:border-[#3f3f46] focus:border-[#52525b] focus:ring-0 rounded-md resize-none text-[12px] px-3 py-2.5 text-[#f0f0f4] placeholder:text-[#3f3f46] leading-relaxed transition-colors"
          />
        </div>

        {/* Submit */}
        <motion.button
          onClick={handleSubmit}
          disabled={!canSubmit}
          whileHover={canSubmit ? { opacity: 0.88 } : {}}
          whileTap={canSubmit ? { scale: 0.985 } : {}}
          className={`w-full h-10 rounded-md text-[13px] font-semibold
                     transition-all duration-150 cursor-pointer
                     ${canSubmit
                       ? 'bg-[#4a7c6f] text-white'
                       : 'bg-[#1c1c24] text-[#3f3f46] cursor-not-allowed'
                     }`}
        >
          Submit Response
        </motion.button>
      </div>
    </div>
  );
}
