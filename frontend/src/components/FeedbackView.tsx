import { motion } from 'motion/react';
import { MathDisplay } from './MathDisplay';

interface Props {
  correct: boolean;
  expectedAnswer: string;
  feedback: string;
  onContinue: () => void;
}

export function FeedbackView({ correct, expectedAnswer, feedback, onContinue }: Props) {
  return (
    <div className="space-y-3">
      {/* Result panel */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
        className="bg-[#111116] border border-[#27272a] rounded-lg overflow-hidden"
      >
        {/* Status bar */}
        <div
          className={`px-6 py-4 border-b border-[#1c1c24] flex items-center gap-3 ${
            correct ? '' : ''
          }`}
        >
          <motion.div
            initial={{ scale: 0.6, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.08, duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            className={`w-1.5 h-1.5 rounded-full shrink-0 ${
              correct ? 'bg-[#4a7c6f]' : 'bg-[#f87171]'
            }`}
          />
          <div>
            <motion.p
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className={`text-[13px] font-semibold tracking-tight ${
                correct ? 'text-[#4a7c6f]' : 'text-[#f87171]'
              }`}
            >
              {correct ? 'Correct' : 'Incorrect'}
            </motion.p>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.18 }}
              className="text-[11px] text-[#3f3f46] mt-0.5"
            >
              {correct
                ? 'Response recorded · Learner model updated'
                : 'Observation logged · Updating belief state'
              }
            </motion.p>
          </div>
        </div>

        <div className="px-6 py-4 space-y-4">
          {/* Expected answer on incorrect */}
          {!correct && expectedAnswer && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="space-y-1.5"
            >
              <p className="text-[10px] font-medium text-[#3f3f46] tracking-[0.1em] uppercase">
                Expected answer
              </p>
              <div className="text-base text-[#f0f0f4]">
                <MathDisplay expr={expectedAnswer} display />
              </div>
            </motion.div>
          )}

          {/* Feedback text */}
          {feedback && feedback !== 'Correct!' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: correct ? 0.18 : 0.28 }}
            >
              {(!correct && expectedAnswer) && (
                <div className="h-px bg-[#1c1c24] mb-4" />
              )}
              <p className="text-[12px] text-[#71717a] leading-[1.7]">{feedback}</p>
            </motion.div>
          )}

          {/* Correct with no additional feedback */}
          {correct && (!feedback || feedback === 'Correct!') && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="text-[12px] text-[#52525b]"
            >
              Confidence and response time recorded as learner signals.
            </motion.p>
          )}
        </div>
      </motion.div>

      {/* Continue action */}
      <motion.button
        onClick={onContinue}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.32 }}
        whileHover={{ opacity: 0.88 }}
        whileTap={{ scale: 0.985 }}
        className="w-full h-10 rounded-md bg-[#4a7c6f] text-white text-[13px] font-semibold
                   cursor-pointer transition-opacity duration-150"
      >
        Continue Session
      </motion.button>
    </div>
  );
}
