import { motion, AnimatePresence } from 'motion/react';
import { MathDisplay } from './MathDisplay';
import { CheckCircle2, XCircle } from 'lucide-react';

interface Props {
  correct: boolean;
  expectedAnswer: string;
  feedback: string;
  onContinue: () => void;
}

export function FeedbackView({ correct, expectedAnswer, feedback, onContinue }: Props) {
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
        className={`rounded-xl border p-5 space-y-3 ${
          correct
            ? 'border-emerald-500/30 bg-emerald-500/5'
            : 'border-destructive/30 bg-destructive/5'
        }`}
      >
        <div className="flex items-center gap-3">
          {correct ? (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring', stiffness: 400, damping: 15 }}
            >
              <CheckCircle2 className="w-6 h-6 text-emerald-500" />
            </motion.div>
          ) : (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring', stiffness: 400, damping: 15 }}
            >
              <XCircle className="w-6 h-6 text-destructive" />
            </motion.div>
          )}
          <span className={`font-semibold ${correct ? 'text-emerald-500' : 'text-destructive'}`}>
            {correct ? 'Correct!' : 'Not quite'}
          </span>
        </div>

        {!correct && expectedAnswer && (
          <div className="pl-9 space-y-1">
            <p className="text-sm text-muted-foreground">Expected answer:</p>
            <div className="text-lg">
              <MathDisplay expr={expectedAnswer} display />
            </div>
          </div>
        )}

        {feedback && feedback !== 'Correct!' && (
          <p className="pl-9 text-sm text-muted-foreground">{feedback}</p>
        )}

        <motion.button
          onClick={onContinue}
          whileTap={{ scale: 0.97 }}
          className="ml-9 mt-2 px-4 py-1.5 text-sm rounded-md bg-primary/10 text-primary
                     hover:bg-primary/20 transition-colors"
        >
          Continue
        </motion.button>
      </motion.div>
    </AnimatePresence>
  );
}
