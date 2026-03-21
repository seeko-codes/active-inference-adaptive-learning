import { motion } from 'motion/react';

interface Props {
  onSelect: (confidence: number) => void;
}

const levels = [
  { value: 1, label: 'Guessing' },
  { value: 2, label: 'Unsure' },
  { value: 3, label: 'Somewhat' },
  { value: 4, label: 'Confident' },
  { value: 5, label: 'Certain' },
];

export function ConfidencePrompt({ onSelect }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="rounded-xl border border-border/50 bg-card/80 backdrop-blur-sm p-5 space-y-3"
    >
      <p className="text-sm text-muted-foreground text-center">
        How confident are you in your answer?
      </p>
      <div className="flex gap-2 justify-center">
        {levels.map(({ value, label }) => (
          <motion.button
            key={value}
            onClick={() => onSelect(value)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex flex-col items-center gap-1 px-3 py-2 rounded-lg border border-border/50
                       hover:border-primary/40 hover:bg-primary/5 transition-colors"
          >
            <span className="text-lg font-semibold text-foreground">{value}</span>
            <span className="text-[10px] text-muted-foreground">{label}</span>
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}
