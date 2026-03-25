import { useState } from 'react';
import { motion } from 'motion/react';

interface Props {
  onSelect: (confidence: number) => void;
}

const levels = [
  { value: 1, label: 'Guessing' },
  { value: 2, label: 'Unsure' },
  { value: 3, label: 'Moderate' },
  { value: 4, label: 'Confident' },
  { value: 5, label: 'Certain' },
] as const;

export function ConfidencePrompt({ onSelect }: Props) {
  const [hovered, setHovered] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);

  const handleClick = (value: number) => {
    setSelected(value);
    setTimeout(() => onSelect(value), 280);
  };

  const activeUpTo = selected ?? hovered;

  return (
    <div className="bg-[#111116] border border-[#27272a] rounded-lg px-6 py-6">
      {/* Header */}
      <div className="mb-5">
        <p className="text-[13px] font-semibold text-[#f0f0f4] tracking-tight mb-1">
          Certainty Estimate
        </p>
        <p className="text-[11px] text-[#3f3f46] leading-relaxed">
          Self-reported certainty is recorded as a learner signal.
        </p>
      </div>

      {/* Scale */}
      <div className="flex items-center gap-1.5 mb-3">
        {levels.map(({ value }, i) => {
          const isActive = activeUpTo !== null && value <= activeUpTo;
          const isSelected = selected === value;

          return (
            <motion.button
              key={value}
              onMouseEnter={() => setHovered(value)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => handleClick(value)}
              initial={{ opacity: 0, y: 4 }}
              animate={{
                opacity: 1,
                y: 0,
                scale: isSelected ? 1.08 : 1,
              }}
              transition={{
                delay: i * 0.04,
                duration: 0.2,
                scale: { type: 'spring', stiffness: 400, damping: 20 },
              }}
              className={`flex-1 h-9 rounded-md border text-[12px] font-mono tabular-nums
                         transition-all duration-100 cursor-pointer select-none
                         ${isActive
                           ? 'border-[#4a7c6f]/40 bg-[#4a7c6f]/8 text-[#4a7c6f]'
                           : 'border-[#27272a] text-[#3f3f46] hover:border-[#3f3f46] hover:text-[#52525b]'
                         }`}
            >
              {value}
            </motion.button>
          );
        })}
      </div>

      {/* Axis labels */}
      <div className="flex justify-between px-0.5">
        <span className="text-[10px] text-[#3f3f46] tracking-[0.08em]">Guessing</span>
        {activeUpTo !== null && (
          <span className="text-[10px] text-[#52525b]">
            {levels[activeUpTo - 1]?.label}
          </span>
        )}
        <span className="text-[10px] text-[#3f3f46] tracking-[0.08em]">Certain</span>
      </div>
    </div>
  );
}
