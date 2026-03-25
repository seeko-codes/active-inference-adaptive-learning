import type { StateSummary } from '@/api';

interface Props {
  state: StateSummary | null;
  actionReason: string;
  studentId?: string;
}

const affectColor: Record<string, string> = {
  frustrated: '#f87171',
  engaged: '#4a7c6f',
  bored: '#71717a',
};

const wmColor: Record<string, string> = {
  low: '#4a7c6f',
  moderate: '#c4a262',
  high: '#f87171',
};

export function SessionDashboard({ state, actionReason, studentId }: Props) {
  if (!state) return null;

  const masteryData = Object.entries(state.mastery_estimates)
    .map(([skill, value]) => ({
      name: skill.replace(/-/g, ' '),
      short: skill.split('-').map((w) => w[0]).join('').toUpperCase(),
      full: skill.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
      value: Math.round((value as number) * 100),
    }))
    .sort((a, b) => b.value - a.value);

  const recentAcc = Math.round(state.recent_accuracy * 100);
  const sessionAcc = Math.round(state.session_accuracy * 100);
  const uncertainty = Math.round(state.uncertainty * 100);

  return (
    <div className="bg-[#111116] border border-[#27272a] rounded-lg overflow-hidden">
      {/* Panel header */}
      <div className="px-4 pt-4 pb-3 border-b border-[#1c1c24]">
        <p className="text-[10px] font-semibold tracking-[0.16em] uppercase text-[#4a7c6f]">
          Learner Model
        </p>
        {studentId && (
          <p className="font-mono text-[10px] text-[#3f3f46] mt-0.5 truncate">{studentId}</p>
        )}
      </div>

      <div className="px-4 py-3 space-y-4">
        {/* Key metrics */}
        <div className="grid grid-cols-2 gap-3">
          <Metric label="Recent acc" value={`${recentAcc}%`} />
          <Metric label="Observations" value={String(state.problems_completed)} />
        </div>

        <Divider />

        {/* Belief state */}
        <Section label="Belief State">
          <BeliefRow label="Affect" value={state.affect} color={affectColor[state.affect] || '#71717a'} />
          <BeliefRow label="Working mem" value={state.wm_load} color={wmColor[state.wm_load] || '#71717a'} />
          <BeliefRow
            label="Uncertainty"
            value={`${uncertainty}%`}
            color={uncertainty > 60 ? '#f87171' : uncertainty > 35 ? '#c4a262' : '#4a7c6f'}
          />
          <BeliefRow
            label="Session acc"
            value={`${sessionAcc}%`}
            color={sessionAcc >= 70 ? '#4a7c6f' : sessionAcc >= 45 ? '#c4a262' : '#f87171'}
          />
        </Section>

        <Divider />

        {/* Policy rationale */}
        {actionReason && (
          <>
            <Section label="Policy Rationale">
              <p className="text-[11px] text-[#52525b] leading-[1.65]">{actionReason}</p>
            </Section>
            <Divider />
          </>
        )}

        {/* Mastery estimates */}
        <Section label="Mastery Estimates">
          <div className="space-y-2.5">
            {masteryData.map(({ short, full, value }) => (
              <div key={short} title={full}>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono text-[10px] text-[#3f3f46]">{short}</span>
                  <span className="font-mono text-[10px] tabular-nums text-[#3f3f46]">
                    {value}%
                  </span>
                </div>
                <div className="h-[2px] bg-[#1c1c24] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700 ease-out"
                    style={{
                      width: `${value}%`,
                        backgroundColor:
                        value >= 70 ? '#4a7c6f' : value >= 40 ? '#c4a262' : '#3f3f46',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Section>

        {/* Skill beliefs summary */}
        {Object.keys(state.beliefs).length > 0 && (
          <>
            <Divider />
            <Section label="Skill Beliefs">
              <div className="space-y-1">
                {Object.entries(state.beliefs).slice(0, 4).map(([skill, state_]) => (
                  <div key={skill} className="flex items-center justify-between">
                    <span className="font-mono text-[10px] text-[#3f3f46] truncate max-w-[120px]">
                      {skill.replace(/-/g, '_')}
                    </span>
                    <span className="font-mono text-[10px] text-[#52525b] shrink-0 ml-2">
                      {state_ as string}
                    </span>
                  </div>
                ))}
              </div>
            </Section>
          </>
        )}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[10px] text-[#3f3f46] tracking-[0.06em] mb-0.5">{label}</p>
      <p className="font-mono text-[18px] font-semibold text-[#f0f0f4] tabular-nums leading-none">
        {value}
      </p>
    </div>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <p className="text-[10px] font-semibold tracking-[0.14em] uppercase text-[#52525b]">
        {label}
      </p>
      {children}
    </div>
  );
}

function BeliefRow({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[11px] text-[#52525b]">{label}</span>
      <div className="flex items-center gap-1.5">
        <div className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
        <span className="font-mono text-[11px] text-[#a0a0ab] capitalize tabular-nums">
          {value}
        </span>
      </div>
    </div>
  );
}

function Divider() {
  return <div className="h-px bg-[#1c1c24]" />;
}
