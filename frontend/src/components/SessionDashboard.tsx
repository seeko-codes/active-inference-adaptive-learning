import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import type { StateSummary } from '@/api';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import { Brain, Activity, Gauge, Target } from 'lucide-react';

interface Props {
  state: StateSummary | null;
  actionReason: string;
}

const affectColors: Record<string, string> = {
  frustrated: 'text-destructive',
  engaged: 'text-emerald-500',
  bored: 'text-yellow-500',
};

const wmColors: Record<string, string> = {
  low: 'text-emerald-500',
  moderate: 'text-yellow-500',
  high: 'text-destructive',
};

export function SessionDashboard({ state, actionReason }: Props) {
  if (!state) return null;

  const masteryData = Object.entries(state.mastery_estimates)
    .map(([skill, value]) => ({
      name: skill.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      short: skill.split('-').map(w => w[0]).join('').toUpperCase(),
      value: Math.round((value as number) * 100),
    }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className="space-y-4">
      {/* Stats row */}
      <div className="grid grid-cols-2 gap-3">
        <Card className="border-border/30 bg-card/60">
          <CardContent className="p-3 flex items-center gap-2">
            <Target className="w-4 h-4 text-primary" />
            <div>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Accuracy</p>
              <p className="text-lg font-semibold">
                {Math.round(state.recent_accuracy * 100)}%
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/30 bg-card/60">
          <CardContent className="p-3 flex items-center gap-2">
            <Activity className="w-4 h-4 text-primary" />
            <div>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Problems</p>
              <p className="text-lg font-semibold">{state.problems_completed}</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Cognitive state */}
      <Card className="border-border/30 bg-card/60">
        <CardHeader className="pb-2 pt-3 px-3">
          <div className="flex items-center gap-1.5">
            <Brain className="w-3.5 h-3.5 text-primary" />
            <span className="text-xs font-medium">Cognitive State</span>
          </div>
        </CardHeader>
        <CardContent className="px-3 pb-3 space-y-2">
          <div className="flex justify-between items-center text-sm">
            <span className="text-muted-foreground">Affect</span>
            <Badge variant="outline" className={`text-xs ${affectColors[state.affect] || ''}`}>
              {state.affect}
            </Badge>
          </div>
          <div className="flex justify-between items-center text-sm">
            <span className="text-muted-foreground">WM Load</span>
            <Badge variant="outline" className={`text-xs ${wmColors[state.wm_load] || ''}`}>
              {state.wm_load}
            </Badge>
          </div>
          <div className="flex justify-between items-center text-sm">
            <span className="text-muted-foreground">Uncertainty</span>
            <span className="text-xs font-mono text-foreground/70">
              {Math.round(state.uncertainty * 100)}%
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Action reason */}
      {actionReason && (
        <Card className="border-border/30 bg-card/60">
          <CardContent className="p-3">
            <div className="flex items-start gap-2">
              <Gauge className="w-3.5 h-3.5 text-primary mt-0.5 shrink-0" />
              <p className="text-xs text-muted-foreground leading-relaxed">{actionReason}</p>
            </div>
          </CardContent>
        </Card>
      )}

      <Separator className="bg-border/30" />

      {/* Skill mastery chart */}
      <Card className="border-border/30 bg-card/60">
        <CardHeader className="pb-1 pt-3 px-3">
          <span className="text-xs font-medium">Skill Mastery</span>
        </CardHeader>
        <CardContent className="px-1 pb-2">
          <ResponsiveContainer width="100%" height={masteryData.length * 28 + 10}>
            <BarChart
              data={masteryData}
              layout="vertical"
              margin={{ left: 4, right: 12, top: 4, bottom: 4 }}
            >
              <XAxis type="number" domain={[0, 100]} hide />
              <YAxis
                type="category"
                dataKey="short"
                width={36}
                tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--card)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
                formatter={(value, _name, entry) => [`${value}%`, (entry as { payload: { name: string } }).payload.name]}
                cursor={false}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={14}>
                {masteryData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={
                      entry.value >= 70
                        ? 'oklch(0.7 0.18 160)'   // green
                        : entry.value >= 40
                          ? 'oklch(0.75 0.15 80)'  // yellow
                          : 'oklch(0.65 0.2 265)'  // indigo
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Session accuracy */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-xs">
          <span className="text-muted-foreground">Session accuracy</span>
          <span className="font-mono text-foreground/70">
            {Math.round(state.session_accuracy * 100)}%
          </span>
        </div>
        <Progress value={state.session_accuracy * 100} className="h-1.5" />
      </div>
    </div>
  );
}
