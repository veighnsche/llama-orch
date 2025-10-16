export interface MetricCardProps {
  label: string;
  value: string | number;
}

export function MetricCard({ label, value }: MetricCardProps) {
  return (
    <div className="rounded-lg border border-border bg-background/70 p-3">
      <div className="mb-0.5 text-xs uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div className="tabular-nums text-3xl font-bold text-foreground">
        {value}
      </div>
    </div>
  );
}
