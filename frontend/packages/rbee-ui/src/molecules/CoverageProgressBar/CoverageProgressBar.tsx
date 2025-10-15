export interface CoverageProgressBarProps {
  /**
   * Label for the progress bar
   * @default "BDD Coverage"
   */
  label?: string;
  /**
   * Number of passing scenarios
   */
  passing: number;
  /**
   * Total number of scenarios
   */
  total: number;
  /**
   * Optional className for the container
   */
  className?: string;
}

export function CoverageProgressBar({
  label = "BDD Coverage",
  passing,
  total,
  className = "",
}: CoverageProgressBarProps) {
  const percentage = Math.round((passing / total) * 100);

  return (
    <div className={className}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-foreground">{label}</span>
        <span className="text-xs text-muted-foreground">
          {passing}/{total} scenarios passing
        </span>
      </div>
      <div className="relative h-2 rounded bg-muted">
        <div
          className="absolute inset-y-0 left-0 bg-chart-3 rounded"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <p className="text-xs text-muted-foreground mt-1">{percentage}% complete</p>
    </div>
  );
}
