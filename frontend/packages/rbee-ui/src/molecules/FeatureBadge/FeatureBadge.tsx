export interface FeatureBadgeProps {
  label: string;
}

/**
 * A small pill-shaped badge for displaying feature highlights or tags.
 * Commonly used in groups to show multiple feature attributes.
 */
export function FeatureBadge({ label }: FeatureBadgeProps) {
  return (
    <span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1 font-sans">
      {label}
    </span>
  );
}
