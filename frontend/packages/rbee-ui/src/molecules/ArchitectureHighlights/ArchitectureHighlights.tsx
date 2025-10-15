export interface ArchitectureHighlight {
  title: string
  details: string[]
}

export interface ArchitectureHighlightsProps {
  /**
   * Array of architecture highlights to display
   */
  highlights: ArchitectureHighlight[]
  /**
   * Optional className for the container
   */
  className?: string
}

export function ArchitectureHighlights({ highlights, className = '' }: ArchitectureHighlightsProps) {
  return (
    <div className={className}>
      <div className="text-xs tracking-wide uppercase text-muted-foreground mb-3">Core Principles</div>
      <h3 className="text-2xl font-bold text-foreground mb-4">Architecture Highlights</h3>
      <ul className="space-y-3">
        {highlights.map((highlight, index) => (
          <li key={index}>
            <div className="text-sm font-medium text-foreground">{highlight.title}</div>
            {highlight.details.map((detail, detailIndex) => (
              <div key={detailIndex} className="text-xs text-muted-foreground">
                {detail}
              </div>
            ))}
          </li>
        ))}
      </ul>
    </div>
  )
}
