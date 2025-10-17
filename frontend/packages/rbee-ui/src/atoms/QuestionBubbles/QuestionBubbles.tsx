import { cn } from '@rbee/ui/utils'

export interface QuestionBubblesProps {
  className?: string
}

/**
 * QuestionBubbles - Theme-aware SVG background for FAQ sections
 *
 * Question mark bubbles with connecting thought lines and lightbulb accents,
 * suggesting inquiry and discovery.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-25">
 *   <QuestionBubbles />
 * </div>
 * ```
 */
export function QuestionBubbles({ className }: QuestionBubblesProps) {
  return (
    <svg
      viewBox="0 0 1200 640"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('pointer-events-none', className)}
      aria-hidden="true"
      preserveAspectRatio="xMidYMid slice"
    >
      <defs>
        <filter id="bubble-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Question bubbles - light theme */}
      <g className="dark:hidden" opacity="0.4">
        <circle cx="300" cy="200" r="40" className="fill-blue-500" opacity="0.15" />
        <circle cx="600" cy="320" r="50" className="fill-blue-500" opacity="0.2" />
        <circle cx="900" cy="240" r="35" className="fill-blue-500" opacity="0.15" />
        <circle cx="450" cy="450" r="45" className="fill-blue-500" opacity="0.18" />
        <circle cx="750" cy="480" r="38" className="fill-blue-500" opacity="0.16" />
      </g>

      {/* Question bubbles - dark theme */}
      <g className="hidden dark:block" opacity="0.5">
        <circle cx="300" cy="200" r="40" className="fill-blue-400" opacity="0.2" />
        <circle cx="600" cy="320" r="50" className="fill-blue-400" opacity="0.25" />
        <circle cx="900" cy="240" r="35" className="fill-blue-400" opacity="0.2" />
        <circle cx="450" cy="450" r="45" className="fill-blue-400" opacity="0.22" />
        <circle cx="750" cy="480" r="38" className="fill-blue-400" opacity="0.21" />
      </g>

      {/* Thought connection lines */}
      <g className="stroke-blue-500 dark:stroke-blue-400" strokeWidth="1" strokeDasharray="3 6" opacity="0.3">
        <path d="M 340 200 Q 450 250 560 320" fill="none" />
        <path d="M 650 320 Q 750 350 750 440" fill="none" />
        <path d="M 860 240 Q 750 300 650 320" fill="none" />
        <path d="M 300 240 Q 350 350 450 410" fill="none" />
      </g>

      {/* Question marks */}
      <g className="fill-blue-500 dark:fill-blue-400" opacity="0.7" filter="url(#bubble-glow)">
        <text x="300" y="215" textAnchor="middle" fontSize="32" fontWeight="700">?</text>
        <text x="600" y="338" textAnchor="middle" fontSize="40" fontWeight="700">?</text>
        <text x="900" y="253" textAnchor="middle" fontSize="28" fontWeight="700">?</text>
        <text x="450" y="468" textAnchor="middle" fontSize="36" fontWeight="700">?</text>
        <text x="750" y="496" textAnchor="middle" fontSize="30" fontWeight="700">?</text>
      </g>

      {/* Lightbulb accents (answers/insights) */}
      <g className="fill-amber-500 dark:fill-amber-400" opacity="0.5" filter="url(#bubble-glow)">
        {/* Lightbulb 1 */}
        <circle cx="500" cy="180" r="8" />
        <rect x="496" y="188" width="8" height="6" rx="1" />
        
        {/* Lightbulb 2 */}
        <circle cx="820" cy="380" r="8" />
        <rect x="816" y="388" width="8" height="6" rx="1" />
      </g>

      {/* Small thought dots */}
      <g className="fill-blue-500 dark:fill-blue-400" opacity="0.4">
        <circle cx="380" cy="230" r="3" />
        <circle cx="420" cy="260" r="2" />
        <circle cx="700" cy="350" r="3" />
        <circle cx="730" cy="380" r="2" />
      </g>
    </svg>
  )
}
