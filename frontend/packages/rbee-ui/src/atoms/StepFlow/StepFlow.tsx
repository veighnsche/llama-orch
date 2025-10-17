import { cn } from '@rbee/ui/utils'

export interface StepFlowProps {
  className?: string
}

/**
 * StepFlow - Theme-aware SVG background for how-it-works sections
 *
 * Sequential numbered steps with progress indicators and curved connecting paths,
 * suggesting a clear progression through a process.
 *
 * @example
 * ```tsx
 * <div className="absolute inset-0 opacity-25">
 *   <StepFlow />
 * </div>
 * ```
 */
export function StepFlow({ className }: StepFlowProps) {
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
        <filter id="step-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Progress path - light theme */}
      <g className="dark:hidden" opacity="0.4">
        <path
          d="M 200 320 Q 400 200 600 320 T 1000 320"
          stroke="rgb(59 130 246)"
          strokeWidth="2"
          fill="none"
        />
      </g>

      {/* Progress path - dark theme */}
      <g className="hidden dark:block" opacity="0.5">
        <path
          d="M 200 320 Q 400 200 600 320 T 1000 320"
          stroke="rgb(96 165 250)"
          strokeWidth="2.5"
          fill="none"
        />
      </g>

      {/* Step circles */}
      <g filter="url(#step-glow)">
        {/* Step 1 - Completed */}
        <circle cx="200" cy="320" r="20" className="fill-none stroke-emerald-500 dark:stroke-emerald-400" strokeWidth="2" />
        <circle cx="200" cy="320" r="16" className="fill-emerald-500 dark:fill-emerald-400" opacity="0.3" />
        
        {/* Step 2 - Completed */}
        <circle cx="600" cy="320" r="20" className="fill-none stroke-emerald-500 dark:stroke-emerald-400" strokeWidth="2" />
        <circle cx="600" cy="320" r="16" className="fill-emerald-500 dark:fill-emerald-400" opacity="0.3" />
        
        {/* Step 3 - In Progress */}
        <circle cx="1000" cy="320" r="20" className="fill-none stroke-blue-500 dark:stroke-blue-400" strokeWidth="2" />
        <circle cx="1000" cy="320" r="16" className="fill-blue-500 dark:fill-blue-400" opacity="0.3" />
      </g>

      {/* Step numbers */}
      <g className="fill-blue-500 dark:fill-blue-400" opacity="0.8">
        <text x="200" y="328" textAnchor="middle" fontSize="16" fontWeight="600">1</text>
        <text x="600" y="328" textAnchor="middle" fontSize="16" fontWeight="600">2</text>
        <text x="1000" y="328" textAnchor="middle" fontSize="16" fontWeight="600">3</text>
      </g>

      {/* Checkmarks for completed steps */}
      <g className="stroke-emerald-500 dark:stroke-emerald-400" strokeWidth="2.5" opacity="0.7">
        <path d="M 192 320 L 198 326 L 208 314" fill="none" />
        <path d="M 592 320 L 598 326 L 608 314" fill="none" />
      </g>

      {/* Pulse effect on current step */}
      <g opacity="0.3">
        <circle cx="1000" cy="320" r="28" className="fill-none stroke-blue-500 dark:stroke-blue-400" strokeWidth="1" />
        <circle cx="1000" cy="320" r="36" className="fill-none stroke-blue-500 dark:stroke-blue-400" strokeWidth="0.5" />
      </g>
    </svg>
  )
}
