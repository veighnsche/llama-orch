import type { SVGProps } from 'react'

export interface DevGridProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function DevGrid({ size = 96, className, ...props }: DevGridProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 96 96"
      className={className}
      {...props}
    >
      {/* Blueprint-style developer illustration */}
      <rect width="96" height="96" className="fill-slate-50 dark:fill-slate-900" />

      {/* Grid pattern */}
      <g opacity="0.15">
        <line x1="24" y1="0" x2="24" y2="96" className="stroke-blue-500" strokeWidth="0.5" />
        <line x1="48" y1="0" x2="48" y2="96" className="stroke-blue-500" strokeWidth="0.5" />
        <line x1="72" y1="0" x2="72" y2="96" className="stroke-blue-500" strokeWidth="0.5" />
        <line x1="0" y1="24" x2="96" y2="24" className="stroke-blue-500" strokeWidth="0.5" />
        <line x1="0" y1="48" x2="96" y2="48" className="stroke-blue-500" strokeWidth="0.5" />
        <line x1="0" y1="72" x2="96" y2="72" className="stroke-blue-500" strokeWidth="0.5" />
      </g>

      {/* Code window */}
      <rect
        x="20"
        y="28"
        width="56"
        height="40"
        rx="4"
        className="fill-slate-100 dark:fill-slate-800 stroke-blue-500"
        strokeWidth="1.5"
      />

      {/* Code lines */}
      <line x1="26" y1="36" x2="46" y2="36" className="stroke-blue-500" strokeWidth="2" strokeLinecap="round" />
      <line
        x1="26"
        y1="44"
        x2="56"
        y2="44"
        className="stroke-blue-500"
        strokeWidth="2"
        strokeLinecap="round"
        opacity="0.6"
      />
      <line
        x1="26"
        y1="52"
        x2="40"
        y2="52"
        className="stroke-blue-500"
        strokeWidth="2"
        strokeLinecap="round"
        opacity="0.4"
      />
      <line x1="26" y1="60" x2="50" y2="60" className="stroke-emerald-500" strokeWidth="2" strokeLinecap="round" />

      {/* GPU chip icon */}
      <rect x="32" y="14" width="32" height="8" rx="2" className="fill-blue-500" opacity="0.8" />
      <circle cx="48" cy="18" r="2" className="fill-slate-50 dark:fill-slate-900" />
    </svg>
  )
}
