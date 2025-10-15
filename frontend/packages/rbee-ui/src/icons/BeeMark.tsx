import type { SVGProps } from 'react'

export interface BeeMarkProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function BeeMark({ size = 24, className, ...props }: BeeMarkProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      className={className}
      {...props}
    >
      {/* Bee body (golden yellow) */}
      <ellipse cx="12" cy="13" rx="5" ry="6" fill="#f59e0b" />

      {/* Stripes (charcoal) */}
      <rect x="7" y="10" width="10" height="1.5" rx="0.75" fill="#18181b" />
      <rect x="7" y="13" width="10" height="1.5" rx="0.75" fill="#18181b" />
      <rect x="7" y="16" width="10" height="1.5" rx="0.75" fill="#18181b" />

      {/* Head (charcoal) - geometric shape for professional look */}
      <rect x="9" y="4.5" width="6" height="5" rx="1.5" fill="#18181b" />

      {/* Left wing (golden yellow with motion) */}
      <ellipse cx="8" cy="11" rx="3" ry="4" fill="#f59e0b" opacity="0.7" transform="rotate(-25 8 11)" />

      {/* Right wing (golden yellow with motion) */}
      <ellipse cx="16" cy="11" rx="3" ry="4" fill="#f59e0b" opacity="0.7" transform="rotate(25 16 11)" />

      {/* Antennae (charcoal) - clean lines with geometric tips */}
      <line x1="10.5" y1="5.5" x2="9" y2="3" stroke="#18181b" strokeWidth="1" strokeLinecap="round" />
      <line x1="13.5" y1="5.5" x2="15" y2="3" stroke="#18181b" strokeWidth="1" strokeLinecap="round" />
      <polygon points="9,2.2 8.5,3.5 9.5,3.5" fill="#18181b" />
      <polygon points="15,2.2 14.5,3.5 15.5,3.5" fill="#18181b" />
    </svg>
  )
}
