import type { SVGProps } from 'react'

export interface HoneycombPatternProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function HoneycombPattern({ size = 56, className, ...props }: HoneycombPatternProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 100 100"
      className={className}
      {...props}
    >
      <defs>
    <pattern id="honeycomb" x="0" y="0" width="56" height="100" patternUnits="userSpaceOnUse">
      <path
        d="M28 66L0 50L0 16L28 0L56 16L56 50L28 66L28 100"
        fill="none"
        stroke="currentColor"
        strokeWidth="0.5"
      />
      <path d="M28 0L28 34L0 50L0 84L28 100L56 84L56 50L28 34" fill="none" stroke="currentColor" strokeWidth="0.5" />
    </pattern>
    <mask id="fade-mask">
      <radialGradient id="fade-gradient">
        <stop offset="0%" stop-color="white" />
        <stop offset="70%" stop-color="white" />
        <stop offset="100%" stop-color="black" />
      </radialGradient>
      <rect width="100%" height="100%" fill="url(#fade-gradient)" />
    </mask>
  </defs>
  <rect width="100%" height="100%" fill="url(#honeycomb)" mask="url(#fade-mask)" />
    </svg>
  )
}
