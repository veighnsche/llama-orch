import type { SVGProps } from 'react'

export interface BeeGlyphProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function BeeGlyph({ size = 4, className, ...props }: BeeGlyphProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 64 64"
      className={className}
      {...props}
    >
      <circle cx="32" cy="32" r="24" />
  <path d="M20 32h24M32 20v24" stroke="white" strokeWidth="4" />
    </svg>
  )
}
