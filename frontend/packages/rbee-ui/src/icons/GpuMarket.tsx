import type { SVGProps } from 'react'

export interface GpuMarketProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function GpuMarket({ size = 96, className, ...props }: GpuMarketProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 96 96"
      className={className}
      {...props}
    >
      {/* Honeycomb marketplace illustration */}
  <rect width="96" height="96" fill="#0f172a"/>
  
  {/* Honeycomb pattern */}
  <g opacity="0.8">
    {/* Top row */}
    <path d="M 32 20 L 40 16 L 48 20 L 48 28 L 40 32 L 32 28 Z" stroke="#10b981" stroke-width="1.5" fill="#10b981" fill-opacity="0.1"/>
    <path d="M 48 20 L 56 16 L 64 20 L 64 28 L 56 32 L 48 28 Z" stroke="#10b981" stroke-width="1.5" fill="#10b981" fill-opacity="0.15"/>
    
    {/* Middle row */}
    <path d="M 24 36 L 32 32 L 40 36 L 40 44 L 32 48 L 24 44 Z" stroke="#10b981" stroke-width="1.5" fill="#10b981" fill-opacity="0.2"/>
    <path d="M 40 36 L 48 32 L 56 36 L 56 44 L 48 48 L 40 44 Z" stroke="#10b981" stroke-width="2" fill="#10b981" fill-opacity="0.3"/>
    <path d="M 56 36 L 64 32 L 72 36 L 72 44 L 64 48 L 56 44 Z" stroke="#10b981" stroke-width="1.5" fill="#10b981" fill-opacity="0.15"/>
    
    {/* Bottom row */}
    <path d="M 32 52 L 40 48 L 48 52 L 48 60 L 40 64 L 32 60 Z" stroke="#10b981" stroke-width="1.5" fill="#10b981" fill-opacity="0.1"/>
    <path d="M 48 52 L 56 48 L 64 52 L 64 60 L 56 64 L 48 60 Z" stroke="#10b981" stroke-width="1.5" fill="#10b981" fill-opacity="0.2"/>
  </g>
  
  {/* Connection lines */}
  <line x1="48" y1="40" x2="48" y2="72" stroke="#3b82f6" stroke-width="1" opacity="0.4" stroke-dasharray="2 2"/>
  <line x1="32" y1="40" x2="64" y2="40" stroke="#3b82f6" stroke-width="1" opacity="0.4" stroke-dasharray="2 2"/>
  
  {/* Dollar sign indicator */}
  <circle cx="48" cy="76" r="8" fill="#10b981" opacity="0.9"/>
  <text x="48" y="81" text-anchor="middle" fill="#0f172a" font-size="12" font-weight="bold">$</text>
    </svg>
  )
}
