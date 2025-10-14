import type { SVGProps } from 'react'

export interface HomelabBeeProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function HomelabBee({ size = 960, className, ...props }: HomelabBeeProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 960 140"
      className={className}
      {...props}
    >
      {/* Homelab hardware with bee orchestration theme */}
  <rect width="960" height="140" fill="#0f172a"/>
  
  {/* Grid pattern background */}
  <g opacity="0.08">
    <line x1="0" y1="35" x2="960" y2="35" stroke="#3b82f6" stroke-width="0.5"/>
    <line x1="0" y1="70" x2="960" y2="70" stroke="#3b82f6" stroke-width="0.5"/>
    <line x1="0" y1="105" x2="960" y2="105" stroke="#3b82f6" stroke-width="0.5"/>
  </g>
  
  {/* Left: Mini PC / Home server */}
  <g transform="translate(200, 40)">
    <rect x="0" y="0" width="60" height="60" rx="4" fill="#1e293b" stroke="#3b82f6" stroke-width="2"/>
    <circle cx="12" cy="12" r="3" fill="#10b981" opacity="0.8"/>
    <rect x="8" y="24" width="44" height="3" rx="1.5" fill="#3b82f6" opacity="0.4"/>
    <rect x="8" y="32" width="44" height="3" rx="1.5" fill="#3b82f6" opacity="0.6"/>
    <rect x="8" y="40" width="44" height="3" rx="1.5" fill="#3b82f6" opacity="0.3"/>
    <rect x="8" y="48" width="20" height="3" rx="1.5" fill="#10b981" opacity="0.7"/>
  </g>
  
  {/* Center: GPU Server rack */}
  <g transform="translate(420, 25)">
    <rect x="0" y="0" width="120" height="90" rx="6" fill="#1e293b" stroke="#10b981" stroke-width="2.5"/>
    {/* Rack slots */}
    <rect x="10" y="12" width="100" height="18" rx="2" fill="#10b981" fill-opacity="0.15" stroke="#10b981" stroke-width="1.5"/>
    <rect x="10" y="36" width="100" height="18" rx="2" fill="#10b981" fill-opacity="0.25" stroke="#10b981" stroke-width="1.5"/>
    <rect x="10" y="60" width="100" height="18" rx="2" fill="#10b981" fill-opacity="0.2" stroke="#10b981" stroke-width="1.5"/>
    {/* LED indicators */}
    <circle cx="20" cy="21" r="2.5" fill="#10b981"/>
    <circle cx="20" cy="45" r="2.5" fill="#10b981"/>
    <circle cx="20" cy="69" r="2.5" fill="#3b82f6"/>
    {/* GPU label hint */}
    <text x="35" y="25" fill="#10b981" font-size="8" opacity="0.6">GPU</text>
  </g>
  
  {/* Right: Another mini PC */}
  <g transform="translate(700, 40)">
    <rect x="0" y="0" width="60" height="60" rx="4" fill="#1e293b" stroke="#3b82f6" stroke-width="2"/>
    <circle cx="12" cy="12" r="3" fill="#10b981" opacity="0.8"/>
    <rect x="8" y="24" width="44" height="3" rx="1.5" fill="#3b82f6" opacity="0.5"/>
    <rect x="8" y="32" width="44" height="3" rx="1.5" fill="#3b82f6" opacity="0.4"/>
    <rect x="8" y="40" width="44" height="3" rx="1.5" fill="#3b82f6" opacity="0.6"/>
    <rect x="8" y="48" width="28" height="3" rx="1.5" fill="#10b981" opacity="0.7"/>
  </g>
  
  {/* Connection lines (network) */}
  <line x1="260" y1="70" x2="420" y2="70" stroke="#3b82f6" stroke-width="2" opacity="0.4" stroke-dasharray="4 4"/>
  <line x1="540" y1="70" x2="700" y2="70" stroke="#3b82f6" stroke-width="2" opacity="0.4" stroke-dasharray="4 4"/>
  
  {/* Data flow indicators */}
  <circle cx="340" cy="70" r="3" fill="#3b82f6" opacity="0.8">
    <animate attributeName="cx" from="260" to="420" dur="3s" repeatCount="indefinite"/>
  </circle>
  <circle cx="620" cy="70" r="3" fill="#3b82f6" opacity="0.8">
    <animate attributeName="cx" from="540" to="700" dur="3s" repeatCount="indefinite"/>
  </circle>
  
  {/* Bee icon (orchestrator) hovering above center */}
  <g transform="translate(460, 5)">
    {/* Bee body */}
    <ellipse cx="20" cy="12" rx="12" ry="10" fill="#f59e0b" opacity="0.9"/>
    {/* Stripes */}
    <rect x="14" y="8" width="3" height="8" fill="#0f172a" opacity="0.6"/>
    <rect x="20" y="8" width="3" height="8" fill="#0f172a" opacity="0.6"/>
    {/* Wings */}
    <ellipse cx="12" cy="8" rx="8" ry="6" fill="#3b82f6" opacity="0.3" stroke="#3b82f6" stroke-width="1"/>
    <ellipse cx="28" cy="8" rx="8" ry="6" fill="#3b82f6" opacity="0.3" stroke="#3b82f6" stroke-width="1"/>
    {/* Antenna */}
    <line x1="18" y1="4" x2="16" y2="0" stroke="#f59e0b" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="22" y1="4" x2="24" y2="0" stroke="#f59e0b" stroke-width="1.5" stroke-linecap="round"/>
    <circle cx="16" cy="0" r="1.5" fill="#10b981"/>
    <circle cx="24" cy="0" r="1.5" fill="#10b981"/>
  </g>
  
  {/* Subtle glow around bee */}
  <circle cx="480" cy="17" r="30" fill="#f59e0b" opacity="0.05"/>
    </svg>
  )
}
