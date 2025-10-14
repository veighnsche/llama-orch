import type { SVGProps } from 'react'

export interface ComplianceShieldProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function ComplianceShield({ size = 96, className, ...props }: ComplianceShieldProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 96 96"
      className={className}
      {...props}
    >
      {/* Audit logs illustration */}
  <rect width="96" height="96" fill="#0f172a"/>
  
  {/* Main document stack */}
  <g>
    {/* Back document */}
    <rect x="28" y="26" width="40" height="48" rx="2" fill="#1e293b" stroke="#f59e0b" stroke-width="1.5" opacity="0.4"/>
    
    {/* Middle document */}
    <rect x="32" y="22" width="40" height="48" rx="2" fill="#1e293b" stroke="#f59e0b" stroke-width="1.5" opacity="0.7"/>
    
    {/* Front document (main) */}
    <rect x="36" y="18" width="40" height="48" rx="2" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
    
    {/* Document header bar */}
    <rect x="36" y="18" width="40" height="8" rx="2" fill="#f59e0b" opacity="0.3"/>
    
    {/* Log entries (lines) */}
    <line x1="42" y1="32" x2="70" y2="32" stroke="#f59e0b" stroke-width="1.5" opacity="0.8"/>
    <line x1="42" y1="38" x2="66" y2="38" stroke="#f59e0b" stroke-width="1.5" opacity="0.6"/>
    <line x1="42" y1="44" x2="68" y2="44" stroke="#f59e0b" stroke-width="1.5" opacity="0.8"/>
    <line x1="42" y1="50" x2="64" y2="50" stroke="#f59e0b" stroke-width="1.5" opacity="0.6"/>
    <line x1="42" y1="56" x2="70" y2="56" stroke="#f59e0b" stroke-width="1.5" opacity="0.8"/>
    
    {/* Timestamp dots */}
    <circle cx="40" cy="32" r="1.5" fill="#f59e0b" opacity="0.8"/>
    <circle cx="40" cy="38" r="1.5" fill="#f59e0b" opacity="0.6"/>
    <circle cx="40" cy="44" r="1.5" fill="#f59e0b" opacity="0.8"/>
    <circle cx="40" cy="50" r="1.5" fill="#f59e0b" opacity="0.6"/>
    <circle cx="40" cy="56" r="1.5" fill="#f59e0b" opacity="0.8"/>
  </g>
  
  {/* Lock icon (security indicator) */}
  <g transform="translate(66, 58)">
    <rect x="-4" y="0" width="8" height="6" rx="1" fill="#f59e0b" opacity="0.9"/>
    <path d="M -2 0 L -2 -2 C -2 -3.1 -1.1 -4 0 -4 C 1.1 -4 2 -3.1 2 -2 L 2 0" stroke="#f59e0b" stroke-width="1.5" fill="none" opacity="0.9"/>
  </g>
    </svg>
  )
}
