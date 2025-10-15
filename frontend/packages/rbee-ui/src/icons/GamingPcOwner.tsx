import type { SVGProps } from 'react'

export interface GamingPcOwnerProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function GamingPcOwner({ size = 96, className, ...props }: GamingPcOwnerProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 96 96"
      className={className}
      {...props}
    >
      {/* Background circle */}
      <circle cx="48" cy="48" r="48" fill="#1a1f2e" />

      {/* Desk surface */}
      <rect x="12" y="68" width="72" height="4" rx="1" fill="#2d3548" />
      <rect x="12" y="72" width="72" height="2" fill="#1f2533" />

      {/* PC Tower (left side) */}
      <g id="pc-tower">
        {/* Tower body */}
        <rect x="16" y="38" width="18" height="30" rx="2" fill="#2d3548" stroke="#3d4556" strokeWidth="1" />

        {/* Tempered glass panel with gradient */}
        <defs>
          <linearGradient id="glassGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#4a5568', stopOpacity: 0.3 }} />
            <stop offset="100%" style={{ stopColor: '#2d3748', stopOpacity: 0.6 }} />
          </linearGradient>
          <linearGradient id="rgbGlow" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#ff0080', stopOpacity: 0.8 }} />
            <stop offset="33%" style={{ stopColor: '#7928ca', stopOpacity: 0.8 }} />
            <stop offset="66%" style={{ stopColor: '#0070f3', stopOpacity: 0.8 }} />
            <stop offset="100%" style={{ stopColor: '#00dfd8', stopOpacity: 0.8 }} />
          </linearGradient>
        </defs>
        <rect x="18" y="40" width="14" height="26" rx="1" fill="url(#glassGradient)" opacity="0.6" />

        {/* GPU fans visible through glass (RGB lit) */}
        <circle cx="25" cy="50" r="3" fill="none" stroke="url(#rgbGlow)" strokeWidth="0.8" opacity="0.9" />
        <circle cx="25" cy="50" r="1.5" fill="url(#rgbGlow)" opacity="0.7" />
        <circle cx="25" cy="58" r="3" fill="none" stroke="url(#rgbGlow)" strokeWidth="0.8" opacity="0.9" />
        <circle cx="25" cy="58" r="1.5" fill="url(#rgbGlow)" opacity="0.7" />

        {/* Power button */}
        <circle cx="25" cy="43" r="1" fill="#ff9500" opacity="0.9" />
        <circle cx="25" cy="43" r="1.5" fill="#ff9500" opacity="0.3" />

        {/* Front panel accents */}
        <rect x="18" y="64" width="14" height="1" fill="#3d4556" />
      </g>

      {/* Monitor (center) */}
      <g id="monitor">
        {/* Monitor stand */}
        <rect x="44" y="64" width="8" height="4" rx="1" fill="#2d3548" />
        <rect x="42" y="68" width="12" height="2" rx="1" fill="#3d4556" />

        {/* Monitor bezel */}
        <rect x="38" y="32" width="28" height="20" rx="1.5" fill="#1f2533" stroke="#2d3548" strokeWidth="1" />

        {/* Screen with gradient (active) */}
        <defs>
          <linearGradient id="screenGlow" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#667eea', stopOpacity: 0.4 }} />
            <stop offset="100%" style={{ stopColor: '#764ba2', stopOpacity: 0.3 }} />
          </linearGradient>
        </defs>
        <rect x="40" y="34" width="24" height="16" rx="0.5" fill="url(#screenGlow)" />

        {/* Screen content suggestion (game UI) */}
        <rect x="42" y="36" width="8" height="2" rx="0.5" fill="#00dfd8" opacity="0.6" />
        <rect x="42" y="39" width="12" height="1" rx="0.3" fill="#ffffff" opacity="0.3" />
        <rect x="42" y="41" width="10" height="1" rx="0.3" fill="#ffffff" opacity="0.3" />
        <circle cx="58" cy="44" r="3" fill="#ff0080" opacity="0.4" />
        <circle cx="58" cy="44" r="1.5" fill="#ff0080" opacity="0.6" />
      </g>

      {/* Keyboard (right side) */}
      <g id="keyboard">
        {/* Keyboard body */}
        <rect x="46" y="70" width="32" height="8" rx="1" fill="#2d3548" stroke="#3d4556" strokeWidth="0.5" />

        {/* Key rows with RGB backlight glow */}
        <rect x="48" y="71.5" width="28" height="1.2" rx="0.3" fill="#4a5568" opacity="0.8" />
        <rect x="48" y="73" width="28" height="1.2" rx="0.3" fill="#4a5568" opacity="0.8" />
        <rect x="48" y="74.5" width="28" height="1.2" rx="0.3" fill="#4a5568" opacity="0.8" />
        <rect x="48" y="76" width="28" height="1.2" rx="0.3" fill="#4a5568" opacity="0.8" />

        {/* RGB backlight effect with orange accent */}
        <rect x="48" y="71.5" width="8" height="1.2" rx="0.3" fill="#ff9500" opacity="0.4" />
        <rect x="57" y="71.5" width="8" height="1.2" rx="0.3" fill="#ff6b00" opacity="0.4" />
        <rect x="66" y="71.5" width="10" height="1.2" rx="0.3" fill="#ff9500" opacity="0.3" />
      </g>

      {/* Mouse (small detail) */}
      <ellipse cx="80" cy="72" rx="2.5" ry="3" fill="#2d3548" stroke="#3d4556" strokeWidth="0.5" />
      <line x1="80" y1="70" x2="80" y2="73" stroke="#4a5568" strokeWidth="0.3" />

      {/* Ambient lighting effects */}
      <defs>
        <radialGradient id="ambientGlow">
          <stop offset="0%" style={{ stopColor: '#667eea', stopOpacity: 0.2 }} />
          <stop offset="100%" style={{ stopColor: '#667eea', stopOpacity: 0 }} />
        </radialGradient>
      </defs>
      <ellipse cx="48" cy="75" rx="35" ry="8" fill="url(#ambientGlow)" />

      {/* Subtle highlights for depth */}
      <line x1="18" y1="40" x2="18" y2="50" stroke="#4a5568" strokeWidth="0.5" opacity="0.5" />
      <line x1="40" y1="34" x2="50" y2="34" stroke="#3d4556" strokeWidth="0.3" opacity="0.6" />
    </svg>
  )
}
