import type { SVGProps } from 'react'

export interface UseCasesHeroProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function UseCasesHero({ size = 1080, className, ...props }: UseCasesHeroProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 1080 760"
      className={className}
      {...props}
    >
      {/* Background */}
  <rect width="1080" height="760" fill="#0f172a"/>
  
  {/* Radial gradient for depth */}
  <defs>
    <radialGradient id="glow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style={{ stopColor: '#f59e0b', stopOpacity: 0.15 }} />
      <stop offset="100%" style={{ stopColor: '#0f172a', stopOpacity: 0 }} />
    </radialGradient>
  </defs>
  <circle cx="540" cy="380" r="300" fill="url(#glow)"/>
  
  {/* Laptop screen (left) */}
  <rect x="200" y="250" width="300" height="200" rx="8" fill="#1e293b" stroke="#334155" strokeWidth="2"/>
  <rect x="210" y="260" width="280" height="170" rx="4" fill="#0a0f1a"/>
  
  {/* Terminal text lines */}
  <text x="220" y="285" fontFamily="monospace" fontSize="12" fill="#10b981" opacity="0.8">$ llama-cli --model llama-70b</text>
  <text x="220" y="305" fontFamily="monospace" fontSize="12" fill="#10b981" opacity="0.7">Loading model...</text>
  <text x="220" y="325" fontFamily="monospace" fontSize="12" fill="#10b981" opacity="0.6">Generating tokens...</text>
  <text x="220" y="345" fontFamily="monospace" fontSize="12" fill="#10b981" opacity="0.5">▊</text>
  
  {/* GPU rack (right) */}
  <rect x="600" y="280" width="200" height="180" rx="4" fill="#1e293b" stroke="#334155" strokeWidth="2"/>
  
  {/* GPU cards with amber glow */}
  <rect x="620" y="300" width="160" height="40" rx="2" fill="#374151" stroke="#f59e0b" strokeWidth="1"/>
  <circle cx="760" cy="320" r="3" fill="#f59e0b">
    <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>
  </circle>
  
  <rect x="620" y="360" width="160" height="40" rx="2" fill="#374151" stroke="#f59e0b" strokeWidth="1"/>
  <circle cx="760" cy="380" r="3" fill="#f59e0b">
    <animate attributeName="opacity" values="1;0.5;1" dur="2s" repeatCount="indefinite"/>
  </circle>
  
  <rect x="620" y="420" width="160" height="40" rx="2" fill="#374151" stroke="#f59e0b" strokeWidth="1"/>
  <circle cx="760" cy="440" r="3" fill="#f59e0b">
    <animate attributeName="opacity" values="0.7;1;0.7" dur="2s" repeatCount="indefinite"/>
  </circle>
  
  {/* Sticky note */}
  <rect x="420" y="180" width="140" height="120" rx="2" fill="#fde047" opacity="0.95"/>
  <text x="435" y="220" fontFamily="cursive" fontSize="16" fill="#0f172a" fontWeight="500">your models,</text>
  <text x="435" y="245" fontFamily="cursive" fontSize="16" fill="#0f172a" fontWeight="500">your rules</text>
  
  {/* Keyboard (foreground, subtle) */}
  <rect x="180" y="480" width="340" height="100" rx="4" fill="#1e293b" opacity="0.6" stroke="#334155" strokeWidth="1"/>
  
  {/* Coffee mug */}
  <ellipse cx="850" cy="520" rx="30" ry="15" fill="#374151"/>
  <rect x="820" y="470" width="60" height="50" rx="4" fill="#475569"/>
  <rect x="880" y="490" width="15" height="30" rx="8" fill="#475569"/>
  
  {/* Placeholder text */}
  <text x="540" y="600" fontFamily="sans-serif" fontSize="18" fill="#64748b" textAnchor="middle" opacity="0.7">
    Placeholder: Generate image using use-cases-hero-GENERATION.md
  </text>
  <text x="540" y="630" fontFamily="sans-serif" fontSize="14" fill="#64748b" textAnchor="middle" opacity="0.5">
    Cozy homelab desk • GPUs glowing amber • Terminal streaming tokens
  </text>
    </svg>
  )
}
