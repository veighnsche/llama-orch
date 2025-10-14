import type { SVGProps } from 'react'

export interface PricingScaleVisualProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function PricingScaleVisual({ size = 1400, className, ...props }: PricingScaleVisualProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 1400 500"
      className={className}
      {...props}
    >
      <defs>
    {/* Background gradients */}
    <radialGradient id="bgRadial" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#1e293b;stop-opacity:0.6" />
      <stop offset="100%" style="stop-color:#0f172a;stop-opacity:1" />
    </radialGradient>
    
    <radialGradient id="amberGlow" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#f59e0b;stop-opacity:0.7" />
      <stop offset="50%" style="stop-color:#f59e0b;stop-opacity:0.3" />
      <stop offset="100%" style="stop-color:#f59e0b;stop-opacity:0" />
    </radialGradient>
    
    <radialGradient id="tealGlow" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#14b8a6;stop-opacity:0.7" />
      <stop offset="50%" style="stop-color:#14b8a6;stop-opacity:0.3" />
      <stop offset="100%" style="stop-color:#14b8a6;stop-opacity:0" />
    </radialGradient>
    
    <linearGradient id="serverGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#475569" />
      <stop offset="50%" style="stop-color:#334155" />
      <stop offset="100%" style="stop-color:#1e293b" />
    </linearGradient>
    
    <filter id="glow">
      <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    
    <filter id="softGlow">
      <feGaussianBlur stdDeviation="8" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  
  {/* Base background */}
  <rect width="1400" height="500" fill="#0f172a"/>
  <rect width="1400" height="500" fill="url(#bgRadial)"/>
  
  {/* LEFT: Single Server (Start Free) */}
  <g id="singleServer">
    {/* Label with background */}
    <g transform="translate(180, 30)">
      <rect x="-70" y="0" width="140" height="35" rx="6" fill="#0f172a" opacity="0.9" stroke="#64748b" stroke-width="1.5"/>
      <text x="0" y="23" font-family="sans-serif" font-size="16" fill="#cbd5e1" text-anchor="middle" font-weight="700" letter-spacing="2">START FREE</text>
    </g>
    
    {/* Server tower */}
    <g transform="translate(100, 120)">
      <rect x="0" y="0" width="160" height="220" rx="8" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="3"/>
      
      {/* Front panel */}
      <rect x="10" y="10" width="140" height="200" rx="4" fill="#1e293b"/>
      
      {/* GPU slot (single) */}
      <g transform="translate(25, 80)">
        <rect width="110" height="60" rx="3" fill="#2d5016" stroke="#3d6b1f" stroke-width="1.5"/>
        {/* Heatsink */}
        <rect x="10" y="10" width="90" height="40" fill="#94a3b8" opacity="0.6"/>
        {/* Fins */}
        <line x1="15" y1="10" x2="15" y2="50" stroke="#cbd5e1" stroke-width="1.5"/>
        <line x1="25" y1="10" x2="25" y2="50" stroke="#cbd5e1" stroke-width="1.5"/>
        <line x1="35" y1="10" x2="35" y2="50" stroke="#cbd5e1" stroke-width="1.5"/>
        <line x1="45" y1="10" x2="45" y2="50" stroke="#cbd5e1" stroke-width="1.5"/>
        <line x1="55" y1="10" x2="55" y2="50" stroke="#cbd5e1" stroke-width="1.5"/>
        <line x1="65" y1="10" x2="65" y2="50" stroke="#cbd5e1" stroke-width="1.5"/>
        <line x1="75" y1="10" x2="75" y2="50" stroke="#cbd5e1" stroke-width="1.5"/>
        <line x1="85" y1="10" x2="85" y2="50" stroke="#cbd5e1" stroke-width="1.5"/>
        <line x1="95" y1="10" x2="95" y2="50" stroke="#cbd5e1" stroke-width="1.5"/>
        {/* LED */}
        <rect x="0" y="60" width="110" height="6" fill="#f59e0b" opacity="0.8" rx="1">
          <animate attributeName="opacity" values="0.6;1;0.6" dur="2s" repeatCount="indefinite"/>
        </rect>
      </g>
      
      {/* Status LEDs */}
      <circle cx="20" cy="25" r="3" fill="#10b981" opacity="0.9">
        <animate attributeName="opacity" values="0.7;1;0.7" dur="2.5s" repeatCount="indefinite"/>
      </circle>
      <circle cx="30" cy="25" r="3" fill="#10b981" opacity="0.7"/>
      
      {/* Amber glow from GPU */}
      <ellipse cx="80" cy="190" rx="80" ry="60" fill="url(#amberGlow)" opacity="0.5" filter="url(#softGlow)"/>
    </g>
    
    {/* Stats */}
    <g transform="translate(80, 380)">
      <rect width="200" height="70" rx="8" fill="#0f172a" opacity="0.9" stroke="#f59e0b" stroke-width="2"/>
      <text x="100" y="25" font-family="sans-serif" font-size="15" fill="#cbd5e1" text-anchor="middle" font-weight="600">1 Worker</text>
      <text x="100" y="52" font-family="sans-serif" font-size="24" fill="#f59e0b" text-anchor="middle" font-weight="800">$0/mo</text>
    </g>
  </g>
  
  {/* ARROW */}
  <g transform="translate(400, 250)">
    <path d="M 0,0 L 180,0 L 165,-15 M 180,0 L 165,15" stroke="#14b8a6" stroke-width="3.5" fill="none" stroke-linecap="round" opacity="0.9"/>
    <text x="90" y="-20" font-family="sans-serif" font-size="16" fill="#14b8a6" text-anchor="middle" font-weight="700" letter-spacing="1">SCALE</text>
  </g>
  
  {/* CENTER: Small Cluster (Growth) */}
  <g id="smallCluster">
    {/* Label with background */}
    <g transform="translate(700, 30)">
      <rect x="-60" y="0" width="120" height="35" rx="6" fill="#0f172a" opacity="0.9" stroke="#64748b" stroke-width="1.5"/>
      <text x="0" y="23" font-family="sans-serif" font-size="16" fill="#cbd5e1" text-anchor="middle" font-weight="700" letter-spacing="2">SCALING</text>
    </g>
    
    {/* Three servers in rack formation */}
    {/* Server 1 */}
    <g transform="translate(600, 150)">
      <rect width="80" height="90" rx="6" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="2"/>
      <rect x="5" y="5" width="70" height="80" rx="3" fill="#1e293b"/>
      {/* Mini GPU */}
      <rect x="15" y="30" width="50" height="30" rx="2" fill="#2d5016"/>
      <rect x="20" y="35" width="40" height="20" fill="#94a3b8" opacity="0.6"/>
      <rect x="15" y="60" width="50" height="3" fill="#f59e0b" opacity="0.8"/>
      <circle cx="12" cy="12" r="2" fill="#10b981" opacity="0.9"/>
    </g>
    
    {/* Server 2 */}
    <g transform="translate(600, 260)">
      <rect width="80" height="90" rx="6" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="2"/>
      <rect x="5" y="5" width="70" height="80" rx="3" fill="#1e293b"/>
      <rect x="15" y="30" width="50" height="30" rx="2" fill="#2d5016"/>
      <rect x="20" y="35" width="40" height="20" fill="#94a3b8" opacity="0.6"/>
      <rect x="15" y="60" width="50" height="3" fill="#f59e0b" opacity="0.8"/>
      <circle cx="12" cy="12" r="2" fill="#10b981" opacity="0.9"/>
    </g>
    
    {/* Server 3 */}
    <g transform="translate(720, 150)">
      <rect width="80" height="90" rx="6" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="2"/>
      <rect x="5" y="5" width="70" height="80" rx="3" fill="#1e293b"/>
      <rect x="15" y="30" width="50" height="30" rx="2" fill="#2d5016"/>
      <rect x="20" y="35" width="40" height="20" fill="#94a3b8" opacity="0.6"/>
      <rect x="15" y="60" width="50" height="3" fill="#f59e0b" opacity="0.8"/>
      <circle cx="12" cy="12" r="2" fill="#10b981" opacity="0.9"/>
    </g>
    
    {/* Server 4 */}
    <g transform="translate(720, 260)">
      <rect width="80" height="90" rx="6" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="2"/>
      <rect x="5" y="5" width="70" height="80" rx="3" fill="#1e293b"/>
      <rect x="15" y="30" width="50" height="30" rx="2" fill="#2d5016"/>
      <rect x="20" y="35" width="40" height="20" fill="#94a3b8" opacity="0.6"/>
      <rect x="15" y="60" width="50" height="3" fill="#f59e0b" opacity="0.8"/>
      <circle cx="12" cy="12" r="2" fill="#10b981" opacity="0.9"/>
    </g>
    
    {/* Network lines */}
    <g opacity="0.7" filter="url(#glow)">
      <line x1="640" y1="240" x2="640" y2="260" stroke="#14b8a6" stroke-width="2"/>
      <line x1="760" y1="240" x2="760" y2="260" stroke="#14b8a6" stroke-width="2"/>
      <line x1="680" y1="195" x2="720" y2="195" stroke="#14b8a6" stroke-width="2"/>
      <line x1="680" y1="305" x2="720" y2="305" stroke="#14b8a6" stroke-width="2"/>
    </g>
    
    {/* Glow */}
    <ellipse cx="700" cy="250" rx="120" ry="100" fill="url(#amberGlow)" opacity="0.4" filter="url(#softGlow)"/>
    
    {/* Stats */}
    <g transform="translate(600, 380)">
      <rect width="200" height="70" rx="8" fill="#0f172a" opacity="0.9" stroke="#f59e0b" stroke-width="2"/>
      <text x="100" y="25" font-family="sans-serif" font-size="15" fill="#cbd5e1" text-anchor="middle" font-weight="600">4 Workers</text>
      <text x="100" y="52" font-family="sans-serif" font-size="24" fill="#f59e0b" text-anchor="middle" font-weight="800">$49/mo</text>
    </g>
  </g>
  
  {/* ARROW */}
  <g transform="translate(920, 250)">
    <path d="M 0,0 L 180,0 L 165,-15 M 180,0 L 165,15" stroke="#14b8a6" stroke-width="3.5" fill="none" stroke-linecap="round" opacity="0.9"/>
    <text x="90" y="-20" font-family="sans-serif" font-size="16" fill="#14b8a6" text-anchor="middle" font-weight="700" letter-spacing="1">GROW</text>
  </g>
  
  {/* RIGHT: Large Cluster (Enterprise) */}
  <g id="largeCluster">
    {/* Label with background */}
    <g transform="translate(1240, 30)">
      <rect x="-70" y="0" width="140" height="35" rx="6" fill="#0f172a" opacity="0.9" stroke="#64748b" stroke-width="1.5"/>
      <text x="0" y="23" font-family="sans-serif" font-size="16" fill="#cbd5e1" text-anchor="middle" font-weight="700" letter-spacing="2">ENTERPRISE</text>
    </g>
    
    {/* Rack of servers (grid) */}
    {/* Row 1 */}
    <rect x="1140" y="140" width="45" height="50" rx="4" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="1.5"/>
    <rect x="1195" y="140" width="45" height="50" rx="4" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="1.5"/>
    <rect x="1250" y="140" width="45" height="50" rx="4" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="1.5"/>
    <rect x="1305" y="140" width="45" height="50" rx="4" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="1.5"/>
    
    {/* Row 2 */}
    <rect x="1140" y="200" width="45" height="50" rx="4" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="1.5"/>
    <rect x="1195" y="200" width="45" height="50" rx="4" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="1.5"/>
    <rect x="1250" y="200" width="45" height="50" rx="4" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="1.5"/>
    <rect x="1305" y="200" width="45" height="50" rx="4" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="1.5"/>
    
    {/* Row 3 */}
    <rect x="1140" y="260" width="45" height="50" rx="4" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="1.5"/>
    <rect x="1195" y="260" width="45" height="50" rx="4" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="1.5"/>
    <rect x="1250" y="260" width="45" height="50" rx="4" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="1.5"/>
    <rect x="1305" y="260" width="45" height="50" rx="4" fill="url(#serverGradient)" stroke="#1e293b" stroke-width="1.5"/>
    
    {/* Add LEDs and GPU indicators */}
    <g opacity="0.8">
      <rect x="1145" y="175" width="35" height="3" fill="#f59e0b"/>
      <rect x="1200" y="175" width="35" height="3" fill="#f59e0b"/>
      <rect x="1255" y="175" width="35" height="3" fill="#f59e0b"/>
      <rect x="1310" y="175" width="35" height="3" fill="#f59e0b"/>
      
      <rect x="1145" y="235" width="35" height="3" fill="#f59e0b"/>
      <rect x="1200" y="235" width="35" height="3" fill="#f59e0b"/>
      <rect x="1255" y="235" width="35" height="3" fill="#f59e0b"/>
      <rect x="1310" y="235" width="35" height="3" fill="#f59e0b"/>
      
      <rect x="1145" y="295" width="35" height="3" fill="#f59e0b"/>
      <rect x="1200" y="295" width="35" height="3" fill="#f59e0b"/>
      <rect x="1255" y="295" width="35" height="3" fill="#f59e0b"/>
      <rect x="1310" y="295" width="35" height="3" fill="#f59e0b"/>
    </g>
    
    {/* Network mesh overlay */}
    <g opacity="0.6" filter="url(#glow)">
      <line x1="1162" y1="190" x2="1217" y2="190" stroke="#14b8a6" stroke-width="1.5"/>
      <line x1="1217" y1="190" x2="1272" y2="190" stroke="#14b8a6" stroke-width="1.5"/>
      <line x1="1272" y1="190" x2="1327" y2="190" stroke="#14b8a6" stroke-width="1.5"/>
      
      <line x1="1162" y1="250" x2="1217" y2="250" stroke="#14b8a6" stroke-width="1.5"/>
      <line x1="1217" y1="250" x2="1272" y2="250" stroke="#14b8a6" stroke-width="1.5"/>
      <line x1="1272" y1="250" x2="1327" y2="250" stroke="#14b8a6" stroke-width="1.5"/>
      
      <line x1="1162" y1="165" x2="1162" y2="285" stroke="#14b8a6" stroke-width="1.5"/>
      <line x1="1217" y1="165" x2="1217" y2="285" stroke="#14b8a6" stroke-width="1.5"/>
      <line x1="1272" y1="165" x2="1272" y2="285" stroke="#14b8a6" stroke-width="1.5"/>
      <line x1="1327" y1="165" x2="1327" y2="285" stroke="#14b8a6" stroke-width="1.5"/>
    </g>
    
    {/* Intense glow */}
    <ellipse cx="1245" cy="225" rx="130" ry="110" fill="url(#amberGlow)" opacity="0.6" filter="url(#softGlow)"/>
    
    {/* Stats */}
    <g transform="translate(1140, 380)">
      <rect width="210" height="70" rx="8" fill="#0f172a" opacity="0.9" stroke="#f59e0b" stroke-width="2"/>
      <text x="105" y="25" font-family="sans-serif" font-size="15" fill="#cbd5e1" text-anchor="middle" font-weight="600">12+ Workers</text>
      <text x="105" y="52" font-family="sans-serif" font-size="24" fill="#f59e0b" text-anchor="middle" font-weight="800">Custom</text>
    </g>
  </g>
  
  {/* Data particles flowing */}
  <g opacity="0.5">
    <circle r="3" fill="#14b8a6">
      <animateMotion dur="4s" repeatCount="indefinite" path="M 180 250 L 580 250"/>
      <animate attributeName="opacity" values="0;1;1;0" dur="4s" repeatCount="indefinite"/>
    </circle>
    <circle r="3" fill="#14b8a6">
      <animateMotion dur="4s" begin="1s" repeatCount="indefinite" path="M 700 250 L 1100 250"/>
      <animate attributeName="opacity" values="0;1;1;0" dur="4s" begin="1s" repeatCount="indefinite"/>
    </circle>
  </g>
    </svg>
  )
}
