import type { SVGProps } from 'react'

export interface FormerCryptoMinerProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function FormerCryptoMiner({ size = 96, className, ...props }: FormerCryptoMinerProps) {
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
  <circle cx="48" cy="48" r="48" fill="#1a1f2e"/>
  
  {/* Industrial shelving unit base */}
  <g id="shelving">
    {/* Bottom shelf */}
    <rect x="12" y="70" width="72" height="2" fill="#3d4556"/>
    <rect x="12" y="72" width="72" height="1" fill="#2d3548"/>
    
    {/* Shelf supports */}
    <rect x="14" y="30" width="2" height="42" fill="#2d3548"/>
    <rect x="80" y="30" width="2" height="42" fill="#2d3548"/>
  </g>
  
  {/* Open-air mining frame (aluminum rails) */}
  <g id="mining-frame">
    {/* Main frame structure */}
    <rect x="20" y="32" width="56" height="3" rx="0.5" fill="#4a5568" stroke="#5a6578" strokeWidth="0.5"/>
    <rect x="20" y="65" width="56" height="3" rx="0.5" fill="#4a5568" stroke="#5a6578" strokeWidth="0.5"/>
    
    {/* Vertical supports */}
    <rect x="20" y="32" width="3" height="36" rx="0.5" fill="#4a5568" stroke="#5a6578" strokeWidth="0.5"/>
    <rect x="73" y="32" width="3" height="36" rx="0.5" fill="#4a5568" stroke="#5a6578" strokeWidth="0.5"/>
    
    {/* Cross brace for stability */}
    <line x1="23" y1="50" x2="73" y2="50" stroke="#4a5568" strokeWidth="2" opacity="0.6"/>
    
    {/* Corner brackets (industrial look) */}
    <rect x="19" y="31" width="5" height="1" fill="#5a6578"/>
    <rect x="19" y="31" width="1" height="5" fill="#5a6578"/>
    <rect x="72" y="31" width="5" height="1" fill="#5a6578"/>
    <rect x="76" y="31" width="1" height="5" fill="#5a6578"/>
  </g>
  
  {/* GPUs mounted horizontally (8 GPUs) */}
  <g id="gpus">
    {/* GPU Row 1 (top) */}
    {/* GPU 1 */}
    <rect x="24" y="36" width="10" height="6" rx="0.5" fill="#2d3548" stroke="#3d4556" strokeWidth="0.5"/>
    <rect x="25" y="37" width="8" height="4" rx="0.3" fill="#1f2533"/>
    <circle cx="27" cy="39" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <circle cx="31" cy="39" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <rect x="25.5" y="37.5" width="2" height="0.5" fill="#00ff00" opacity="0.6"/>
    
    {/* GPU 2 */}
    <rect x="36" y="36" width="10" height="6" rx="0.5" fill="#2d3548" stroke="#3d4556" strokeWidth="0.5"/>
    <rect x="37" y="37" width="8" height="4" rx="0.3" fill="#1f2533"/>
    <circle cx="39" cy="39" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <circle cx="43" cy="39" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <rect x="37.5" y="37.5" width="2" height="0.5" fill="#00ff00" opacity="0.6"/>
    
    {/* GPU 3 */}
    <rect x="48" y="36" width="10" height="6" rx="0.5" fill="#2d3548" stroke="#3d4556" strokeWidth="0.5"/>
    <rect x="49" y="37" width="8" height="4" rx="0.3" fill="#1f2533"/>
    <circle cx="51" cy="39" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <circle cx="55" cy="39" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <rect x="49.5" y="37.5" width="2" height="0.5" fill="#00ff00" opacity="0.6"/>
    
    {/* GPU 4 */}
    <rect x="60" y="36" width="10" height="6" rx="0.5" fill="#2d3548" stroke="#3d4556" strokeWidth="0.5"/>
    <rect x="61" y="37" width="8" height="4" rx="0.3" fill="#1f2533"/>
    <circle cx="63" cy="39" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <circle cx="67" cy="39" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <rect x="61.5" y="37.5" width="2" height="0.5" fill="#00ff00" opacity="0.6"/>
    
    {/* GPU Row 2 (bottom) */}
    {/* GPU 5 */}
    <rect x="24" y="44" width="10" height="6" rx="0.5" fill="#2d3548" stroke="#3d4556" strokeWidth="0.5"/>
    <rect x="25" y="45" width="8" height="4" rx="0.3" fill="#1f2533"/>
    <circle cx="27" cy="47" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <circle cx="31" cy="47" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <rect x="25.5" y="45.5" width="2" height="0.5" fill="#00ff00" opacity="0.6"/>
    
    {/* GPU 6 */}
    <rect x="36" y="44" width="10" height="6" rx="0.5" fill="#2d3548" stroke="#3d4556" strokeWidth="0.5"/>
    <rect x="37" y="45" width="8" height="4" rx="0.3" fill="#1f2533"/>
    <circle cx="39" cy="47" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <circle cx="43" cy="47" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <rect x="37.5" y="45.5" width="2" height="0.5" fill="#00ff00" opacity="0.6"/>
    
    {/* GPU 7 */}
    <rect x="48" y="44" width="10" height="6" rx="0.5" fill="#2d3548" stroke="#3d4556" strokeWidth="0.5"/>
    <rect x="49" y="45" width="8" height="4" rx="0.3" fill="#1f2533"/>
    <circle cx="51" cy="47" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <circle cx="55" cy="47" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <rect x="49.5" y="45.5" width="2" height="0.5" fill="#00ff00" opacity="0.6"/>
    
    {/* GPU 8 */}
    <rect x="60" y="44" width="10" height="6" rx="0.5" fill="#2d3548" stroke="#3d4556" strokeWidth="0.5"/>
    <rect x="61" y="45" width="8" height="4" rx="0.3" fill="#1f2533"/>
    <circle cx="63" cy="47" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <circle cx="67" cy="47" r="1.2" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    <rect x="61.5" y="45.5" width="2" height="0.5" fill="#00ff00" opacity="0.6"/>
  </g>
  
  {/* PCIe Risers (industrial-grade) */}
  <g id="risers">
    {/* Riser cables (visible between GPUs and frame) */}
    <path d="M 29 42 L 29 50" stroke="#ff9500" strokeWidth="1.5" opacity="0.4"/>
    <path d="M 41 42 L 41 50" stroke="#ff9500" strokeWidth="1.5" opacity="0.4"/>
    <path d="M 53 42 L 53 50" stroke="#ff9500" strokeWidth="1.5" opacity="0.4"/>
    <path d="M 65 42 L 65 50" stroke="#ff9500" strokeWidth="1.5" opacity="0.4"/>
    
    {/* Riser boards (small PCBs) */}
    <rect x="28" y="50" width="2" height="3" rx="0.2" fill="#00ff00" opacity="0.3"/>
    <rect x="40" y="50" width="2" height="3" rx="0.2" fill="#00ff00" opacity="0.3"/>
    <rect x="52" y="50" width="2" height="3" rx="0.2" fill="#00ff00" opacity="0.3"/>
    <rect x="64" y="50" width="2" height="3" rx="0.2" fill="#00ff00" opacity="0.3"/>
  </g>
  
  {/* Cable management (zip ties) */}
  <g id="cable-management">
    {/* Power cables bundled */}
    <path d="M 22 55 Q 30 55 35 58" stroke="#ff0000" strokeWidth="2" opacity="0.5"/>
    <path d="M 22 55 Q 32 55 40 58" stroke="#ff0000" strokeWidth="2" opacity="0.5"/>
    <path d="M 22 55 Q 34 55 50 58" stroke="#ff0000" strokeWidth="2" opacity="0.5"/>
    
    {/* Zip tie markers */}
    <rect x="34" y="57" width="1.5" height="3" rx="0.3" fill="#1f2533"/>
    <rect x="39" y="57" width="1.5" height="3" rx="0.3" fill="#1f2533"/>
    <rect x="49" y="57" width="1.5" height="3" rx="0.3" fill="#1f2533"/>
  </g>
  
  {/* PSU (Power Supply Unit) */}
  <g id="psu">
    <rect x="24" y="58" width="16" height="6" rx="0.5" fill="#2d3548" stroke="#3d4556" strokeWidth="0.5"/>
    <rect x="25" y="59" width="14" height="4" rx="0.3" fill="#1f2533"/>
    {/* Fan grill */}
    <circle cx="32" cy="61" r="1.5" fill="none" stroke="#4a5568" strokeWidth="0.4"/>
    {/* Label */}
    <rect x="26" y="59.5" width="4" height="1" rx="0.2" fill="#ff9500" opacity="0.6"/>
    <text x="27" y="60.3" fontFamily="monospace" fontSize="0.8" fill="#1f2533">1200W</text>
  </g>
  
  {/* LED strip lighting (orange) */}
  <defs>
    <linearGradient id="ledStrip" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#ff9500;stop-opacity:0.6" />
      <stop offset="50%" style="stop-color:#ff6b00;stop-opacity:0.8" />
      <stop offset="100%" style="stop-color:#ff9500;stop-opacity:0.6" />
    </linearGradient>
    <radialGradient id="ledGlow">
      <stop offset="0%" style="stop-color:#ff9500;stop-opacity:0.3" />
      <stop offset="100%" style="stop-color:#ff9500;stop-opacity:0" />
    </radialGradient>
  </defs>
  <rect x="20" y="30" width="56" height="1" fill="url(#ledStrip)"/>
  <ellipse cx="48" cy="30" rx="35" ry="15" fill="url(#ledGlow)"/>
  
  {/* Efficiency indicator (green status) */}
  <circle cx="72" cy="60" r="1.5" fill="#00ff00" opacity="0.8"/>
  <circle cx="72" cy="60" r="2.5" fill="#00ff00" opacity="0.2"/>
  
  {/* Depth and shadow */}
  <line x1="20" y1="35" x2="20" y2="65" stroke="#1f2533" strokeWidth="0.5" opacity="0.4"/>
  <line x1="76" y1="35" x2="76" y2="65" stroke="#5a6578" strokeWidth="0.3" opacity="0.6"/>
    </svg>
  )
}
