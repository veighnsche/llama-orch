import type { SVGProps } from 'react'

export interface IndustriesHeroProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function IndustriesHero({ size = 1920, className, ...props }: IndustriesHeroProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 1920 600"
      className={className}
      {...props}
    >
      <defs>
    {/* Seamless transition gradients */}
    <linearGradient id="transition1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#0f172a;stop-opacity:1" />
      <stop offset="80%" style="stop-color:#0f172a;stop-opacity:0" />
    </linearGradient>
    <linearGradient id="transition2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="20%" style="stop-color:#0f172a;stop-opacity:0" />
      <stop offset="100%" style="stop-color:#0f172a;stop-opacity:1" />
    </linearGradient>
    
    {/* Teal accent lighting system */}
    <linearGradient id="tealAccent" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" style="stop-color:#14b8a6;stop-opacity:0.4" />
      <stop offset="100%" style="stop-color:#14b8a6;stop-opacity:0" />
    </linearGradient>
    
    <radialGradient id="tealGlow" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#14b8a6;stop-opacity:0.6" />
      <stop offset="100%" style="stop-color:#14b8a6;stop-opacity:0" />
    </radialGradient>
    
    {/* Rim lighting effects */}
    <linearGradient id="rimLight" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#14b8a6;stop-opacity:0" />
      <stop offset="50%" style="stop-color:#14b8a6;stop-opacity:0.3" />
      <stop offset="100%" style="stop-color:#14b8a6;stop-opacity:0" />
    </linearGradient>
    
    {/* Vignette */}
    <radialGradient id="vignette" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#0f172a;stop-opacity:0" />
      <stop offset="100%" style="stop-color:#000000;stop-opacity:0.5" />
    </radialGradient>
    
    {/* Metal gradient for vault */}
    <linearGradient id="metalGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#475569" />
      <stop offset="25%" style="stop-color:#cbd5e1" />
      <stop offset="50%" style="stop-color:#f1f5f9" />
      <stop offset="75%" style="stop-color:#cbd5e1" />
      <stop offset="100%" style="stop-color:#475569" />
    </linearGradient>
    
    {/* Marble gradient */}
    <linearGradient id="marbleGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#f8fafc" />
      <stop offset="50%" style="stop-color:#e2e8f0" />
      <stop offset="100%" style="stop-color:#cbd5e1" />
    </linearGradient>
    
    {/* LED glows */}
    <radialGradient id="ledGreen" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#10b981;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#10b981;stop-opacity:0.4" />
      <stop offset="100%" style="stop-color:#10b981;stop-opacity:0" />
    </radialGradient>
    
    <radialGradient id="ledAmber" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#f59e0b;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#f59e0b;stop-opacity:0.4" />
      <stop offset="100%" style="stop-color:#f59e0b;stop-opacity:0" />
    </radialGradient>
  </defs>
  
  {/* Base background - deep navy */}
  <rect width="1920" height="600" fill="#0f172a"/>
  
  {/* Vignette overlay */}
  <rect width="1920" height="600" fill="url(#vignette)" opacity="0.3"/>
  
  {/* SECTION 1: FINANCIAL SERVICES (0-384px) */}
  <g id="financial">
    {/* Vault wall */}
    <rect x="20" y="80" width="340" height="440" fill="#374151" stroke="#475569" strokeWidth="4"/>
    <rect x="30" y="90" width="320" height="420" fill="#1e293b"/>
    
    {/* Main vault door body */}
    <ellipse cx="190" cy="300" rx="140" ry="140" fill="url(#metalGradient)"/>
    <ellipse cx="190" cy="300" rx="135" ry="135" fill="#64748b" opacity="0.3"/>
    
    {/* Concentric circles */}
    <circle cx="190" cy="300" r="120" fill="none" stroke="#94a3b8" strokeWidth="3"/>
    <circle cx="190" cy="300" r="100" fill="none" stroke="#cbd5e1" strokeWidth="2"/>
    <circle cx="190" cy="300" r="80" fill="none" stroke="#94a3b8" strokeWidth="2"/>
    <circle cx="190" cy="300" r="60" fill="none" stroke="#cbd5e1" strokeWidth="1.5"/>
    
    {/* Spoke wheel mechanism - 8 spokes */}
    <line x1="190" y1="300" x2="190" y2="200" stroke="#475569" strokeWidth="8" strokeLinecap="round"/>
    <line x1="190" y1="300" x2="261" y2="229" stroke="#475569" strokeWidth="8" strokeLinecap="round"/>
    <line x1="190" y1="300" x2="290" y2="300" stroke="#475569" strokeWidth="8" strokeLinecap="round"/>
    <line x1="190" y1="300" x2="261" y2="371" stroke="#475569" strokeWidth="8" strokeLinecap="round"/>
    <line x1="190" y1="300" x2="190" y2="400" stroke="#475569" strokeWidth="8" strokeLinecap="round"/>
    <line x1="190" y1="300" x2="119" y2="371" stroke="#475569" strokeWidth="8" strokeLinecap="round"/>
    <line x1="190" y1="300" x2="90" y2="300" stroke="#475569" strokeWidth="8" strokeLinecap="round"/>
    <line x1="190" y1="300" x2="119" y2="229" stroke="#475569" strokeWidth="8" strokeLinecap="round"/>
    
    {/* Center hub */}
    <circle cx="190" cy="300" r="35" fill="#64748b"/>
    <circle cx="190" cy="300" r="30" fill="#475569"/>
    <circle cx="190" cy="300" r="20" fill="#1e293b"/>
    <circle cx="190" cy="300" r="15" fill="#64748b"/>
    
    {/* Digital keypad on right side */}
    <rect x="300" y="250" width="50" height="100" rx="4" fill="#1e293b" stroke="#14b8a6" strokeWidth="2"/>
    {/* Screen */}
    <rect x="305" y="255" width="40" height="35" rx="2" fill="#0a0f1a"/>
    <text x="325" y="275" fontFamily="monospace" fontSize="10" fill="#14b8a6" textAnchor="middle">****</text>
    {/* Keypad grid */}
    <circle cx="315" cy="305" r="5" fill="#14b8a6" opacity="0.9">
      <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="325" cy="305" r="5" fill="#14b8a6" opacity="0.7"/>
    <circle cx="335" cy="305" r="5" fill="#14b8a6" opacity="0.8"/>
    <circle cx="315" cy="318" r="5" fill="#14b8a6" opacity="0.6"/>
    <circle cx="325" cy="318" r="5" fill="#14b8a6" opacity="0.9"/>
    <circle cx="335" cy="318" r="5" fill="#14b8a6" opacity="0.7"/>
    <circle cx="315" cy="331" r="5" fill="#14b8a6" opacity="0.8"/>
    <circle cx="325" cy="331" r="5" fill="#14b8a6" opacity="0.6"/>
    <circle cx="335" cy="331" r="5" fill="#14b8a6" opacity="0.9"/>
    
    {/* Thick steel wall visible (door partially open) */}
    <rect x="330" y="200" width="30" height="200" fill="#374151"/>
    <rect x="335" y="200" width="20" height="200" fill="#475569"/>
    
    {/* GDPR holographic badge */}
    <g opacity="0.95">
      <rect x="40" y="140" width="90" height="70" rx="8" fill="#14b8a6" fillOpacity="0.15" stroke="#14b8a6" strokeWidth="2.5">
        <animate attributeName="opacity" values="0.8;1;0.8" dur="4s" repeatCount="indefinite"/>
      </rect>
      <text x="85" y="168" fontFamily="sans-serif" fontSize="20" fontWeight="700" fill="#14b8a6" textAnchor="middle">GDPR</text>
      <text x="85" y="192" fontFamily="sans-serif" fontSize="11" fill="#14b8a6" textAnchor="middle" opacity="0.9">COMPLIANT</text>
    </g>
    
    {/* Teal accent lighting at base */}
    <rect x="0" y="480" width="384" height="120" fill="url(#tealAccent)" opacity="0.5"/>
    
    {/* Dramatic side lighting effect */}
    <rect x="0" y="100" width="80" height="400" fill="url(#rimLight)" opacity="0.4"/>
    
    {/* Brushed metal texture simulation */}
    <g opacity="0.15">
      <line x1="160" y1="250" x2="220" y2="250" stroke="#cbd5e1" strokeWidth="0.5"/>
      <line x1="155" y1="270" x2="225" y2="270" stroke="#cbd5e1" strokeWidth="0.5"/>
      <line x1="150" y1="290" x2="230" y2="290" stroke="#cbd5e1" strokeWidth="0.5"/>
      <line x1="155" y1="310" x2="225" y2="310" stroke="#cbd5e1" strokeWidth="0.5"/>
      <line x1="160" y1="330" x2="220" y2="330" stroke="#cbd5e1" strokeWidth="0.5"/>
      <line x1="165" y1="350" x2="215" y2="350" stroke="#cbd5e1" strokeWidth="0.5"/>
    </g>
    
    {/* Transition gradient to next section */}
    <rect x="320" y="0" width="64" height="600" fill="url(#transition1)"/>
  </g>
  
  {/* SECTION 2: HEALTHCARE (384-672px) */}
  <g id="healthcare">
    {/* Clean room background */}
    <rect x="384" y="0" width="288" height="600" fill="#0a0f1a"/>
    
    {/* White tile floor with reflections */}
    <g opacity="0.6">
      <rect x="400" y="500" width="50" height="50" fill="#e2e8f0" opacity="0.3"/>
      <rect x="450" y="500" width="50" height="50" fill="#f1f5f9" opacity="0.2"/>
      <rect x="500" y="500" width="50" height="50" fill="#e2e8f0" opacity="0.3"/>
      <rect x="550" y="500" width="50" height="50" fill="#f1f5f9" opacity="0.2"/>
      <rect x="600" y="500" width="50" height="50" fill="#e2e8f0" opacity="0.3"/>
      <rect x="400" y="550" width="50" height="50" fill="#f1f5f9" opacity="0.2"/>
      <rect x="450" y="550" width="50" height="50" fill="#e2e8f0" opacity="0.3"/>
      <rect x="500" y="550" width="50" height="50" fill="#f1f5f9" opacity="0.2"/>
      <rect x="550" y="550" width="50" height="50" fill="#e2e8f0" opacity="0.3"/>
      <rect x="600" y="550" width="50" height="50" fill="#f1f5f9" opacity="0.2"/>
    </g>
    
    {/* Server rack frame */}
    <rect x="420" y="60" width="230" height="460" fill="#1e293b" stroke="#334155" strokeWidth="4"/>
    <rect x="425" y="65" width="220" height="450" fill="#0a0f1a"/>
    
    {/* Blade servers with detailed LEDs */}
    {/* Server 1 */}
    <rect x="435" y="80" width="200" height="32" fill="#374151" stroke="#475569" strokeWidth="1.5" rx="2"/>
    <circle cx="620" cy="96" r="4" fill="url(#ledGreen)">
      <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="608" cy="96" r="3" fill="url(#ledAmber)"/>
    <circle cx="596" cy="96" r="3" fill="url(#ledGreen)"/>
    
    {/* Server 2 */}
    <rect x="435" y="120" width="200" height="32" fill="#374151" stroke="#475569" strokeWidth="1.5" rx="2"/>
    <circle cx="620" cy="136" r="4" fill="url(#ledGreen)">
      <animate attributeName="opacity" values="1;0.5;1" dur="1.8s" repeatCount="indefinite"/>
    </circle>
    <circle cx="608" cy="136" r="3" fill="url(#ledGreen)"/>
    <circle cx="596" cy="136" r="3" fill="url(#ledAmber)"/>
    
    {/* Server 3 */}
    <rect x="435" y="160" width="200" height="32" fill="#374151" stroke="#475569" strokeWidth="1.5" rx="2"/>
    <circle cx="620" cy="176" r="4" fill="url(#ledGreen)"/>
    <circle cx="608" cy="176" r="3" fill="url(#ledAmber)">
      <animate attributeName="opacity" values="0.7;1;0.7" dur="2.2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="596" cy="176" r="3" fill="url(#ledGreen)"/>
    
    {/* Server 4 */}
    <rect x="435" y="200" width="200" height="32" fill="#374151" stroke="#475569" strokeWidth="1.5" rx="2"/>
    <circle cx="620" cy="216" r="4" fill="url(#ledGreen)">
      <animate attributeName="opacity" values="0.8;1;0.8" dur="1.5s" repeatCount="indefinite"/>
    </circle>
    <circle cx="608" cy="216" r="3" fill="url(#ledGreen)"/>
    <circle cx="596" cy="216" r="3" fill="url(#ledGreen)"/>
    
    {/* Server 5 */}
    <rect x="435" y="240" width="200" height="32" fill="#374151" stroke="#475569" strokeWidth="1.5" rx="2"/>
    <circle cx="620" cy="256" r="4" fill="url(#ledGreen)"/>
    <circle cx="608" cy="256" r="3" fill="url(#ledAmber)"/>
    <circle cx="596" cy="256" r="3" fill="url(#ledGreen)">
      <animate attributeName="opacity" values="0.6;1;0.6" dur="2.8s" repeatCount="indefinite"/>
    </circle>
    
    {/* Server 6 */}
    <rect x="435" y="280" width="200" height="32" fill="#374151" stroke="#475569" strokeWidth="1.5" rx="2"/>
    <circle cx="620" cy="296" r="4" fill="url(#ledGreen)">
      <animate attributeName="opacity" values="0.6;1;0.6" dur="2.5s" repeatCount="indefinite"/>
    </circle>
    <circle cx="608" cy="296" r="3" fill="url(#ledGreen)"/>
    <circle cx="596" cy="296" r="3" fill="url(#ledAmber)"/>
    
    {/* Server 7 */}
    <rect x="435" y="320" width="200" height="32" fill="#374151" stroke="#475569" strokeWidth="1.5" rx="2"/>
    <circle cx="620" cy="336" r="4" fill="url(#ledGreen)"/>
    <circle cx="608" cy="336" r="3" fill="url(#ledAmber)">
      <animate attributeName="opacity" values="0.5;1;0.5" dur="1.7s" repeatCount="indefinite"/>
    </circle>
    <circle cx="596" cy="336" r="3" fill="url(#ledGreen)"/>
    
    {/* Server 8 */}
    <rect x="435" y="360" width="200" height="32" fill="#374151" stroke="#475569" strokeWidth="1.5" rx="2"/>
    <circle cx="620" cy="376" r="4" fill="url(#ledGreen)">
      <animate attributeName="opacity" values="1;0.7;1" dur="2.3s" repeatCount="indefinite"/>
    </circle>
    <circle cx="608" cy="376" r="3" fill="url(#ledGreen)"/>
    <circle cx="596" cy="376" r="3" fill="url(#ledGreen)"/>
    
    {/* Server 9 */}
    <rect x="435" y="400" width="200" height="32" fill="#374151" stroke="#475569" strokeWidth="1.5" rx="2"/>
    <circle cx="620" cy="416" r="4" fill="url(#ledGreen)"/>
    <circle cx="608" cy="416" r="3" fill="url(#ledAmber)"/>
    <circle cx="596" cy="416" r="3" fill="url(#ledGreen)"/>
    
    {/* Server 10 */}
    <rect x="435" y="440" width="200" height="32" fill="#374151" stroke="#475569" strokeWidth="1.5" rx="2"/>
    <circle cx="620" cy="456" r="4" fill="url(#ledGreen)">
      <animate attributeName="opacity" values="0.7;1;0.7" dur="1.9s" repeatCount="indefinite"/>
    </circle>
    <circle cx="608" cy="456" r="3" fill="url(#ledGreen)"/>
    <circle cx="596" cy="456" r="3" fill="url(#ledAmber)"/>
    
    {/* HIPAA COMPLIANT label */}
    <rect x="440" y="100" width="130" height="30" fill="#3b82f6" rx="3"/>
    <text x="505" y="120" fontFamily="sans-serif" fontSize="13" fontWeight="700" fill="#ffffff" textAnchor="middle">HIPAA COMPLIANT</text>
    
    {/* PHI ENCRYPTED label */}
    <rect x="440" y="260" width="130" height="30" fill="#14b8a6" rx="3"/>
    <text x="505" y="280" fontFamily="sans-serif" fontSize="12" fontWeight="700" fill="#ffffff" textAnchor="middle">PHI ENCRYPTED</text>
    
    {/* Medical cross integration (subtle) */}
    <g opacity="0.25">
      <rect x="458" y="370" width="12" height="40" fill="#14b8a6"/>
      <rect x="446" y="382" width="36" height="16" fill="#14b8a6"/>
    </g>
    
    {/* Cable management overhead */}
    <line x1="430" y1="55" x2="640" y2="55" stroke="#475569" strokeWidth="3"/>
    <line x1="445" y1="55" x2="445" y2="80" stroke="#475569" strokeWidth="2"/>
    <line x1="485" y1="55" x2="485" y2="80" stroke="#475569" strokeWidth="2"/>
    <line x1="525" y1="55" x2="525" y2="80" stroke="#475569" strokeWidth="2"/>
    <line x1="565" y1="55" x2="565" y2="80" stroke="#475569" strokeWidth="2"/>
    <line x1="605" y1="55" x2="605" y2="80" stroke="#475569" strokeWidth="2"/>
    
    {/* Blue LED floor strip lighting */}
    <rect x="420" y="525" width="230" height="12" fill="#3b82f6" opacity="0.7" rx="2">
      <animate attributeName="opacity" values="0.4;0.9;0.4" dur="3s" repeatCount="indefinite"/>
    </rect>
    
    {/* Teal accent lighting */}
    <rect x="384" y="480" width="288" height="120" fill="url(#tealAccent)" opacity="0.3"/>
    
    {/* Transition to next section */}
    <rect x="608" y="0" width="64" height="600" fill="url(#transition1)"/>
  </g>
  
  {/* SECTION 3: LEGAL (672-960px) */}
  <g id="legal">
    <rect x="672" y="0" width="288" height="600" fill="#1e293b"/>
    
    {/* Marble floor with geometric inlay pattern */}
    <g opacity="0.8">
      {/* Checkerboard pattern */}
      <rect x="690" y="480" width="55" height="60" fill="#cbd5e1"/>
      <rect x="745" y="480" width="55" height="60" fill="#f1f5f9"/>
      <rect x="800" y="480" width="55" height="60" fill="#cbd5e1"/>
      <rect x="855" y="480" width="55" height="60" fill="#f1f5f9"/>
      <rect x="690" y="540" width="55" height="60" fill="#f1f5f9"/>
      <rect x="745" y="540" width="55" height="60" fill="#cbd5e1"/>
      <rect x="800" y="540" width="55" height="60" fill="#f1f5f9"/>
      <rect x="855" y="540" width="55" height="60" fill="#cbd5e1"/>
      
      {/* Geometric diamond accents */}
      <path d="M 767 510 L 777 500 L 787 510 L 777 520 Z" fill="#94a3b8" opacity="0.6"/>
      <path d="M 877 510 L 887 500 L 897 510 L 887 520 Z" fill="#94a3b8" opacity="0.6"/>
    </g>
    
    {/* Towering Corinthian columns (30 feet tall) */}
    {/* Column 1 */}
    <g>
      <rect x="710" y="80" width="45" height="400" fill="url(#marbleGradient)" stroke="#cbd5e1" strokeWidth="2"/>
      {/* Fluting detail */}
      <line x1="720" y1="80" x2="720" y2="480" stroke="#94a3b8" strokeWidth="1" opacity="0.4"/>
      <line x1="732.5" y1="80" x2="732.5" y2="480" stroke="#94a3b8" strokeWidth="1" opacity="0.4"/>
      <line x1="745" y1="80" x2="745" y2="480" stroke="#94a3b8" strokeWidth="1" opacity="0.4"/>
      {/* Capital (Corinthian style) */}
      <rect x="700" y="65" width="65" height="25" fill="#f8fafc" stroke="#cbd5e1" strokeWidth="1"/>
      <ellipse cx="732.5" cy="60" rx="38" ry="10" fill="#f1f5f9"/>
      {/* Acanthus leaf detail on capital */}
      <path d="M 710 75 Q 720 65 732.5 75 Q 745 65 755 75" fill="none" stroke="#94a3b8" strokeWidth="1.5"/>
      {/* Base */}
      <rect x="700" y="475" width="65" height="15" fill="#f8fafc"/>
      <rect x="705" y="490" width="55" height="10" fill="#e2e8f0"/>
    </g>
    
    {/* Column 2 */}
    <g>
      <rect x="795" y="80" width="45" height="400" fill="url(#marbleGradient)" stroke="#cbd5e1" strokeWidth="2"/>
      <line x1="805" y1="80" x2="805" y2="480" stroke="#94a3b8" strokeWidth="1" opacity="0.4"/>
      <line x1="817.5" y1="80" x2="817.5" y2="480" stroke="#94a3b8" strokeWidth="1" opacity="0.4"/>
      <line x1="830" y1="80" x2="830" y2="480" stroke="#94a3b8" strokeWidth="1" opacity="0.4"/>
      <rect x="785" y="65" width="65" height="25" fill="#f8fafc" stroke="#cbd5e1" strokeWidth="1"/>
      <ellipse cx="817.5" cy="60" rx="38" ry="10" fill="#f1f5f9"/>
      <path d="M 795 75 Q 805 65 817.5 75 Q 830 65 840 75" fill="none" stroke="#94a3b8" strokeWidth="1.5"/>
      <rect x="785" y="475" width="65" height="15" fill="#f8fafc"/>
      <rect x="790" y="490" width="55" height="10" fill="#e2e8f0"/>
    </g>
    
    {/* Column 3 */}
    <g>
      <rect x="880" y="80" width="45" height="400" fill="url(#marbleGradient)" stroke="#cbd5e1" strokeWidth="2"/>
      <line x1="890" y1="80" x2="890" y2="480" stroke="#94a3b8" strokeWidth="1" opacity="0.4"/>
      <line x1="902.5" y1="80" x2="902.5" y2="480" stroke="#94a3b8" strokeWidth="1" opacity="0.4"/>
      <line x1="915" y1="80" x2="915" y2="480" stroke="#94a3b8" strokeWidth="1" opacity="0.4"/>
      <rect x="870" y="65" width="65" height="25" fill="#f8fafc" stroke="#cbd5e1" strokeWidth="1"/>
      <ellipse cx="902.5" cy="60" rx="38" ry="10" fill="#f1f5f9"/>
      <path d="M 880 75 Q 890 65 902.5 75 Q 915 65 925 75" fill="none" stroke="#94a3b8" strokeWidth="1.5"/>
      <rect x="870" y="475" width="65" height="15" fill="#f8fafc"/>
      <rect x="875" y="490" width="55" height="10" fill="#e2e8f0"/>
    </g>
    
    {/* Bronze scales of justice statue (4 feet tall, foreground) */}
    <g opacity="0.9">
      {/* Stand/pedestal */}
      <rect x="752" y="380" width="56" height="100" fill="#78350f" stroke="#92400e" strokeWidth="2"/>
      <ellipse cx="780" cy="380" rx="30" ry="8" fill="#92400e"/>
      
      {/* Central pole */}
      <rect x="775" y="280" width="10" height="100" fill="#a16207" stroke="#92400e" strokeWidth="1.5"/>
      
      {/* Balance beam */}
      <rect x="720" y="275" width="120" height="8" fill="#a16207" stroke="#92400e" strokeWidth="1.5" rx="2"/>
      
      {/* Left scale */}
      <line x1="740" y1="279" x2="740" y2="310" stroke="#92400e" strokeWidth="2"/>
      <ellipse cx="740" cy="315" rx="20" ry="6" fill="#a16207" stroke="#92400e" strokeWidth="1.5"/>
      <path d="M 720 315 Q 720 325 740 330 Q 760 325 760 315" fill="#78350f" stroke="#92400e" strokeWidth="1.5"/>
      
      {/* Right scale */}
      <line x1="820" y1="279" x2="820" y2="310" stroke="#92400e" strokeWidth="2"/>
      <ellipse cx="820" cy="315" rx="20" ry="6" fill="#a16207" stroke="#92400e" strokeWidth="1.5"/>
      <path d="M 800 315 Q 800 325 820 330 Q 840 325 840 315" fill="#78350f" stroke="#92400e" strokeWidth="1.5"/>
      
      {/* Top finial */}
      <circle cx="780" cy="270" r="8" fill="#a16207" stroke="#92400e" strokeWidth="1.5"/>
    </g>
    
    {/* Dramatic uplighting on columns */}
    <rect x="700" y="400" width="65" height="80" fill="url(#tealAccent)" opacity="0.5"/>
    <rect x="785" y="400" width="65" height="80" fill="url(#tealAccent)" opacity="0.5"/>
    <rect x="870" y="400" width="65" height="80" fill="url(#tealAccent)" opacity="0.5"/>
    
    {/* Teal accent along column bases */}
    <rect x="700" y="485" width="65" height="5" fill="#14b8a6" opacity="0.6">
      <animate attributeName="opacity" values="0.4;0.8;0.4" dur="4s" repeatCount="indefinite"/>
    </rect>
    <rect x="785" y="485" width="65" height="5" fill="#14b8a6" opacity="0.6">
      <animate attributeName="opacity" values="0.6;0.8;0.6" dur="4s" repeatCount="indefinite"/>
    </rect>
    <rect x="870" y="485" width="65" height="5" fill="#14b8a6" opacity="0.6">
      <animate attributeName="opacity" values="0.5;0.8;0.5" dur="4s" repeatCount="indefinite"/>
    </rect>
    
    {/* Marble veining texture */}
    <g opacity="0.2">
      <path d="M 715 150 Q 720 200 718 250 Q 715 300 720 350" stroke="#64748b" strokeWidth="1" fill="none"/>
      <path d="M 800 180 Q 805 230 803 280 Q 800 330 805 380" stroke="#64748b" strokeWidth="1" fill="none"/>
      <path d="M 885 120 Q 890 170 888 220 Q 885 270 890 320" stroke="#64748b" strokeWidth="1" fill="none"/>
    </g>
    
    {/* Transition to next section */}
    <rect x="896" y="0" width="64" height="600" fill="url(#transition1)"/>
  </g>
  
  {/* SECTION 4: GOVERNMENT (960-1248px) */}
  <g id="government">
    {/* Dark navy wall */}
    <rect x="960" y="0" width="288" height="600" fill="#0c1629"/>
    
    {/* Official government seal (6 feet diameter) mounted on wall */}
    <g transform="translate(1104, 200)">
      {/* Outer ring */}
      <circle cx="0" cy="0" r="100" fill="#1e293b" stroke="#d4af37" strokeWidth="4"/>
      <circle cx="0" cy="0" r="95" fill="#334155" stroke="#d4af37" strokeWidth="2"/>
      
      {/* Inner decorative ring */}
      <circle cx="0" cy="0" r="80" fill="none" stroke="#d4af37" strokeWidth="2.5"/>
      
      {/* Eagle with shield */}
      {/* Shield */}
      <rect x="-25" y="-20" width="50" height="60" fill="#1e40af" stroke="#d4af37" strokeWidth="2" rx="3"/>
      <rect x="-25" y="-20" width="50" height="15" fill="#dc2626"/>
      <rect x="-25" y="-5" width="50" height="15" fill="#f1f5f9"/>
      <rect x="-25" y="10" width="50" height="15" fill="#dc2626"/>
      <rect x="-25" y="25" width="50" height="15" fill="#f1f5f9"/>
      
      {/* Eagle head */}
      <circle cx="0" cy="-35" r="15" fill="#d4af37"/>
      <path d="M -5 -35 L 0 -45 L 5 -35" fill="#d4af37"/>
      
      {/* Eagle wings */}
      <ellipse cx="-35" cy="-15" rx="25" ry="15" fill="#d4af37" opacity="0.9"/>
      <ellipse cx="35" cy="-15" rx="25" ry="15" fill="#d4af37" opacity="0.9"/>
      
      {/* Olive branches (left) */}
      <path d="M -60 20 Q -50 15 -45 25 Q -40 35 -35 30" stroke="#10b981" strokeWidth="3" fill="none"/>
      <ellipse cx="-52" cy="18" rx="4" ry="6" fill="#10b981"/>
      <ellipse cx="-48" cy="28" rx="4" ry="6" fill="#10b981"/>
      <ellipse cx="-42" cy="32" rx="4" ry="6" fill="#10b981"/>
      
      {/* Arrows (right) */}
      <line x1="35" y1="30" x2="60" y2="20" stroke="#d4af37" strokeWidth="3"/>
      <path d="M 60 20 L 55 17 L 57 22 Z" fill="#d4af37"/>
      <line x1="35" y1="25" x2="60" y2="15" stroke="#d4af37" strokeWidth="3"/>
      <path d="M 60 15 L 55 12 L 57 17 Z" fill="#d4af37"/>
      
      {/* E PLURIBUS UNUM banner */}
      <path d="M -60 50 Q 0 45 60 50" stroke="#f1f5f9" strokeWidth="8" fill="none"/>
      <text x="0" y="55" fontFamily="serif" fontSize="9" fontWeight="700" fill="#1e293b" textAnchor="middle">E PLURIBUS UNUM</text>
      
      {/* Gold metallic finish effect */}
      <circle cx="-30" cy="-40" r="3" fill="#fef3c7" opacity="0.8"/>
      <circle cx="25" cy="-25" r="2" fill="#fef3c7" opacity="0.6"/>
    </g>
    
    {/* Security clearance badge (holographic, floating) */}
    <g transform="translate(1000, 120)" opacity="0.95">
      <rect x="0" y="0" width="90" height="120" rx="6" fill="#1e293b" stroke="#dc2626" strokeWidth="3">
        <animate attributeName="stroke" values="#dc2626;#3b82f6;#dc2626" dur="3s" repeatCount="indefinite"/>
      </rect>
      {/* Holographic shimmer */}
      <rect x="5" y="5" width="80" height="110" rx="4" fill="url(#tealGlow)" opacity="0.3">
        <animate attributeName="opacity" values="0.2;0.5;0.2" dur="2s" repeatCount="indefinite"/>
      </rect>
      
      <text x="45" y="35" fontFamily="sans-serif" fontSize="16" fontWeight="900" fill="#dc2626" textAnchor="middle">TOP</text>
      <text x="45" y="55" fontFamily="sans-serif" fontSize="16" fontWeight="900" fill="#dc2626" textAnchor="middle">SECRET</text>
      <rect x="15" y="65" width="60" height="2" fill="#dc2626"/>
      <text x="45" y="85" fontFamily="sans-serif" fontSize="10" fontWeight="700" fill="#14b8a6" textAnchor="middle">ITAR</text>
      <text x="45" y="100" fontFamily="sans-serif" fontSize="10" fontWeight="700" fill="#14b8a6" textAnchor="middle">COMPLIANT</text>
    </g>
    
    {/* Additional security badge */}
    <g transform="translate(1158, 430)" opacity="0.9">
      <rect x="0" y="0" width="80" height="60" rx="4" fill="#1e293b" stroke="#3b82f6" strokeWidth="2.5"/>
      <text x="40" y="25" fontFamily="sans-serif" fontSize="11" fontWeight="700" fill="#3b82f6" textAnchor="middle">SECURITY</text>
      <text x="40" y="42" fontFamily="sans-serif" fontSize="11" fontWeight="700" fill="#3b82f6" textAnchor="middle">CLEARANCE</text>
      <text x="40" y="55" fontFamily="monospace" fontSize="9" fill="#64748b" textAnchor="middle">L5-DELTA</text>
    </g>
    
    {/* Patriotic red/blue lighting mixed with teal */}
    {/* Red accent */}
    <ellipse cx="1000" cy="100" rx="80" ry="100" fill="#dc2626" opacity="0.15"/>
    <ellipse cx="1240" cy="500" rx="60" ry="80" fill="#dc2626" opacity="0.12"/>
    
    {/* Blue accent */}
    <ellipse cx="1100" cy="500" rx="100" ry="80" fill="#3b82f6" opacity="0.15"/>
    <ellipse cx="1200" cy="150" rx="70" ry="90" fill="#3b82f6" opacity="0.12"/>
    
    {/* Teal brand accent */}
    <rect x="960" y="480" width="288" height="120" fill="url(#tealAccent)" opacity="0.4"/>
    
    {/* Wall texture detail */}
    <g opacity="0.1">
      <rect x="980" y="100" width="250" height="1" fill="#64748b"/>
      <rect x="980" y="200" width="250" height="1" fill="#64748b"/>
      <rect x="980" y="300" width="250" height="1" fill="#64748b"/>
      <rect x="980" y="400" width="250" height="1" fill="#64748b"/>
    </g>
    
    {/* Transition to next section */}
    <rect x="1184" y="0" width="64" height="600" fill="url(#transition1)"/>
  </g>
  
  {/* SECTION 5: EDUCATION (1248-1536px) */}
  <g id="education">
    {/* Minimalist white walls */}
    <rect x="1248" y="0" width="288" height="600" fill="#f1f5f9"/>
    
    {/* Light wood furniture (desks) */}
    <rect x="1268" y="400" width="120" height="180" fill="#d4a574" rx="4"/>
    <rect x="1268" y="405" width="120" height="8" fill="#b8935f"/>
    <rect x="1408" y="400" width="120" height="180" fill="#d4a574" rx="4"/>
    <rect x="1408" y="405" width="120" height="8" fill="#b8935f"/>
    
    {/* Modern monitors with encrypted login screens */}
    {/* Monitor 1 */}
    <g transform="translate(1288, 180)">
      <rect x="0" y="0" width="80" height="60" rx="3" fill="#0a0f1a" stroke="#334155" strokeWidth="3"/>
      <rect x="3" y="3" width="74" height="50" fill="#1e293b"/>
      {/* Login screen */}
      <rect x="10" y="10" width="60" height="30" fill="#0f172a" stroke="#14b8a6" strokeWidth="1" rx="2"/>
      {/* Padlock icon */}
      <rect x="35" y="23" width="10" height="8" fill="#14b8a6" rx="1"/>
      <path d="M 37 23 Q 37 18 40 18 Q 43 18 43 23" stroke="#14b8a6" strokeWidth="2" fill="none"/>
      <text x="40" y="50" fontFamily="monospace" fontSize="7" fill="#64748b" textAnchor="middle">ENCRYPTED</text>
      {/* FERPA watermark */}
      <text x="40" y="28" fontFamily="sans-serif" fontSize="6" fill="#14b8a6" textAnchor="middle" opacity="0.5">FERPA</text>
      {/* Stand */}
      <rect x="35" y="60" width="10" height="20" fill="#334155"/>
      <rect x="25" y="80" width="30" height="5" fill="#334155"/>
    </g>
    
    {/* Monitor 2 */}
    <g transform="translate(1408, 180)">
      <rect x="0" y="0" width="80" height="60" rx="3" fill="#0a0f1a" stroke="#334155" strokeWidth="3"/>
      <rect x="3" y="3" width="74" height="50" fill="#1e293b"/>
      <rect x="10" y="10" width="60" height="30" fill="#0f172a" stroke="#14b8a6" strokeWidth="1" rx="2"/>
      <rect x="35" y="23" width="10" height="8" fill="#14b8a6" rx="1"/>
      <path d="M 37 23 Q 37 18 40 18 Q 43 18 43 23" stroke="#14b8a6" strokeWidth="2" fill="none"/>
      <text x="40" y="50" fontFamily="monospace" fontSize="7" fill="#64748b" textAnchor="middle">PROTECTED</text>
      <text x="40" y="28" fontFamily="sans-serif" fontSize="6" fill="#14b8a6" textAnchor="middle" opacity="0.5">FERPA</text>
      <rect x="35" y="60" width="10" height="20" fill="#334155"/>
      <rect x="25" y="80" width="30" height="5" fill="#334155"/>
    </g>
    
    {/* Monitor 3 */}
    <g transform="translate(1288, 320)">
      <rect x="0" y="0" width="80" height="60" rx="3" fill="#0a0f1a" stroke="#334155" strokeWidth="3"/>
      <rect x="3" y="3" width="74" height="50" fill="#1e293b"/>
      <rect x="10" y="10" width="60" height="30" fill="#0f172a" stroke="#14b8a6" strokeWidth="1" rx="2"/>
      <rect x="35" y="23" width="10" height="8" fill="#14b8a6" rx="1"/>
      <path d="M 37 23 Q 37 18 40 18 Q 43 18 43 23" stroke="#14b8a6" strokeWidth="2" fill="none"/>
      <text x="40" y="50" fontFamily="monospace" fontSize="7" fill="#64748b" textAnchor="middle">SECURE</text>
      <text x="40" y="28" fontFamily="sans-serif" fontSize="6" fill="#14b8a6" textAnchor="middle" opacity="0.5">FERPA</text>
      <rect x="35" y="60" width="10" height="20" fill="#334155"/>
      <rect x="25" y="80" width="30" height="5" fill="#334155"/>
    </g>
    
    {/* Monitor 4 */}
    <g transform="translate(1408, 320)">
      <rect x="0" y="0" width="80" height="60" rx="3" fill="#0a0f1a" stroke="#334155" strokeWidth="3"/>
      <rect x="3" y="3" width="74" height="50" fill="#1e293b"/>
      <rect x="10" y="10" width="60" height="30" fill="#0f172a" stroke="#14b8a6" strokeWidth="1" rx="2"/>
      <rect x="35" y="23" width="10" height="8" fill="#14b8a6" rx="1"/>
      <path d="M 37 23 Q 37 18 40 18 Q 43 18 43 23" stroke="#14b8a6" strokeWidth="2" fill="none"/>
      <text x="40" y="50" fontFamily="monospace" fontSize="7" fill="#64748b" textAnchor="middle">LOCKED</text>
      <text x="40" y="28" fontFamily="sans-serif" fontSize="6" fill="#14b8a6" textAnchor="middle" opacity="0.5">FERPA</text>
      <rect x="35" y="60" width="10" height="20" fill="#334155"/>
      <rect x="25" y="80" width="30" height="5" fill="#334155"/>
    </g>
    
    {/* Privacy shields on screens (subtle overlay) */}
    <rect x="1291" y="190" width="74" height="50" fill="#0f172a" opacity="0.3"/>
    <rect x="1411" y="190" width="74" height="50" fill="#0f172a" opacity="0.3"/>
    <rect x="1291" y="330" width="74" height="50" fill="#0f172a" opacity="0.3"/>
    <rect x="1411" y="330" width="74" height="50" fill="#0f172a" opacity="0.3"/>
    
    {/* Teal accent lighting under desks */}
    <rect x="1268" y="575" width="120" height="15" fill="#14b8a6" opacity="0.5" rx="2">
      <animate attributeName="opacity" values="0.3;0.6;0.3" dur="3s" repeatCount="indefinite"/>
    </rect>
    <rect x="1408" y="575" width="120" height="15" fill="#14b8a6" opacity="0.5" rx="2">
      <animate attributeName="opacity" values="0.5;0.7;0.5" dur="3.5s" repeatCount="indefinite"/>
    </rect>
    
    {/* FERPA PROTECTED label */}
    <g transform="translate(1320, 100)">
      <rect x="0" y="0" width="140" height="50" rx="6" fill="#10b981" opacity="0.9"/>
      <text x="70" y="25" fontFamily="sans-serif" fontSize="16" fontWeight="700" fill="#ffffff" textAnchor="middle">FERPA</text>
      <text x="70" y="42" fontFamily="sans-serif" fontSize="11" fill="#ffffff" textAnchor="middle">PROTECTED</text>
    </g>
    
    {/* Contemporary educational tech aesthetic */}
    <g opacity="0.2">
      <line x1="1268" y1="160" x2="1528" y2="160" stroke="#cbd5e1" strokeWidth="2"/>
      <circle cx="1280" cy="160" r="4" fill="#14b8a6"/>
      <circle cx="1380" cy="160" r="4" fill="#14b8a6"/>
      <circle cx="1480" cy="160" r="4" fill="#14b8a6"/>
    </g>
    
    {/* Minimalist wall detail */}
    <g opacity="0.1">
      <rect x="1260" y="50" width="2" height="500" fill="#cbd5e1"/>
      <rect x="1520" y="50" width="2" height="500" fill="#cbd5e1"/>
    </g>
    
    {/* Transition to next section */}
    <rect x="1472" y="0" width="64" height="600" fill="url(#transition1)"/>
  </g>
  
  {/* SECTION 6: MANUFACTURING (1536-1920px) */}
  <g id="manufacturing">
    {/* Industrial factory floor background */}
    <rect x="1536" y="0" width="384" height="600" fill="#0f172a"/>
    
    {/* Industrial metal floor with grating */}
    <g opacity="0.3">
      <rect x="1550" y="500" width="350" height="100" fill="#1e293b"/>
      <line x1="1550" y1="520" x2="1900" y2="520" stroke="#475569" strokeWidth="1"/>
      <line x1="1550" y1="540" x2="1900" y2="540" stroke="#475569" strokeWidth="1"/>
      <line x1="1550" y1="560" x2="1900" y2="560" stroke="#475569" strokeWidth="1"/>
      <line x1="1550" y1="580" x2="1900" y2="580" stroke="#475569" strokeWidth="1"/>
      <line x1="1580" y1="500" x2="1580" y2="600" stroke="#475569" strokeWidth="0.5"/>
      <line x1="1620" y1="500" x2="1620" y2="600" stroke="#475569" strokeWidth="0.5"/>
      <line x1="1660" y1="500" x2="1660" y2="600" stroke="#475569" strokeWidth="0.5"/>
      <line x1="1700" y1="500" x2="1700" y2="600" stroke="#475569" strokeWidth="0.5"/>
      <line x1="1740" y1="500" x2="1740" y2="600" stroke="#475569" strokeWidth="0.5"/>
      <line x1="1780" y1="500" x2="1780" y2="600" stroke="#475569" strokeWidth="0.5"/>
      <line x1="1820" y1="500" x2="1820" y2="600" stroke="#475569" strokeWidth="0.5"/>
      <line x1="1860" y1="500" x2="1860" y2="600" stroke="#475569" strokeWidth="0.5"/>
    </g>
    
    {/* Precision assembly line */}
    <rect x="1560" y="350" width="340" height="120" fill="#374151" stroke="#64748b" strokeWidth="3" rx="4"/>
    <rect x="1565" y="355" width="330" height="110" fill="#1e293b"/>
    
    {/* Conveyor belt with motion */}
    <rect x="1570" y="380" width="320" height="50" fill="#475569" rx="2"/>
    {/* Conveyor segments (motion implied) */}
    <rect x="1580" y="390" width="40" height="30" fill="#64748b" opacity="0.8"/>
    <rect x="1640" y="390" width="40" height="30" fill="#64748b" opacity="0.6"/>
    <rect x="1700" y="390" width="40" height="30" fill="#64748b" opacity="0.7"/>
    <rect x="1760" y="390" width="40" height="30" fill="#64748b" opacity="0.5"/>
    <rect x="1820" y="390" width="40" height="30" fill="#64748b" opacity="0.9"/>
    
    {/* Robotic arms with motion blur effect */}
    {/* Robot arm 1 */}
    <g opacity="0.9">
      <rect x="1590" y="280" width="15" height="80" fill="#64748b" rx="2"/>
      <circle cx="1597.5" cy="280" r="12" fill="#475569" stroke="#94a3b8" strokeWidth="2"/>
      {/* Arm segments with blur */}
      <rect x="1590" y="240" width="15" height="40" fill="#94a3b8" opacity="0.7"/>
      <rect x="1588" y="238" width="19" height="44" fill="#94a3b8" opacity="0.3"/>
      {/* Gripper */}
      <rect x="1585" y="230" width="8" height="15" fill="#cbd5e1"/>
      <rect x="1600" y="230" width="8" height="15" fill="#cbd5e1"/>
    </g>
    
    {/* Robot arm 2 */}
    <g opacity="0.85">
      <rect x="1750" y="260" width="15" height="100" fill="#64748b" rx="2"/>
      <circle cx="1757.5" cy="260" r="12" fill="#475569" stroke="#94a3b8" strokeWidth="2"/>
      <rect x="1750" y="210" width="15" height="50" fill="#94a3b8" opacity="0.6"/>
      <rect x="1748" y="208" width="19" height="54" fill="#94a3b8" opacity="0.25"/>
      <rect x="1745" y="200" width="8" height="15" fill="#cbd5e1"/>
      <rect x="1760" y="200" width="8" height="15" fill="#cbd5e1"/>
    </g>
    
    {/* INTELLECTUAL PROPERTY PROTECTED signs with shield icons */}
    <g transform="translate(1580, 80)">
      <rect x="0" y="0" width="300" height="80" fill="#f59e0b" opacity="0.9" stroke="#fbbf24" strokeWidth="3" rx="4"/>
      {/* Warning stripes */}
      <rect x="0" y="0" width="20" height="80" fill="#000000" opacity="0.3"/>
      <rect x="280" y="0" width="20" height="80" fill="#000000" opacity="0.3"/>
      {/* Shield icon */}
      <path d="M 50 20 L 35 27 L 35 50 L 50 60 L 65 50 L 65 27 Z" fill="#1e293b" stroke="#fbbf24" strokeWidth="2"/>
      <text x="50" y="45" fontFamily="sans-serif" fontSize="18" fill="#fbbf24" textAnchor="middle" fontWeight="bold">IP</text>
      {/* Text */}
      <text x="150" y="35" fontFamily="sans-serif" fontSize="18" fontWeight="900" fill="#1e293b" textAnchor="middle">INTELLECTUAL</text>
      <text x="150" y="55" fontFamily="sans-serif" fontSize="18" fontWeight="900" fill="#1e293b" textAnchor="middle">PROPERTY</text>
      <text x="150" y="72" fontFamily="sans-serif" fontSize="14" fontWeight="700" fill="#1e293b" textAnchor="middle">PROTECTED</text>
    </g>
    
    {/* CAD workstation monitors with 3D models */}
    <g transform="translate(1720, 180)">
      <rect x="0" y="0" width="160" height="120" rx="4" fill="#0a0f1a" stroke="#64748b" strokeWidth="3"/>
      <rect x="5" y="5" width="150" height="100" fill="#1e293b"/>
      {/* 3D model wireframe */}
      <path d="M 40 30 L 70 30 L 80 50 L 40 50 Z" fill="none" stroke="#14b8a6" strokeWidth="2"/>
      <path d="M 70 30 L 80 20 L 80 40 L 80 50" fill="none" stroke="#14b8a6" strokeWidth="2"/>
      <path d="M 40 30 L 50 20 L 80 20" fill="none" stroke="#14b8a6" strokeWidth="2"/>
      <path d="M 50 20 L 50 40 L 40 50" fill="none" stroke="#14b8a6" strokeWidth="2"/>
      {/* Grid */}
      <line x1="20" y1="70" x2="140" y2="70" stroke="#475569" strokeWidth="1"/>
      <line x1="20" y1="80" x2="140" y2="80" stroke="#475569" strokeWidth="1"/>
      <line x1="40" y1="60" x2="40" y2="90" stroke="#475569" strokeWidth="1"/>
      <line x1="60" y1="60" x2="60" y2="90" stroke="#475569" strokeWidth="1"/>
      <line x1="80" y1="60" x2="80" y2="90" stroke="#475569" strokeWidth="1"/>
      <line x1="100" y1="60" x2="100" y2="90" stroke="#475569" strokeWidth="1"/>
      <line x1="120" y1="60" x2="120" y2="90" stroke="#475569" strokeWidth="1"/>
      {/* CONFIDENTIAL watermark */}
      <text x="80" y="95" fontFamily="sans-serif" fontSize="10" fontWeight="700" fill="#dc2626" textAnchor="middle" opacity="0.8">CONFIDENTIAL</text>
    </g>
    
    {/* Industrial metal surfaces with rivets */}
    <g opacity="0.4">
      <circle cx="1570" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1590" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1610" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1630" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1650" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1670" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1690" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1710" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1730" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1750" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1770" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1790" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1810" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1830" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1850" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1870" cy="340" r="3" fill="#94a3b8"/>
      <circle cx="1890" cy="340" r="3" fill="#94a3b8"/>
    </g>
    
    {/* Warning stripes on floor */}
    <g opacity="0.6">
      <rect x="1550" y="480" width="40" height="15" fill="#fbbf24" transform="skewX(-20)"/>
      <rect x="1600" y="480" width="40" height="15" fill="#1e293b" transform="skewX(-20)"/>
      <rect x="1650" y="480" width="40" height="15" fill="#fbbf24" transform="skewX(-20)"/>
      <rect x="1700" y="480" width="40" height="15" fill="#1e293b" transform="skewX(-20)"/>
      <rect x="1750" y="480" width="40" height="15" fill="#fbbf24" transform="skewX(-20)"/>
      <rect x="1800" y="480" width="40" height="15" fill="#1e293b" transform="skewX(-20)"/>
      <rect x="1850" y="480" width="40" height="15" fill="#fbbf24" transform="skewX(-20)"/>
    </g>
    
    {/* Orange safety lighting */}
    <ellipse cx="1600" cy="150" rx="100" ry="80" fill="#f59e0b" opacity="0.15"/>
    <ellipse cx="1820" cy="200" rx="80" ry="100" fill="#f59e0b" opacity="0.18"/>
    
    {/* Teal accent strips */}
    <rect x="1560" y="470" width="340" height="8" fill="#14b8a6" opacity="0.6" rx="2">
      <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.5s" repeatCount="indefinite"/>
    </rect>
    <rect x="1536" y="480" width="384" height="120" fill="url(#tealAccent)" opacity="0.3"/>
    
    {/* High-tech manufacturing aesthetic elements */}
    <g opacity="0.3">
      <line x1="1560" y1="200" x2="1900" y2="200" stroke="#14b8a6" strokeWidth="1" strokeDasharray="5,5">
        <animate attributeName="stroke-dashoffset" from="0" to="10" dur="1s" repeatCount="indefinite"/>
      </line>
    </g>
  </g>
  
  {/* Global depth of field simulation (foreground sharp, background soft) */}
  <rect width="1920" height="600" fill="url(#vignette)" opacity="0.2"/>
  
  {/* Unified teal brand cohesion overlay */}
  <g opacity="0.08">
    <circle cx="192" cy="300" r="60" fill="url(#tealGlow)"/>
    <circle cx="536" cy="300" r="60" fill="url(#tealGlow)"/>
    <circle cx="816" cy="300" r="60" fill="url(#tealGlow)"/>
    <circle cx="1104" cy="300" r="60" fill="url(#tealGlow)"/>
    <circle cx="1392" cy="300" r="60" fill="url(#tealGlow)"/>
    <circle cx="1728" cy="300" r="60" fill="url(#tealGlow)"/>
  </g>
  
  {/* Subtle rim lighting separating subjects from backgrounds */}
  <g opacity="0.15">
    <rect x="0" y="0" width="1920" height="3" fill="#14b8a6"/>
    <rect x="0" y="597" width="1920" height="3" fill="#14b8a6"/>
  </g>
  
  {/* Final color grading overlay (lifted blacks, teal/orange palette) */}
  <rect width="1920" height="600" fill="#0f172a" opacity="0.05"/>
    </svg>
  )
}
