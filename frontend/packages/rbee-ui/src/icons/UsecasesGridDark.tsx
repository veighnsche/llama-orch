import type { SVGProps } from 'react'

export interface UsecasesGridDarkProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function UsecasesGridDark({ size = 1920, className, ...props }: UsecasesGridDarkProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 1920 640"
      className={className}
      {...props}
    >
      <defs>
    {/* Background gradients */}
    <radialGradient id="bgRadial" cx="50%" cy="40%">
      <stop offset="0%" style="stop-color:#1e293b;stop-opacity:0.6" />
      <stop offset="100%" style="stop-color:#0f172a;stop-opacity:1" />
    </radialGradient>
    
    <radialGradient id="vignette" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#0f172a;stop-opacity:0" />
      <stop offset="100%" style="stop-color:#000000;stop-opacity:0.7" />
    </radialGradient>
    
    {/* Lighting effects */}
    <radialGradient id="amberGlow" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#f59e0b;stop-opacity:0.8" />
      <stop offset="50%" style="stop-color:#f59e0b;stop-opacity:0.3" />
      <stop offset="100%" style="stop-color:#f59e0b;stop-opacity:0" />
    </radialGradient>
    
    <radialGradient id="blueGlow" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#3b82f6;stop-opacity:0.6" />
      <stop offset="50%" style="stop-color:#3b82f6;stop-opacity:0.2" />
      <stop offset="100%" style="stop-color:#3b82f6;stop-opacity:0" />
    </radialGradient>
    
    <radialGradient id="tealGlow" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#14b8a6;stop-opacity:0.8" />
      <stop offset="50%" style="stop-color:#14b8a6;stop-opacity:0.3" />
      <stop offset="100%" style="stop-color:#14b8a6;stop-opacity:0" />
    </radialGradient>
    
    {/* Network line gradient */}
    <linearGradient id="networkLine" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#14b8a6;stop-opacity:0.2" />
      <stop offset="50%" style="stop-color:#14b8a6;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#14b8a6;stop-opacity:0.2" />
    </linearGradient>
    
    {/* Wood texture gradient */}
    <linearGradient id="woodGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#3e2723" />
      <stop offset="50%" style="stop-color:#4e342e" />
      <stop offset="100%" style="stop-color:#3e2723" />
    </linearGradient>
    
    {/* Monitor glow */}
    <radialGradient id="monitorGlow" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#64748b;stop-opacity:0.4" />
      <stop offset="100%" style="stop-color:#64748b;stop-opacity:0" />
    </radialGradient>
    
    {/* Blur filters */}
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    
    <filter id="softGlow">
      <feGaussianBlur stdDeviation="5" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  
  {/* Base background - deep navy */}
  <rect width="1920" height="640" fill="#0f172a"/>
  <rect width="1920" height="640" fill="url(#bgRadial)"/>
  
  {/* Dark walnut desk surface */}
  <ellipse cx="960" cy="580" rx="900" ry="60" fill="url(#woodGradient)" opacity="0.8"/>
  <ellipse cx="960" cy="580" rx="850" ry="50" fill="#2d1b13" opacity="0.6"/>
  
  {/* Teal accent strip lighting on walls */}
  <rect x="0" y="40" width="1920" height="8" fill="#14b8a6" opacity="0.3">
    <animate attributeName="opacity" values="0.2;0.4;0.2" dur="4s" repeatCount="indefinite"/>
  </rect>
  <rect x="0" y="600" width="1920" height="8" fill="#14b8a6" opacity="0.3">
    <animate attributeName="opacity" values="0.3;0.5;0.3" dur="4s" repeatCount="indefinite"/>
  </rect>
  
  {/* LEFT THIRD: Gaming PC Tower */}
  <g id="gamingPC">
    {/* Desk placement */}
    <rect x="220" y="520" width="280" height="60" fill="url(#woodGradient)" opacity="0.9"/>
    
    {/* Tower case (20 inches tall, black) */}
    <rect x="260" y="260" width="200" height="260" rx="8" fill="#0a0f1a" stroke="#1e293b" stroke-width="3"/>
    
    {/* Tempered glass side panel */}
    <rect x="270" y="270" width="90" height="240" rx="6" fill="#0f172a" opacity="0.4" stroke="#334155" stroke-width="1.5"/>
    <rect x="275" y="275" width="80" height="230" rx="4" fill="#0a0f1a" opacity="0.2"/>
    
    {/* GPU 1: NVIDIA RTX 4090 (top) */}
    <g transform="translate(280, 310)">
      {/* PCB green circuit board */}
      <rect x="0" y="0" width="70" height="50" rx="2" fill="#2d5016" stroke="#3d6b1f" stroke-width="1.5"/>
      {/* Circuit traces */}
      <line x1="10" y1="10" x2="60" y2="10" stroke="#4a7c1f" stroke-width="0.5"/>
      <line x1="10" y1="20" x2="60" y2="20" stroke="#4a7c1f" stroke-width="0.5"/>
      <line x1="10" y1="30" x2="60" y2="30" stroke="#4a7c1f" stroke-width="0.5"/>
      <line x1="10" y1="40" x2="60" y2="40" stroke="#4a7c1f" stroke-width="0.5"/>
      {/* Silver heatsink fins */}
      <rect x="5" y="5" width="60" height="40" fill="#94a3b8" opacity="0.7"/>
      <line x1="10" y1="5" x2="10" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="15" y1="5" x2="15" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="20" y1="5" x2="20" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="25" y1="5" x2="25" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="30" y1="5" x2="30" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="35" y1="5" x2="35" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="40" y1="5" x2="40" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="45" y1="5" x2="45" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="50" y1="5" x2="50" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="55" y1="5" x2="55" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="60" y1="5" x2="60" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      {/* Amber LED strip */}
      <rect x="0" y="50" width="70" height="4" fill="#f59e0b" opacity="0.9" rx="1">
        <animate attributeName="opacity" values="0.7;1;0.7" dur="2s" repeatCount="indefinite"/>
      </rect>
      {/* NVIDIA logo area */}
      <text x="35" y="28" font-family="sans-serif" font-size="8" fill="#10b981" text-anchor="middle" font-weight="700" opacity="0.8">NVIDIA</text>
    </g>
    
    {/* GPU 2: NVIDIA RTX 4090 (bottom) */}
    <g transform="translate(280, 400)">
      <rect x="0" y="0" width="70" height="50" rx="2" fill="#2d5016" stroke="#3d6b1f" stroke-width="1.5"/>
      <line x1="10" y1="10" x2="60" y2="10" stroke="#4a7c1f" stroke-width="0.5"/>
      <line x1="10" y1="20" x2="60" y2="20" stroke="#4a7c1f" stroke-width="0.5"/>
      <line x1="10" y1="30" x2="60" y2="30" stroke="#4a7c1f" stroke-width="0.5"/>
      <line x1="10" y1="40" x2="60" y2="40" stroke="#4a7c1f" stroke-width="0.5"/>
      <rect x="5" y="5" width="60" height="40" fill="#94a3b8" opacity="0.7"/>
      <line x1="10" y1="5" x2="10" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="15" y1="5" x2="15" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="20" y1="5" x2="20" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="25" y1="5" x2="25" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="30" y1="5" x2="30" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="35" y1="5" x2="35" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="40" y1="5" x2="40" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="45" y1="5" x2="45" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="50" y1="5" x2="50" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="55" y1="5" x2="55" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="60" y1="5" x2="60" y2="45" stroke="#cbd5e1" stroke-width="1"/>
      <rect x="0" y="50" width="70" height="4" fill="#f59e0b" opacity="0.9" rx="1">
        <animate attributeName="opacity" values="0.9;1;0.9" dur="1.8s" repeatCount="indefinite"/>
      </rect>
      <text x="35" y="28" font-family="sans-serif" font-size="8" fill="#10b981" text-anchor="middle" font-weight="700" opacity="0.8">NVIDIA</text>
    </g>
    
    {/* Teal RGB accent lighting inside case */}
    <rect x="370" y="270" width="80" height="6" fill="#14b8a6" opacity="0.7" rx="2">
      <animate attributeName="opacity" values="0.5;0.9;0.5" dur="3s" repeatCount="indefinite"/>
    </rect>
    <rect x="370" y="500" width="80" height="6" fill="#14b8a6" opacity="0.7" rx="2">
      <animate attributeName="opacity" values="0.7;0.9;0.7" dur="3s" repeatCount="indefinite"/>
    </rect>
    
    {/* Case details */}
    <rect x="370" y="280" width="80" height="220" fill="#0a0f1a"/>
    <circle cx="450" cy="290" r="3" fill="#10b981" opacity="0.8">
      <animate attributeName="opacity" values="0.6;1;0.6" dur="2.5s" repeatCount="indefinite"/>
    </circle>
    
    {/* Amber glow from GPUs */}
    <ellipse cx="315" cy="360" rx="80" ry="100" fill="url(#amberGlow)" opacity="0.6" filter="url(#softGlow)"/>
    <ellipse cx="315" cy="430" rx="80" ry="100" fill="url(#amberGlow)" opacity="0.5" filter="url(#softGlow)"/>
    
    {/* Desk reflection of GPU glow */}
    <ellipse cx="360" cy="540" rx="100" ry="20" fill="#f59e0b" opacity="0.15"/>
  </g>
  
  {/* CENTER THIRD: Professional Workstation */}
  <g id="workstation">
    {/* Desk placement */}
    <rect x="740" y="520" width="440" height="60" fill="url(#woodGradient)" opacity="0.9"/>
    
    {/* Left Monitor (27-inch, landscape, code editor) */}
    <g transform="translate(760, 200)">
      {/* Monitor frame */}
      <rect x="0" y="0" width="180" height="140" rx="6" fill="#1e293b" stroke="#334155" stroke-width="3"/>
      {/* Screen */}
      <rect x="6" y="6" width="168" height="115" rx="3" fill="#0a0f1a"/>
      {/* Code editor content */}
      <rect x="10" y="10" width="160" height="105" fill="#0f172a"/>
      {/* Syntax highlighting simulation */}
      <line x1="15" y1="20" x2="60" y2="20" stroke="#3b82f6" stroke-width="2"/>
      <line x1="65" y1="20" x2="90" y2="20" stroke="#10b981" stroke-width="2"/>
      <line x1="15" y1="30" x2="45" y2="30" stroke="#f59e0b" stroke-width="2"/>
      <line x1="50" y1="30" x2="110" y2="30" stroke="#14b8a6" stroke-width="2"/>
      <line x1="25" y1="40" x2="75" y2="40" stroke="#3b82f6" stroke-width="2"/>
      <line x1="80" y1="40" x2="120" y2="40" stroke="#a855f7" stroke-width="2"/>
      <line x1="25" y1="50" x2="95" y2="50" stroke="#10b981" stroke-width="2"/>
      <line x1="15" y1="60" x2="55" y2="60" stroke="#f59e0b" stroke-width="2"/>
      <line x1="60" y1="60" x2="100" y2="60" stroke="#14b8a6" stroke-width="2"/>
      <line x1="25" y1="70" x2="85" y2="70" stroke="#3b82f6" stroke-width="2"/>
      <line x1="15" y1="80" x2="70" y2="80" stroke="#10b981" stroke-width="2"/>
      <line x1="25" y1="90" x2="105" y2="90" stroke="#a855f7" stroke-width="2"/>
      <line x1="15" y1="100" x2="50" y2="100" stroke="#f59e0b" stroke-width="2"/>
      {/* Cursor blink */}
      <rect x="120" y="94" width="2" height="8" fill="#3b82f6">
        <animate attributeName="opacity" values="0;1;0" dur="1s" repeatCount="indefinite"/>
      </rect>
      {/* Stand */}
      <rect x="70" y="121" width="40" height="15" fill="#334155" rx="2"/>
      <rect x="75" y="136" width="30" height="30" fill="#1e293b" rx="3"/>
    </g>
    
    {/* Right Monitor (27-inch, landscape, terminal) */}
    <g transform="translate(960, 200)">
      {/* Monitor frame */}
      <rect x="0" y="0" width="180" height="140" rx="6" fill="#1e293b" stroke="#334155" stroke-width="3"/>
      {/* Screen */}
      <rect x="6" y="6" width="168" height="115" rx="3" fill="#0a0f1a"/>
      {/* Terminal content */}
      <rect x="10" y="10" width="160" height="105" fill="#000000"/>
      {/* Streaming green text output */}
      <text x="15" y="25" font-family="monospace" font-size="7" fill="#10b981">$ rbee inference start</text>
      <text x="15" y="35" font-family="monospace" font-size="7" fill="#10b981">[OK] Worker ready</text>
      <text x="15" y="45" font-family="monospace" font-size="7" fill="#10b981">[OK] Model loaded</text>
      <text x="15" y="55" font-family="monospace" font-size="7" fill="#10b981">[OK] GPU detected</text>
      <text x="15" y="65" font-family="monospace" font-size="7" fill="#14b8a6">Generating tokens...</text>
      <rect x="15" y="70" width="80" height="2" fill="#10b981" opacity="0.8"/>
      <text x="15" y="80" font-family="monospace" font-size="7" fill="#64748b">████████████░░░░ 75%</text>
      <text x="15" y="90" font-family="monospace" font-size="7" fill="#10b981">45.3 tok/s</text>
      {/* Cursor */}
      <rect x="15" y="95" width="6" height="8" fill="#10b981">
        <animate attributeName="opacity" values="0;1;0" dur="0.8s" repeatCount="indefinite"/>
      </rect>
      {/* Stand */}
      <rect x="70" y="121" width="40" height="15" fill="#334155" rx="2"/>
      <rect x="75" y="136" width="30" height="30" fill="#1e293b" rx="3"/>
    </g>
    
    {/* Monitor arms (sleek aluminum) */}
    <rect x="830" y="360" width="8" height="60" fill="#94a3b8" rx="2"/>
    <rect x="1030" y="360" width="8" height="60" fill="#94a3b8" rx="2"/>
    <rect x="820" y="418" width="230" height="6" fill="#94a3b8" rx="2"/>
    
    {/* Black mechanical keyboard */}
    <rect x="800" y="450" width="180" height="50" rx="4" fill="#0a0f1a" stroke="#1e293b" stroke-width="2"/>
    {/* Key caps */}
    <g opacity="0.6">
      <rect x="810" y="460" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="822" y="460" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="834" y="460" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="846" y="460" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="858" y="460" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="870" y="460" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="810" y="472" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="822" y="472" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="834" y="472" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="846" y="472" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="858" y="472" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="870" y="472" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="810" y="484" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="822" y="484" width="8" height="8" rx="1" fill="#334155"/>
      <rect x="834" y="484" width="32" height="8" rx="1" fill="#334155"/>
      <rect x="870" y="484" width="8" height="8" rx="1" fill="#334155"/>
    </g>
    
    {/* Mouse */}
    <ellipse cx="1000" cy="475" rx="18" ry="28" fill="#0a0f1a" stroke="#1e293b" stroke-width="2"/>
    <line x1="1000" y1="455" x2="1000" y2="485" stroke="#1e293b" stroke-width="1"/>
    
    {/* Cable management (clean) */}
    <line x1="838" y1="420" x2="838" y2="520" stroke="#334155" stroke-width="2" opacity="0.3"/>
    <line x1="1038" y1="420" x2="1038" y2="520" stroke="#334155" stroke-width="2" opacity="0.3"/>
    
    {/* Monitor glow (cool blue-white) */}
    <ellipse cx="850" cy="270" rx="120" ry="90" fill="url(#blueGlow)" opacity="0.5" filter="url(#softGlow)"/>
    <ellipse cx="1050" cy="270" rx="120" ry="90" fill="url(#blueGlow)" opacity="0.4" filter="url(#softGlow)"/>
    
    {/* Desk reflection */}
    <ellipse cx="950" cy="540" rx="180" ry="25" fill="#3b82f6" opacity="0.1"/>
  </g>
  
  {/* RIGHT THIRD: Mac Studio Setup */}
  <g id="macStudio">
    {/* Desk placement */}
    <rect x="1280" y="520" width="360" height="60" fill="url(#woodGradient)" opacity="0.9"/>
    
    {/* Mac Studio (compact, silver aluminum, 4 inches tall) */}
    <g transform="translate(1380, 420)">
      {/* Main body */}
      <rect x="0" y="0" width="120" height="48" rx="8" fill="#cbd5e1" stroke="#94a3b8" stroke-width="2"/>
      {/* Top surface (slight gradient) */}
      <rect x="2" y="2" width="116" height="10" rx="6" fill="#e2e8f0" opacity="0.8"/>
      {/* Apple logo area */}
      <circle cx="60" cy="24" r="8" fill="#94a3b8" opacity="0.4"/>
      {/* Status LED */}
      <circle cx="110" cy="24" r="2" fill="#10b981">
        <animate attributeName="opacity" values="0.7;1;0.7" dur="3s" repeatCount="indefinite"/>
      </circle>
      {/* Port indicators */}
      <rect x="5" y="38" width="4" height="6" rx="1" fill="#64748b"/>
      <rect x="12" y="38" width="4" height="6" rx="1" fill="#64748b"/>
      <rect x="19" y="38" width="4" height="6" rx="1" fill="#64748b"/>
      {/* Aluminum texture lines */}
      <line x1="10" y1="15" x2="110" y2="15" stroke="#94a3b8" stroke-width="0.3" opacity="0.5"/>
      <line x1="10" y1="20" x2="110" y2="20" stroke="#94a3b8" stroke-width="0.3" opacity="0.5"/>
      <line x1="10" y1="30" x2="110" y2="30" stroke="#94a3b8" stroke-width="0.3" opacity="0.5"/>
    </g>
    
    {/* 24-inch Display */}
    <g transform="translate(1320, 180)">
      {/* Monitor frame (thin bezels) */}
      <rect x="0" y="0" width="240" height="170" rx="8" fill="#e2e8f0" stroke="#cbd5e1" stroke-width="3"/>
      {/* Screen */}
      <rect x="8" y="8" width="224" height="140" rx="4" fill="#0a0f1a"/>
      {/* Design software interface */}
      <rect x="12" y="12" width="216" height="132" fill="#1e293b"/>
      
      {/* Design software UI */}
      {/* Top toolbar */}
      <rect x="15" y="15" width="210" height="15" fill="#0f172a"/>
      <circle cx="20" cy="22" r="2" fill="#ef4444"/>
      <circle cx="28" cy="22" r="2" fill="#f59e0b"/>
      <circle cx="36" cy="22" r="2" fill="#10b981"/>
      <rect x="50" y="18" width="30" height="8" rx="2" fill="#334155"/>
      <rect x="85" y="18" width="30" height="8" rx="2" fill="#334155"/>
      
      {/* Canvas area with design elements */}
      <rect x="18" y="35" width="198" height="100" fill="#0f172a"/>
      {/* Design shapes */}
      <rect x="50" y="50" width="60" height="60" rx="4" fill="#14b8a6" opacity="0.6"/>
      <circle cx="150" cy="70" r="25" fill="#3b82f6" opacity="0.6"/>
      <path d="M 120 100 L 140 60 L 160 100 Z" fill="#a855f7" opacity="0.6"/>
      
      {/* Layers panel */}
      <rect x="170" y="35" width="44" height="100" fill="#0a0f1a"/>
      <text x="175" y="45" font-family="sans-serif" font-size="6" fill="#94a3b8">Layers</text>
      <rect x="173" y="50" width="38" height="12" fill="#334155" rx="1"/>
      <rect x="173" y="65" width="38" height="12" fill="#334155" rx="1"/>
      <rect x="173" y="80" width="38" height="12" fill="#1e293b" rx="1"/>
      
      {/* Chin/base */}
      <rect x="90" y="148" width="60" height="18" fill="#e2e8f0" rx="2"/>
      <ellipse cx="120" cy="170" rx="35" ry="8" fill="#cbd5e1"/>
    </g>
    
    {/* Wireless keyboard (aluminum, compact) */}
    <rect x="1340" y="450" width="200" height="45" rx="4" fill="#e2e8f0" stroke="#cbd5e1" stroke-width="2"/>
    {/* Keys */}
    <g opacity="0.7">
      <rect x="1350" y="460" width="8" height="8" rx="1" fill="#f8fafc"/>
      <rect x="1362" y="460" width="8" height="8" rx="1" fill="#f8fafc"/>
      <rect x="1374" y="460" width="8" height="8" rx="1" fill="#f8fafc"/>
      <rect x="1386" y="460" width="8" height="8" rx="1" fill="#f8fafc"/>
      <rect x="1398" y="460" width="8" height="8" rx="1" fill="#f8fafc"/>
      <rect x="1410" y="460" width="8" height="8" rx="1" fill="#f8fafc"/>
      <rect x="1350" y="472" width="8" height="8" rx="1" fill="#f8fafc"/>
      <rect x="1362" y="472" width="8" height="8" rx="1" fill="#f8fafc"/>
      <rect x="1374" y="472" width="8" height="8" rx="1" fill="#f8fafc"/>
      <rect x="1386" y="472" width="8" height="8" rx="1" fill="#f8fafc"/>
      <rect x="1398" y="472" width="8" height="8" rx="1" fill="#f8fafc"/>
      <rect x="1410" y="472" width="8" height="8" rx="1" fill="#f8fafc"/>
      <rect x="1350" y="484" width="32" height="8" rx="1" fill="#f8fafc"/>
    </g>
    
    {/* Wireless trackpad */}
    <rect x="1550" y="455" width="70" height="35" rx="4" fill="#e2e8f0" stroke="#cbd5e1" stroke-width="2"/>
    <rect x="1555" y="460" width="60" height="25" rx="2" fill="#f8fafc" opacity="0.5"/>
    
    {/* Desk plant (small, white ceramic pot) */}
    <g transform="translate(1300, 435)">
      {/* Pot */}
      <ellipse cx="20" cy="35" rx="15" ry="8" fill="#f1f5f9"/>
      <rect x="5" y="25" width="30" height="15" fill="#f8fafc" rx="2"/>
      <ellipse cx="20" cy="25" rx="15" ry="6" fill="#ffffff"/>
      {/* Plant leaves */}
      <ellipse cx="15" cy="15" rx="8" ry="12" fill="#10b981" opacity="0.8"/>
      <ellipse cx="25" cy="12" rx="7" ry="10" fill="#10b981" opacity="0.7"/>
      <ellipse cx="20" cy="8" rx="6" ry="11" fill="#14b8a6" opacity="0.8"/>
    </g>
    
    {/* Neutral white monitor glow */}
    <ellipse cx="1440" cy="265" rx="130" ry="100" fill="#f8fafc" opacity="0.15" filter="url(#softGlow)"/>
    
    {/* Desk reflection */}
    <ellipse cx="1440" cy="540" rx="150" ry="20" fill="#cbd5e1" opacity="0.08"/>
  </g>
  
  {/* NETWORK VISUALIZATION: Glowing teal mesh topology */}
  <g id="network" filter="url(#glow)">
    {/* Network lines float 6 inches above desk (y=250-350) */}
    
    {/* Gaming PC to Workstation (main connection) */}
    <line x1="460" y1="300" x2="760" y2="280" stroke="#14b8a6" stroke-width="3" stroke-linecap="round" opacity="0.9">
      <animate attributeName="opacity" values="0.7;1;0.7" dur="2s" repeatCount="indefinite"/>
    </line>
    
    {/* Workstation to Mac Studio (main connection) */}
    <line x1="1140" y1="280" x2="1380" y2="300" stroke="#14b8a6" stroke-width="3" stroke-linecap="round" opacity="0.9">
      <animate attributeName="opacity" values="0.8;1;0.8" dur="2.2s" repeatCount="indefinite"/>
    </line>
    
    {/* Gaming PC to Mac Studio (mesh topology) */}
    <path d="M 460 320 Q 920 380 1380 320" stroke="#14b8a6" stroke-width="2.5" fill="none" stroke-linecap="round" opacity="0.6">
      <animate attributeName="opacity" values="0.4;0.7;0.4" dur="3s" repeatCount="indefinite"/>
    </path>
    
    {/* Node indicators at each machine */}
    {/* Gaming PC node */}
    <circle cx="460" cy="310" r="8" fill="#14b8a6" opacity="0.8" stroke="#14b8a6" stroke-width="2">
      <animate attributeName="r" values="8;10;8" dur="2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="460" cy="310" r="15" fill="none" stroke="#14b8a6" opacity="0.3" stroke-width="1">
      <animate attributeName="r" values="15;20;15" dur="2s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.3;0;0.3" dur="2s" repeatCount="indefinite"/>
    </circle>
    
    {/* Workstation node */}
    <circle cx="950" cy="280" r="8" fill="#14b8a6" opacity="0.8" stroke="#14b8a6" stroke-width="2">
      <animate attributeName="r" values="8;10;8" dur="2.2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="950" cy="280" r="15" fill="none" stroke="#14b8a6" opacity="0.3" stroke-width="1">
      <animate attributeName="r" values="15;20;15" dur="2.2s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.3;0;0.3" dur="2.2s" repeatCount="indefinite"/>
    </circle>
    
    {/* Mac Studio node */}
    <circle cx="1380" cy="310" r="8" fill="#14b8a6" opacity="0.8" stroke="#14b8a6" stroke-width="2">
      <animate attributeName="r" values="8;10;8" dur="2.4s" repeatCount="indefinite"/>
    </circle>
    <circle cx="1380" cy="310" r="15" fill="none" stroke="#14b8a6" opacity="0.3" stroke-width="1">
      <animate attributeName="r" values="15;20;15" dur="2.4s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.3;0;0.3" dur="2.4s" repeatCount="indefinite"/>
    </circle>
    
    {/* Data packets (glowing dots traveling along lines) */}
    {/* Packet 1: Gaming PC to Workstation */}
    <circle cx="460" cy="300" r="4" fill="#14b8a6" filter="url(#glow)">
      <animate attributeName="cx" values="460;760" dur="2s" repeatCount="indefinite"/>
      <animate attributeName="cy" values="300;280" dur="2s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;1;0" dur="2s" repeatCount="indefinite"/>
    </circle>
    
    {/* Packet 2: Workstation to Mac Studio */}
    <circle cx="1140" cy="280" r="4" fill="#14b8a6" filter="url(#glow)">
      <animate attributeName="cx" values="1140;1380" dur="2s" begin="0.5s" repeatCount="indefinite"/>
      <animate attributeName="cy" values="280;300" dur="2s" begin="0.5s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;1;0" dur="2s" begin="0.5s" repeatCount="indefinite"/>
    </circle>
    
    {/* Packet 3: Gaming PC to Mac Studio (mesh) */}
    <circle cx="460" cy="320" r="4" fill="#14b8a6" filter="url(#glow)">
      <animateMotion dur="3s" begin="1s" repeatCount="indefinite" path="M 460 320 Q 920 380 1380 320"/>
      <animate attributeName="opacity" values="0;1;1;0" dur="3s" begin="1s" repeatCount="indefinite"/>
    </circle>
    
    {/* Packet 4: Mac Studio to Gaming PC (reverse) */}
    <circle cx="1380" cy="300" r="4" fill="#14b8a6" filter="url(#glow)">
      <animate attributeName="cx" values="1380;760" dur="2.5s" begin="1.5s" repeatCount="indefinite"/>
      <animate attributeName="cy" values="300;280" dur="2.5s" begin="1.5s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;1;0" dur="2.5s" begin="1.5s" repeatCount="indefinite"/>
    </circle>
    
    {/* Outer glow effect on network lines */}
    <line x1="460" y1="300" x2="760" y2="280" stroke="#14b8a6" stroke-width="6" stroke-linecap="round" opacity="0.2" filter="url(#softGlow)"/>
    <line x1="1140" y1="280" x2="1380" y2="300" stroke="#14b8a6" stroke-width="6" stroke-linecap="round" opacity="0.2" filter="url(#softGlow)"/>
    <path d="M 460 320 Q 920 380 1380 320" stroke="#14b8a6" stroke-width="5" fill="none" stroke-linecap="round" opacity="0.15" filter="url(#softGlow)"/>
  </g>
  
  {/* FLOATING UI OVERLAYS: Semi-transparent info panels */}
  <g id="uiOverlays">
    {/* Main cluster info panel */}
    <g transform="translate(710, 60)">
      {/* Panel background with drop shadow effect */}
      <rect x="2" y="2" width="500" height="85" rx="10" fill="#000000" opacity="0.3"/>
      <rect x="0" y="0" width="500" height="85" rx="10" fill="#0f172a" opacity="0.85" stroke="#14b8a6" stroke-width="2"/>
      
      {/* Title */}
      <text x="250" y="30" font-family="sans-serif" font-size="22" fill="#f1f5f9" text-anchor="middle" font-weight="700">
        Private AI Cluster
      </text>
      
      {/* Stats row */}
      <g transform="translate(0, 45)">
        {/* GPU count with icon */}
        <g transform="translate(80, 0)">
          <rect x="0" y="0" width="80" height="28" rx="6" fill="#14b8a6" opacity="0.2"/>
          <rect x="5" y="5" width="10" height="8" rx="1" fill="#14b8a6" opacity="0.8"/>
          <rect x="5" y="15" width="10" height="8" rx="1" fill="#14b8a6" opacity="0.8"/>
          <text x="25" y="18" font-family="sans-serif" font-size="14" fill="#14b8a6" font-weight="600">8 GPUs</text>
        </g>
        
        {/* Node count with icon */}
        <g transform="translate(210, 0)">
          <rect x="0" y="0" width="80" height="28" rx="6" fill="#14b8a6" opacity="0.2"/>
          <circle cx="10" cy="10" r="3" fill="#14b8a6" opacity="0.8"/>
          <circle cx="10" cy="18" r="3" fill="#14b8a6" opacity="0.8"/>
          <circle cx="18" cy="14" r="3" fill="#14b8a6" opacity="0.8"/>
          <line x1="10" y1="10" x2="18" y2="14" stroke="#14b8a6" stroke-width="1"/>
          <line x1="10" y1="18" x2="18" y2="14" stroke="#14b8a6" stroke-width="1"/>
          <text x="25" y="18" font-family="sans-serif" font-size="14" fill="#14b8a6" font-weight="600">3 nodes</text>
        </g>
        
        {/* Cost savings */}
        <g transform="translate(340, 0)">
          <rect x="0" y="0" width="120" height="28" rx="6" fill="#10b981" opacity="0.2"/>
          <text x="10" y="10" font-family="sans-serif" font-size="20" fill="#10b981" font-weight="700">$</text>
          <line x1="15" y1="5" x2="25" y2="15" stroke="#10b981" stroke-width="2"/>
          <text x="35" y="18" font-family="sans-serif" font-size="14" fill="#10b981" font-weight="700">0/mo API</text>
        </g>
      </g>
    </g>
    
    {/* Individual workstation labels (subtle) */}
    {/* Gaming PC label */}
    <g transform="translate(300, 240)">
      <rect x="0" y="0" width="100" height="24" rx="4" fill="#0f172a" opacity="0.7" stroke="#f59e0b" stroke-width="1"/>
      <text x="50" y="16" font-family="sans-serif" font-size="10" fill="#f59e0b" text-anchor="middle" font-weight="600">Gaming PC</text>
    </g>
    
    {/* Workstation label */}
    <g transform="translate(850, 160)">
      <rect x="0" y="0" width="100" height="24" rx="4" fill="#0f172a" opacity="0.7" stroke="#3b82f6" stroke-width="1"/>
      <text x="50" y="16" font-family="sans-serif" font-size="10" fill="#3b82f6" text-anchor="middle" font-weight="600">Workstation</text>
    </g>
    
    {/* Mac Studio label */}
    <g transform="translate(1380, 360)">
      <rect x="0" y="0" width="100" height="24" rx="4" fill="#0f172a" opacity="0.7" stroke="#cbd5e1" stroke-width="1"/>
      <text x="50" y="16" font-family="sans-serif" font-size="10" fill="#f8fafc" text-anchor="middle" font-weight="600">Mac Studio</text>
    </g>
  </g>
  
  {/* LIGHTING: Professional studio lighting with key light from upper left */}
  <g id="lighting" opacity="0.4">
    {/* Key light from upper left */}
    <defs>
      <radialGradient id="keyLight" cx="30%" cy="30%">
        <stop offset="0%" style="stop-color:#f8fafc;stop-opacity:0.3" />
        <stop offset="100%" style="stop-color:#f8fafc;stop-opacity:0" />
      </radialGradient>
    </defs>
    <ellipse cx="400" cy="150" rx="600" ry="400" fill="url(#keyLight)"/>
    
    {/* Soft shadows (subtle) */}
    <ellipse cx="360" cy="540" rx="90" ry="15" fill="#000000" opacity="0.2"/>
    <ellipse cx="950" cy="540" rx="170" ry="20" fill="#000000" opacity="0.2"/>
    <ellipse cx="1440" cy="540" rx="140" ry="18" fill="#000000" opacity="0.15"/>
  </g>
  
  {/* CINEMATIC COLOR GRADING: Lifted blacks, teal/amber palette */}
  <rect width="1920" height="640" fill="#1a1f2e" opacity="0.15"/>
  
  {/* VIGNETTE: Darkening edges */}
  <rect width="1920" height="640" fill="url(#vignette)" opacity="0.4"/>
  
  {/* DEPTH OF FIELD: Background blur simulation */}
  <rect x="0" y="0" width="1920" height="100" fill="url(#bgRadial)" opacity="0.3"/>
  <rect x="0" y="540" width="1920" height="100" fill="url(#bgRadial)" opacity="0.3"/>
    </svg>
  )
}
