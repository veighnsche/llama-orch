import type { SVGProps } from 'react'

export interface PricingOrchestratorProps extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function PricingOrchestrator({ size = 600, className, ...props }: PricingOrchestratorProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 600 600"
      className={className}
      {...props}
    >
      <defs>
    {/* Background gradient */}
    <radialGradient id="bgGradient" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#1e293b;stop-opacity:0.8" />
      <stop offset="100%" style="stop-color:#0f172a;stop-opacity:1" />
    </radialGradient>
    
    {/* Amber glow for connections */}
    <radialGradient id="amberGlow" cx="50%" cy="50%">
      <stop offset="0%" style="stop-color:#f59e0b;stop-opacity:0.6" />
      <stop offset="100%" style="stop-color:#f59e0b;stop-opacity:0" />
    </radialGradient>
    
    {/* Node gradient (navy with rim lighting) */}
    <linearGradient id="nodeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#334155;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#1e293b;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0f172a;stop-opacity:1" />
    </linearGradient>
    
    {/* Queen node gradient (larger, brighter) */}
    <radialGradient id="queenGradient" cx="30%" cy="30%">
      <stop offset="0%" style="stop-color:#475569;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#334155;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#1e293b;stop-opacity:1" />
    </radialGradient>
    
    {/* Amber rim light */}
    <linearGradient id="rimLight" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#f59e0b;stop-opacity:0" />
      <stop offset="50%" style="stop-color:#f59e0b;stop-opacity:0.4" />
      <stop offset="100%" style="stop-color:#f59e0b;stop-opacity:0" />
    </linearGradient>
    
    {/* Glow filter */}
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  
  {/* Background */}
  <rect width="600" height="600" fill="url(#bgGradient)"/>
  
  {/* Ambient glow spots */}
  <circle cx="300" cy="300" r="200" fill="url(#amberGlow)" opacity="0.15"/>
  <circle cx="150" cy="150" r="80" fill="url(#amberGlow)" opacity="0.1"/>
  <circle cx="450" cy="450" r="80" fill="url(#amberGlow)" opacity="0.1"/>
  
  {/* Connection lines (data streams) - hexagonal pattern */}
  {/* Ring 1 to center */}
  <g opacity="0.6" filter="url(#glow)">
    <line x1="300" y1="180" x2="300" y2="280" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4 4">
      <animate attributeName="stroke-dashoffset" from="0" to="8" dur="1s" repeatCount="indefinite"/>
    </line>
    <line x1="404" y1="240" x2="330" y2="285" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4 4">
      <animate attributeName="stroke-dashoffset" from="0" to="8" dur="1.2s" repeatCount="indefinite"/>
    </line>
    <line x1="404" y1="360" x2="330" y2="315" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4 4">
      <animate attributeName="stroke-dashoffset" from="0" to="8" dur="0.9s" repeatCount="indefinite"/>
    </line>
    <line x1="300" y1="420" x2="300" y2="320" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4 4">
      <animate attributeName="stroke-dashoffset" from="0" to="8" dur="1.1s" repeatCount="indefinite"/>
    </line>
    <line x1="196" y1="360" x2="270" y2="315" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4 4">
      <animate attributeName="stroke-dashoffset" from="0" to="8" dur="1.3s" repeatCount="indefinite"/>
    </line>
    <line x1="196" y1="240" x2="270" y2="285" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4 4">
      <animate attributeName="stroke-dashoffset" from="0" to="8" dur="0.95s" repeatCount="indefinite"/>
    </line>
  </g>
  
  {/* Ring 2 to Ring 1 connections */}
  <g opacity="0.4" filter="url(#glow)">
    <line x1="300" y1="100" x2="300" y2="180" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3 3">
      <animate attributeName="stroke-dashoffset" from="0" to="6" dur="1.5s" repeatCount="indefinite"/>
    </line>
    <line x1="450" y1="150" x2="404" y2="240" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3 3">
      <animate attributeName="stroke-dashoffset" from="0" to="6" dur="1.4s" repeatCount="indefinite"/>
    </line>
    <line x1="508" y1="300" x2="404" y2="300" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3 3">
      <animate attributeName="stroke-dashoffset" from="0" to="6" dur="1.6s" repeatCount="indefinite"/>
    </line>
    <line x1="450" y1="450" x2="404" y2="360" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3 3">
      <animate attributeName="stroke-dashoffset" from="0" to="6" dur="1.3s" repeatCount="indefinite"/>
    </line>
    <line x1="300" y1="500" x2="300" y2="420" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3 3">
      <animate attributeName="stroke-dashoffset" from="0" to="6" dur="1.7s" repeatCount="indefinite"/>
    </line>
    <line x1="150" y1="450" x2="196" y2="360" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3 3">
      <animate attributeName="stroke-dashoffset" from="0" to="6" dur="1.2s" repeatCount="indefinite"/>
    </line>
    <line x1="92" y1="300" x2="196" y2="300" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3 3">
      <animate attributeName="stroke-dashoffset" from="0" to="6" dur="1.8s" repeatCount="indefinite"/>
    </line>
    <line x1="150" y1="150" x2="196" y2="240" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3 3">
      <animate attributeName="stroke-dashoffset" from="0" to="6" dur="1.35s" repeatCount="indefinite"/>
    </line>
  </g>
  
  {/* Worker nodes - Ring 2 (outer) */}
  <g className="worker-nodes-outer">
    {/* Top */}
    <g transform="translate(300, 100)">
      <circle r="16" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1" opacity="0.9"/>
      <circle r="16" fill="none" stroke="url(#rimLight)" stroke-width="2" opacity="0.6"/>
      <circle r="8" fill="#1e293b" opacity="0.8"/>
    </g>
    
    {/* Top-right */}
    <g transform="translate(450, 150)">
      <circle r="16" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1" opacity="0.9"/>
      <circle r="16" fill="none" stroke="url(#rimLight)" stroke-width="2" opacity="0.6"/>
      <circle r="8" fill="#1e293b" opacity="0.8"/>
    </g>
    
    {/* Right */}
    <g transform="translate(508, 300)">
      <circle r="16" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1" opacity="0.9"/>
      <circle r="16" fill="none" stroke="url(#rimLight)" stroke-width="2" opacity="0.6"/>
      <circle r="8" fill="#1e293b" opacity="0.8"/>
    </g>
    
    {/* Bottom-right */}
    <g transform="translate(450, 450)">
      <circle r="16" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1" opacity="0.9"/>
      <circle r="16" fill="none" stroke="url(#rimLight)" stroke-width="2" opacity="0.6"/>
      <circle r="8" fill="#1e293b" opacity="0.8"/>
    </g>
    
    {/* Bottom */}
    <g transform="translate(300, 500)">
      <circle r="16" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1" opacity="0.9"/>
      <circle r="16" fill="none" stroke="url(#rimLight)" stroke-width="2" opacity="0.6"/>
      <circle r="8" fill="#1e293b" opacity="0.8"/>
    </g>
    
    {/* Bottom-left */}
    <g transform="translate(150, 450)">
      <circle r="16" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1" opacity="0.9"/>
      <circle r="16" fill="none" stroke="url(#rimLight)" stroke-width="2" opacity="0.6"/>
      <circle r="8" fill="#1e293b" opacity="0.8"/>
    </g>
    
    {/* Left */}
    <g transform="translate(92, 300)">
      <circle r="16" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1" opacity="0.9"/>
      <circle r="16" fill="none" stroke="url(#rimLight)" stroke-width="2" opacity="0.6"/>
      <circle r="8" fill="#1e293b" opacity="0.8"/>
    </g>
    
    {/* Top-left */}
    <g transform="translate(150, 150)">
      <circle r="16" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1" opacity="0.9"/>
      <circle r="16" fill="none" stroke="url(#rimLight)" stroke-width="2" opacity="0.6"/>
      <circle r="8" fill="#1e293b" opacity="0.8"/>
    </g>
  </g>
  
  {/* Worker nodes - Ring 1 (inner) */}
  <g className="worker-nodes-inner">
    {/* Top */}
    <g transform="translate(300, 180)">
      <circle r="20" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1.5" opacity="0.95"/>
      <circle r="20" fill="none" stroke="url(#rimLight)" stroke-width="2.5" opacity="0.7"/>
      <circle r="10" fill="#1e293b" opacity="0.9"/>
    </g>
    
    {/* Top-right */}
    <g transform="translate(404, 240)">
      <circle r="20" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1.5" opacity="0.95"/>
      <circle r="20" fill="none" stroke="url(#rimLight)" stroke-width="2.5" opacity="0.7"/>
      <circle r="10" fill="#1e293b" opacity="0.9"/>
    </g>
    
    {/* Bottom-right */}
    <g transform="translate(404, 360)">
      <circle r="20" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1.5" opacity="0.95"/>
      <circle r="20" fill="none" stroke="url(#rimLight)" stroke-width="2.5" opacity="0.7"/>
      <circle r="10" fill="#1e293b" opacity="0.9"/>
    </g>
    
    {/* Bottom */}
    <g transform="translate(300, 420)">
      <circle r="20" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1.5" opacity="0.95"/>
      <circle r="20" fill="none" stroke="url(#rimLight)" stroke-width="2.5" opacity="0.7"/>
      <circle r="10" fill="#1e293b" opacity="0.9"/>
    </g>
    
    {/* Bottom-left */}
    <g transform="translate(196, 360)">
      <circle r="20" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1.5" opacity="0.95"/>
      <circle r="20" fill="none" stroke="url(#rimLight)" stroke-width="2.5" opacity="0.7"/>
      <circle r="10" fill="#1e293b" opacity="0.9"/>
    </g>
    
    {/* Top-left */}
    <g transform="translate(196, 240)">
      <circle r="20" fill="url(#nodeGradient)" stroke="#f59e0b" stroke-width="1.5" opacity="0.95"/>
      <circle r="20" fill="none" stroke="url(#rimLight)" stroke-width="2.5" opacity="0.7"/>
      <circle r="10" fill="#1e293b" opacity="0.9"/>
    </g>
  </g>
  
  {/* Central Queen node */}
  <g transform="translate(300, 300)" filter="url(#glow)">
    {/* Outer glow ring */}
    <circle r="60" fill="url(#amberGlow)" opacity="0.3">
      <animate attributeName="r" values="60;65;60" dur="3s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.3;0.4;0.3" dur="3s" repeatCount="indefinite"/>
    </circle>
    
    {/* Main node body */}
    <circle r="45" fill="url(#queenGradient)" stroke="#f59e0b" stroke-width="2.5" opacity="1"/>
    
    {/* Rim lighting effect */}
    <circle r="45" fill="none" stroke="url(#rimLight)" stroke-width="4" opacity="0.8"/>
    
    {/* Inner hexagon detail */}
    <path d="M 0,-25 L 21.65,-12.5 L 21.65,12.5 L 0,25 L -21.65,12.5 L -21.65,-12.5 Z" 
          fill="none" stroke="#f59e0b" stroke-width="2" opacity="0.6"/>
    
    {/* Center core */}
    <circle r="15" fill="#0f172a" opacity="0.9"/>
    <circle r="8" fill="#f59e0b" opacity="0.8">
      <animate attributeName="opacity" values="0.8;1;0.8" dur="2s" repeatCount="indefinite"/>
    </circle>
  </g>
  
  {/* Subtle data particles */}
  <g opacity="0.4">
    <circle cx="300" cy="140" r="2" fill="#f59e0b">
      <animateMotion path="M 0,0 L 0,140" dur="2s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;0" dur="2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="380" cy="200" r="2" fill="#f59e0b">
      <animateMotion path="M 0,0 L -80,85" dur="2.5s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;0" dur="2.5s" repeatCount="indefinite"/>
    </circle>
    <circle cx="220" cy="400" r="2" fill="#f59e0b">
      <animateMotion path="M 0,0 L 80,-85" dur="2.2s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;0" dur="2.2s" repeatCount="indefinite"/>
    </circle>
  </g>
    </svg>
  )
}
