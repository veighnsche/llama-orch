import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface TemplateBackgroundProps {
  /** Background variant */
  variant?:
    | 'none'
    | 'background'
    | 'secondary'
    | 'card'
    | 'muted'
    | 'accent'
    | 'primary'
    | 'destructive'
    | 'subtle-border'
    | 'gradient-primary'
    | 'gradient-secondary'
    | 'gradient-destructive'
    | 'gradient-radial'
    | 'gradient-mesh'
    | 'gradient-warm'
    | 'gradient-cool'
    | 'pattern-dots'
    | 'pattern-grid'
    | 'pattern-honeycomb'
    | 'pattern-waves'
    | 'pattern-circuit'
    | 'pattern-diagonal'
  /** Optional custom decoration element (e.g., SVG patterns, shapes) */
  decoration?: ReactNode
  /** Optional overlay opacity (0-100) */
  overlayOpacity?: number
  /** Optional overlay color */
  overlayColor?: 'black' | 'white' | 'primary' | 'secondary'
  /** Enable blur effect on background */
  blur?: boolean
  /** Pattern size for pattern variants */
  patternSize?: 'small' | 'medium' | 'large'
  /** Pattern opacity (0-100) */
  patternOpacity?: number
  /** Additional CSS classes */
  className?: string
  /** Content to render on top of background */
  children: ReactNode
}

const variantClasses = {
  none: '',
  background: 'bg-background',
  secondary: 'bg-secondary',
  card: 'bg-card',
  muted: 'bg-muted',
  accent: 'bg-accent',
  primary: 'bg-primary text-primary-foreground',
  destructive: 'bg-destructive text-destructive-foreground',
  'subtle-border':
    'bg-background relative before:absolute before:inset-x-0 before:top-0 before:h-px before:bg-border/60',
  'gradient-primary': 'bg-gradient-to-b from-background via-primary/8 to-background',
  'gradient-secondary': 'bg-gradient-to-b from-background via-secondary/10 to-background',
  'gradient-destructive': 'bg-gradient-to-b from-background via-destructive/8 to-background',
  'gradient-radial':
    'bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/20 via-background to-background',
  'gradient-mesh': 'bg-gradient-to-br from-primary/10 via-background to-secondary/10',
  'gradient-warm': 'bg-gradient-to-br from-amber-500/5 via-background to-orange-500/5',
  'gradient-cool': 'bg-gradient-to-br from-blue-500/5 via-background to-cyan-500/5',
  'pattern-dots': 'bg-background',
  'pattern-grid': 'bg-background',
  'pattern-honeycomb': 'bg-background',
  'pattern-waves': 'bg-background',
  'pattern-circuit': 'bg-background',
  'pattern-diagonal': 'bg-background',
} as const

const overlayColorClasses = {
  black: 'bg-black',
  white: 'bg-white',
  primary: 'bg-primary',
  secondary: 'bg-secondary',
} as const

/**
 * TemplateBackground organism - Comprehensive background control for templates
 *
 * Provides full control over template backgrounds including:
 * - Solid colors (background, secondary, card, muted, accent)
 * - Brand colors (primary, destructive)
 * - Gradients (primary, secondary, destructive, radial, mesh, warm, cool)
 * - SVG Patterns (dots, grid, honeycomb, waves, circuit, diagonal)
 * - Decorations (custom SVG patterns, shapes)
 * - Overlays (with opacity and color control)
 * - Blur effects
 *
 * @example
 * <TemplateBackground variant="pattern-honeycomb" patternOpacity={10}>
 *   <YourContent />
 * </TemplateBackground>
 *
 * @example
 * <TemplateBackground variant="gradient-primary" decoration={<CustomPattern />}>
 *   <YourContent />
 * </TemplateBackground>
 */
export function TemplateBackground({
  variant = 'background',
  decoration,
  overlayOpacity,
  overlayColor = 'black',
  blur = false,
  patternSize = 'medium',
  patternOpacity = 8,
  className,
  children,
}: TemplateBackgroundProps) {
  const hasOverlay = overlayOpacity !== undefined && overlayOpacity > 0
  const isPatternVariant = variant?.startsWith('pattern-')
  const patternType = isPatternVariant ? variant.replace('pattern-', '') : null

  // Generate pattern decoration based on variant
  const renderPattern = () => {
    if (!isPatternVariant || !patternType) return null

    const opacity = patternOpacity / 100
    const patternId = `pattern-${patternType}-${Math.random().toString(36).substr(2, 9)}`

    switch (patternType) {
      case 'dots':
        return (
          <svg className="absolute inset-0 w-full h-full" aria-hidden="true">
            <defs>
              <pattern
                id={patternId}
                x="0"
                y="0"
                width={patternSize === 'small' ? '20' : patternSize === 'large' ? '40' : '30'}
                height={patternSize === 'small' ? '20' : patternSize === 'large' ? '40' : '30'}
                patternUnits="userSpaceOnUse"
              >
                <circle
                  cx={patternSize === 'small' ? '10' : patternSize === 'large' ? '20' : '15'}
                  cy={patternSize === 'small' ? '10' : patternSize === 'large' ? '20' : '15'}
                  r="1"
                  fill="currentColor"
                  opacity={opacity}
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill={`url(#${patternId})`} className="text-muted-foreground" />
          </svg>
        )

      case 'grid':
        return (
          <svg className="absolute inset-0 w-full h-full" aria-hidden="true">
            <defs>
              <pattern
                id={patternId}
                x="0"
                y="0"
                width={patternSize === 'small' ? '20' : patternSize === 'large' ? '60' : '40'}
                height={patternSize === 'small' ? '20' : patternSize === 'large' ? '60' : '40'}
                patternUnits="userSpaceOnUse"
              >
                <path
                  d={`M ${patternSize === 'small' ? '20' : patternSize === 'large' ? '60' : '40'} 0 L 0 0 0 ${patternSize === 'small' ? '20' : patternSize === 'large' ? '60' : '40'}`}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="0.5"
                  opacity={opacity}
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill={`url(#${patternId})`} className="text-muted-foreground" />
          </svg>
        )

      case 'honeycomb': {
        const cellSize = patternSize === 'small' ? 40 : patternSize === 'large' ? 112 : 70
        const cellHeight = patternSize === 'small' ? 69.28 : patternSize === 'large' ? 200 : 121.24
        return (
          <svg className="absolute inset-0 w-full h-full" aria-hidden="true">
            <defs>
              <pattern id={patternId} x="0" y="0" width={cellSize} height={cellHeight} patternUnits="userSpaceOnUse">
                <path
                  d={`M${cellSize / 2} ${cellHeight * 0.68}L0 ${cellHeight * 0.515}L0 ${cellHeight * 0.165}L${cellSize / 2} 0L${cellSize} ${cellHeight * 0.165}L${cellSize} ${cellHeight * 0.515}L${cellSize / 2} ${cellHeight * 0.68}L${cellSize / 2} ${cellHeight}`}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="0.5"
                  opacity={opacity}
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill={`url(#${patternId})`} className="text-muted-foreground" />
          </svg>
        )
      }

      case 'waves':
        return (
          <svg className="absolute inset-0 w-full h-full" aria-hidden="true">
            <defs>
              <pattern
                id={patternId}
                x="0"
                y="0"
                width={patternSize === 'small' ? '60' : patternSize === 'large' ? '120' : '90'}
                height={patternSize === 'small' ? '30' : patternSize === 'large' ? '60' : '45'}
                patternUnits="userSpaceOnUse"
              >
                <path
                  d={`M0 ${patternSize === 'small' ? '15' : patternSize === 'large' ? '30' : '22.5'} Q ${patternSize === 'small' ? '15' : patternSize === 'large' ? '30' : '22.5'} ${patternSize === 'small' ? '7.5' : patternSize === 'large' ? '15' : '11.25'} ${patternSize === 'small' ? '30' : patternSize === 'large' ? '60' : '45'} ${patternSize === 'small' ? '15' : patternSize === 'large' ? '30' : '22.5'} T ${patternSize === 'small' ? '60' : patternSize === 'large' ? '120' : '90'} ${patternSize === 'small' ? '15' : patternSize === 'large' ? '30' : '22.5'}`}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="0.5"
                  opacity={opacity}
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill={`url(#${patternId})`} className="text-muted-foreground" />
          </svg>
        )

      case 'circuit':
        return (
          <svg className="absolute inset-0 w-full h-full" aria-hidden="true">
            <defs>
              <pattern
                id={patternId}
                x="0"
                y="0"
                width={patternSize === 'small' ? '40' : patternSize === 'large' ? '80' : '60'}
                height={patternSize === 'small' ? '40' : patternSize === 'large' ? '80' : '60'}
                patternUnits="userSpaceOnUse"
              >
                <circle
                  cx={patternSize === 'small' ? '10' : patternSize === 'large' ? '20' : '15'}
                  cy={patternSize === 'small' ? '10' : patternSize === 'large' ? '20' : '15'}
                  r="1.5"
                  fill="currentColor"
                  opacity={opacity}
                />
                <circle
                  cx={patternSize === 'small' ? '30' : patternSize === 'large' ? '60' : '45'}
                  cy={patternSize === 'small' ? '30' : patternSize === 'large' ? '60' : '45'}
                  r="1.5"
                  fill="currentColor"
                  opacity={opacity}
                />
                <path
                  d={`M${patternSize === 'small' ? '10' : patternSize === 'large' ? '20' : '15'} ${patternSize === 'small' ? '10' : patternSize === 'large' ? '20' : '15'} L${patternSize === 'small' ? '30' : patternSize === 'large' ? '60' : '45'} ${patternSize === 'small' ? '10' : patternSize === 'large' ? '20' : '15'} L${patternSize === 'small' ? '30' : patternSize === 'large' ? '60' : '45'} ${patternSize === 'small' ? '30' : patternSize === 'large' ? '60' : '45'}`}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="0.5"
                  opacity={opacity * 0.5}
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill={`url(#${patternId})`} className="text-muted-foreground" />
          </svg>
        )

      case 'diagonal':
        return (
          <svg className="absolute inset-0 w-full h-full" aria-hidden="true">
            <defs>
              <pattern
                id={patternId}
                x="0"
                y="0"
                width={patternSize === 'small' ? '20' : patternSize === 'large' ? '40' : '30'}
                height={patternSize === 'small' ? '20' : patternSize === 'large' ? '40' : '30'}
                patternUnits="userSpaceOnUse"
                patternTransform="rotate(45)"
              >
                <line
                  x1="0"
                  y1="0"
                  x2="0"
                  y2={patternSize === 'small' ? '20' : patternSize === 'large' ? '40' : '30'}
                  stroke="currentColor"
                  strokeWidth="0.5"
                  opacity={opacity}
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill={`url(#${patternId})`} className="text-muted-foreground" />
          </svg>
        )

      default:
        return null
    }
  }

  return (
    <div className={cn('relative', variantClasses[variant], className)}>
      {/* Built-in pattern layer */}
      {isPatternVariant && (
        <div className={cn('absolute inset-0 overflow-hidden', blur && 'blur-sm')} aria-hidden="true">
          {renderPattern()}
        </div>
      )}

      {/* Custom decoration layer */}
      {decoration && (
        <div className={cn('absolute inset-0 overflow-hidden', blur && 'blur-sm')} aria-hidden="true">
          {decoration}
        </div>
      )}

      {/* Overlay layer */}
      {hasOverlay && (
        <div
          className={cn('absolute inset-0', overlayColorClasses[overlayColor])}
          style={{ opacity: overlayOpacity / 100 }}
          aria-hidden="true"
        />
      )}

      {/* Content layer */}
      <div className="relative z-10">{children}</div>
    </div>
  )
}
