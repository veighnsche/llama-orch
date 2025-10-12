import { cn } from '@/lib/utils'

export interface TestimonialCardProps {
  /** Person's name */
  name: string
  /** Person's role */
  role: string
  /** Testimonial quote */
  quote: string
  /** Avatar (image URL or gradient colors) */
  avatar?: string | { from: string; to: string }
  /** Additional CSS classes */
  className?: string
}

export function TestimonialCard({
  name,
  role,
  quote,
  avatar,
  className,
}: TestimonialCardProps) {
  const renderAvatar = () => {
    if (!avatar) {
      return (
        <div className="h-12 w-12 rounded-full bg-gradient-to-br from-primary to-chart-2"></div>
      )
    }

    if (typeof avatar === 'string') {
      return (
        <img
          src={avatar}
          alt={name}
          className="h-12 w-12 rounded-full object-cover"
        />
      )
    }

    return (
      <div
        className={cn(
          'h-12 w-12 rounded-full bg-gradient-to-br',
          `from-${avatar.from} to-${avatar.to}`
        )}
      ></div>
    )
  }

  return (
    <div
      className={cn(
        'bg-card border border-border rounded-lg p-6 space-y-4',
        className
      )}
    >
      <div className="flex items-center gap-3">
        {renderAvatar()}
        <div>
          <div className="font-bold text-card-foreground">{name}</div>
          <div className="text-sm text-muted-foreground">{role}</div>
        </div>
      </div>
      <p className="text-muted-foreground leading-relaxed">{quote}</p>
    </div>
  )
}
