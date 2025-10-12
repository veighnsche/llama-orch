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

export function TestimonialCard({ name, role, quote, avatar, className }: TestimonialCardProps) {
  const gradientClasses = {
    primary: 'from-primary to-primary',
    'chart-1': 'from-chart-1 to-chart-1',
    'chart-2': 'from-chart-2 to-chart-2',
    'chart-3': 'from-chart-3 to-chart-3',
    'chart-4': 'from-chart-4 to-chart-4',
    'chart-5': 'from-chart-5 to-chart-5',
    'primary-chart-2': 'from-primary to-chart-2',
    'chart-1-chart-3': 'from-chart-1 to-chart-3',
    'chart-2-chart-4': 'from-chart-2 to-chart-4',
  }

  const renderAvatar = () => {
    if (!avatar) {
      return <div className="h-12 w-12 rounded-full bg-gradient-to-br from-primary to-chart-2"></div>
    }

    if (typeof avatar === 'string') {
      return <img src={avatar} alt={name} className="h-12 w-12 rounded-full object-cover" />
    }

    const gradientKey = `${avatar.from}-${avatar.to}` as keyof typeof gradientClasses
    const gradient = gradientClasses[gradientKey] || gradientClasses.primary

    return <div className={cn('h-12 w-12 rounded-full bg-gradient-to-br', gradient)}></div>
  }

  return (
    <div className={cn('bg-card border border-border rounded-lg p-6 space-y-4', className)}>
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
