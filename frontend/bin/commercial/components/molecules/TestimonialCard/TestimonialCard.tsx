import { cn } from '@/lib/utils'
import Image from 'next/image'

export interface TestimonialCardProps {
  /** Person's name */
  name: string
  /** Person's role */
  role: string
  /** Testimonial quote */
  quote: string
  /** Avatar (image URL or gradient colors) */
  avatar?: string | { from: string; to: string }
  /** Company information */
  company?: { name: string; logo?: string }
  /** Verified badge */
  verified?: boolean
  /** Source link (tweet, GH issue, blog) */
  link?: string
  /** Date of testimonial (ISO or human string) */
  date?: string
  /** Rating (1-5 stars) */
  rating?: 1 | 2 | 3 | 4 | 5
  /** Highlight badge (e.g., "$500/mo → $0") */
  highlight?: string
  /** Additional CSS classes */
  className?: string
}

export function TestimonialCard({
  name,
  role,
  quote,
  avatar,
  company,
  verified,
  link,
  date,
  rating,
  highlight,
  className,
}: TestimonialCardProps) {
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
      return (
        <div className="h-12 w-12 rounded-full bg-gradient-to-br from-primary to-chart-2 motion-safe:animate-in motion-safe:fade-in motion-safe:duration-300"></div>
      )
    }

    if (typeof avatar === 'string') {
      return (
        <img
          src={avatar}
          alt={name}
          className="h-12 w-12 rounded-full object-cover motion-safe:animate-in motion-safe:fade-in motion-safe:duration-300"
        />
      )
    }

    const gradientKey = `${avatar.from}-${avatar.to}` as keyof typeof gradientClasses
    const gradient = gradientClasses[gradientKey] || gradientClasses.primary

    return (
      <div
        className={cn(
          'h-12 w-12 rounded-full bg-gradient-to-br motion-safe:animate-in motion-safe:fade-in motion-safe:duration-300',
          gradient
        )}
      ></div>
    )
  }

  const renderStars = () => {
    if (!rating) return null
    return (
      <div className="flex gap-0.5" aria-label={`Rating: ${rating} out of 5 stars`}>
        {Array.from({ length: 5 }).map((_, i) => (
          <svg
            key={i}
            className={cn('h-3 w-3', i < rating ? 'text-amber-500' : 'text-muted-foreground/30')}
            fill="currentColor"
            viewBox="0 0 20 20"
            aria-hidden="true"
          >
            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
          </svg>
        ))}
      </div>
    )
  }

  return (
    <article
      className={cn(
        'bg-card/90 border border-border rounded-xl p-6 flex flex-col gap-4',
        'hover:border-primary/40 motion-safe:hover:translate-y-[-2px] motion-safe:hover:shadow-lg',
        'motion-safe:transition-all motion-safe:duration-200',
        className
      )}
      itemScope
      itemType="https://schema.org/Review"
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3 flex-1 min-w-0">
          {renderAvatar()}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-bold text-card-foreground" itemProp="author">
                {name}
              </span>
              {company?.logo && (
                <Image src={company.logo} alt={company.name} width={24} height={24} className="object-contain" />
              )}
            </div>
            <div className="text-sm text-muted-foreground">{role}</div>
            {company?.name && !company.logo && <div className="text-xs text-muted-foreground/80">{company.name}</div>}
          </div>
        </div>
        {verified && (
          <span className="text-[11px] bg-primary/10 text-primary px-2 py-0.5 rounded-full whitespace-nowrap font-medium">
            Verified
          </span>
        )}
      </div>

      {/* Rating */}
      {renderStars()}

      {/* Quote block */}
      <blockquote className="flex-1">
        <p className="text-sm leading-6 text-muted-foreground line-clamp-6 md:line-clamp-none" itemProp="reviewBody">
          <span className="text-primary mr-1">&ldquo;</span>
          {quote}
        </p>
      </blockquote>

      {/* Footer row */}
      {(highlight || date || link) && (
        <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground/80 flex-wrap">
          <div className="flex items-center gap-2">
            {highlight && (
              <span className="bg-chart-3/10 text-chart-3 px-2 py-1 rounded font-medium whitespace-nowrap">
                {highlight}
              </span>
            )}
            {date && <time dateTime={date}>{date}</time>}
          </div>
          {link && (
            <a
              href={link}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary transition-colors underline"
            >
              Source
            </a>
          )}
        </div>
      )}
    </article>
  )
}
