import { cn } from '@/lib/utils'
import Image from 'next/image'

export type Testimonial = {
  quote: string
  author: string
  role?: string
  avatar?: string // emoji | URL | initials
  companyLogoSrc?: string // optional Next.js Image
}

export type Stat = {
  label: string
  value: string
  tone?: 'default' | 'primary'
}

export type TestimonialsSectionProps = {
  title: string
  subtitle?: string
  testimonials: Testimonial[]
  stats?: Stat[] // up to 4
  id?: string
  className?: string
}

function getAvatarContent(avatar?: string) {
  if (!avatar) return '👤'
  // Check if emoji (single char or emoji sequence)
  if (/^[\p{Emoji}\u200d]+$/u.test(avatar)) return avatar
  // Check if initials (1-2 uppercase letters)
  if (/^[A-Z]{1,2}$/.test(avatar)) return avatar
  // Otherwise treat as URL
  return null
}

export function TestimonialsSection({
  title,
  subtitle,
  testimonials,
  stats,
  id,
  className,
}: TestimonialsSectionProps) {
  const hasLogos = testimonials.some((t) => t.companyLogoSrc)

  return (
    <section
      id={id}
      className={cn(
        'border-b border-border py-24 motion-safe:animate-in motion-safe:fade-in-50 motion-safe:duration-400',
        className,
      )}
    >
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        {/* Heading block */}
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">{title}</h2>
          {subtitle && <p className="mt-4 text-lg text-muted-foreground">{subtitle}</p>}
        </div>

        {/* Optional logo strip */}
        {hasLogos && (
          <div className="mx-auto mt-8 flex max-w-4xl flex-wrap items-center justify-center gap-8">
            {testimonials
              .filter((t) => t.companyLogoSrc)
              .slice(0, 6)
              .map((t, i) => (
                <div key={i} className="opacity-70 transition hover:opacity-100">
                  <Image
                    src={t.companyLogoSrc!}
                    alt={`Monochrome company logo; subtle, flat, high-contrast for dark UI`}
                    width={120}
                    height={40}
                    className="h-10 w-auto object-contain grayscale"
                  />
                </div>
              ))}
          </div>
        )}

        {/* Quotes grid */}
        <div className="mx-auto mt-16 grid max-w-6xl gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {testimonials.map((testimonial, index) => {
            const avatarContent = getAvatarContent(testimonial.avatar)
            const isUrl = testimonial.avatar && !avatarContent
            const delay = 80 * (index + 1)

            return (
              <article
                key={index}
                tabIndex={0}
                className={cn(
                  'rounded-xl border border-border/80 bg-card p-6 transition hover:bg-card/80',
                  'focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:outline-none',
                  'motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400',
                )}
                style={{ animationDelay: `${delay}ms` }}
              >
                {/* Top row: optional avatar */}
                {testimonial.avatar && (
                  <div className="mb-4 flex justify-end">
                    {isUrl ? (
                      <Image
                        src={testimonial.avatar}
                        alt={`Portrait of ${testimonial.author}${testimonial.role ? `, ${testimonial.role}` : ''}`}
                        width={40}
                        height={40}
                        className="h-10 w-10 rounded-full object-cover"
                      />
                    ) : (
                      <div className="flex h-10 w-10 items-center justify-center text-2xl" aria-hidden="true">
                        {avatarContent}
                      </div>
                    )}
                  </div>
                )}

                {/* Quote with large inline opening mark */}
                <blockquote className="relative">
                  <span className="absolute -left-1 -top-2 text-5xl font-serif leading-none text-primary/40" aria-hidden="true">
                    &ldquo;
                  </span>
                  <p className="pl-6 text-balance leading-relaxed text-muted-foreground">{testimonial.quote}</p>
                </blockquote>

                {/* Author block */}
                <div className="mt-4">
                  <div className="flex items-center gap-2">
                    <cite className="not-italic font-semibold text-card-foreground">{testimonial.author}</cite>
                    {testimonial.companyLogoSrc && (
                      <Image
                        src={testimonial.companyLogoSrc}
                        alt="Company logo"
                        width={20}
                        height={20}
                        className="h-5 w-auto object-contain opacity-70"
                      />
                    )}
                  </div>
                  {testimonial.role && <div className="text-sm text-muted-foreground">{testimonial.role}</div>}
                </div>
              </article>
            )
          })}
        </div>

        {/* Stats row */}
        {stats && stats.length > 0 && (
          <div className="mx-auto mt-12 grid max-w-4xl gap-6 sm:grid-cols-4">
            {stats.map((stat, index) => {
              const toneClass = stat.tone === 'primary' ? 'text-primary' : 'text-foreground'
              const delay = 200 + 50 * index
              const ariaLabel = `${stat.value} ${stat.label}`

              return (
                <div
                  key={index}
                  className="text-center motion-safe:animate-in motion-safe:fade-in-50 motion-safe:duration-300"
                  style={{ animationDelay: `${delay}ms` }}
                >
                  <div className={cn('text-3xl font-bold', toneClass)} aria-label={ariaLabel}>
                    {stat.value}
                  </div>
                  <div className="text-sm text-muted-foreground">{stat.label}</div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </section>
  )
}

// Backward-compatible wrapper for homepage
export function SocialProofSection() {
  return (
    <TestimonialsSection
      title="Trusted by developers who value independence"
      testimonials={[
        {
          avatar: '👨‍💻',
          author: 'Alex K.',
          role: 'Solo Developer',
          quote:
            'Spent $80/mo on Claude. Now I run Llama-70B on my gaming PC + old workstation. Same quality, $0 cost.',
        },
        {
          avatar: '👩‍💼',
          author: 'Sarah M.',
          role: 'CTO',
          quote:
            "We pooled our team's hardware and cut AI spend from $500/mo to zero. OpenAI-compatible API—no code changes.",
        },
        {
          avatar: '👨‍🔧',
          author: 'Marcus T.',
          role: 'DevOps',
          quote: 'Cascading shutdown ends orphaned processes and VRAM leaks. Ctrl+C and everything cleans up.',
        },
      ]}
      stats={[
        { value: '1,200+', label: 'GitHub stars' },
        { value: '500+', label: 'Active installations' },
        { value: '8,000+', label: 'GPUs orchestrated' },
        { value: '€0', label: 'Avg. monthly cost', tone: 'primary' },
      ]}
    />
  )
}
