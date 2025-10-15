import { SectionContainer, type StatItem, StatsGrid, TestimonialCard } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import Image from 'next/image'

export type Testimonial = {
  quote: string
  author: string
  role?: string
  avatar?: string // emoji | URL | initials
  companyLogoSrc?: string // optional Next.js Image
}

export type TestimonialsSectionProps = {
  title: string
  subtitle?: string
  testimonials: Testimonial[]
  stats?: StatItem[] // up to 4
  id?: string
  className?: string
}

export function TestimonialsSection({ title, subtitle, testimonials, stats, id, className }: TestimonialsSectionProps) {
  const hasLogos = testimonials.some((t) => t.companyLogoSrc)

  return (
    <SectionContainer
      title={title}
      description={subtitle}
      align="center"
      maxWidth="7xl"
      paddingY="2xl"
      className={cn(
        'border-b border-border motion-safe:animate-in motion-safe:fade-in-50 motion-safe:duration-400',
        className,
      )}
      headingId={id}
    >
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
        {testimonials.map((testimonial, index) => (
          <TestimonialCard
            key={index}
            name={testimonial.author}
            role={testimonial.role || ''}
            quote={testimonial.quote}
            avatar={testimonial.avatar}
            company={testimonial.companyLogoSrc ? { name: '', logo: testimonial.companyLogoSrc } : undefined}
          />
        ))}
      </div>

      {/* Stats row */}
      {stats && stats.length > 0 && (
        <StatsGrid stats={stats} variant="cards" columns={4} className="mx-auto mt-12 max-w-4xl" />
      )}
    </SectionContainer>
  )
}
