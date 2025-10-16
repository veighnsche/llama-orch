import {
  Avatar,
  AvatarFallback,
  AvatarImage,
  Badge,
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  QuoteBlock,
} from '@rbee/ui/atoms'
import { RatingStars } from '@rbee/ui/atoms/RatingStars'
import { cn } from '@rbee/ui/utils'
import Image from 'next/image'

export interface TestimonialCardProps {
  /** Person's name */
  name: string
  /** Person's role */
  role: string
  /** Testimonial quote */
  quote: string
  /** Avatar (image URL, emoji, or initials) */
  avatar?: string
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
  // Check if avatar is emoji or initials
  const isEmoji = avatar && /^[\p{Emoji}\u200d]+$/u.test(avatar)
  const isInitials = avatar && /^[A-Z]{1,2}$/.test(avatar)
  const isImageUrl = avatar && !isEmoji && !isInitials

  return (
    <Card
      className={cn('hover:border-primary/40 hover:shadow-lg transition-all duration-200', className)}
      itemScope
      itemType="https://schema.org/Review"
    >
      <CardHeader className="pb-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <Avatar className="h-12 w-12">
              {isImageUrl && <AvatarImage src={avatar} alt={name} />}
              <AvatarFallback
                className={cn(
                  isEmoji || isInitials
                    ? 'bg-gradient-to-br from-primary/10 to-chart-2/10 text-2xl'
                    : 'bg-gradient-to-br from-primary to-chart-2',
                )}
              >
                {isEmoji || isInitials ? avatar : name.slice(0, 2).toUpperCase()}
              </AvatarFallback>
            </Avatar>
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
            <Badge variant="outline" className="bg-primary/10 text-primary border-transparent">
              Verified
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Rating */}
        {rating && <RatingStars rating={rating} size="sm" />}

        {/* Quote block */}
        <QuoteBlock size="sm" showClosingQuote itemProp="reviewBody">
          {quote}
        </QuoteBlock>
      </CardContent>

      {/* Footer row */}
      {(highlight || date || link) && (
        <CardFooter className="flex-wrap gap-3 text-xs text-muted-foreground/80">
          <div className="flex items-center gap-2">
            {highlight && (
              <Badge variant="outline" className="bg-chart-3/10 text-chart-3 border-transparent">
                {highlight}
              </Badge>
            )}
            {date && <time dateTime={date}>{date}</time>}
          </div>
          {link && (
            <a
              href={link}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary transition-colors underline ml-auto"
            >
              Source
            </a>
          )}
        </CardFooter>
      )}
    </Card>
  )
}
