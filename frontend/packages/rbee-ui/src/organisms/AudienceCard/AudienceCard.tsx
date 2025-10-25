import { Card, CardContent } from '@rbee/ui/atoms/Card'
import { BulletListItem } from '@rbee/ui/molecules/BulletListItem'
import { ButtonCardFooter } from '@rbee/ui/molecules/ButtonCardFooter'
import { cn, parseInlineMarkdown } from '@rbee/ui/utils'
import { cva } from 'class-variance-authority'
import type { ReactNode } from 'react'

const _audienceCardVariants = cva('', {
  variants: {
    color: {
      primary: '',
      'chart-1': '',
      'chart-2': '',
      'chart-3': '',
      'chart-4': '',
      'chart-5': '',
    },
  },
  defaultVariants: {
    color: 'primary',
  },
})

export type AudienceCardColor = 'primary' | 'chart-1' | 'chart-2' | 'chart-3' | 'chart-4' | 'chart-5'

const cardContainerVariants = cva(
  'border border-border group relative overflow-hidden transition-all duration-300 hover:scale-[1.02]',
  {
    variants: {
      color: {
        primary: 'hover:border-primary/50',
        'chart-1': 'hover:border-chart-1/50',
        'chart-2': 'hover:border-chart-2/50',
        'chart-3': 'hover:border-chart-3/50',
        'chart-4': 'hover:border-chart-4/50',
        'chart-5': 'hover:border-chart-5/50',
      },
    },
    defaultVariants: {
      color: 'primary',
    },
  },
)

const gradientVariants = cva(
  'absolute inset-0 -z-10 bg-gradient-to-br opacity-0 transition-all duration-500 group-hover:to-transparent group-hover:opacity-100',
  {
    variants: {
      color: {
        primary: 'from-primary/0 via-primary/0 to-primary/0 group-hover:from-primary/5 group-hover:via-primary/10',
        'chart-1': 'from-chart-1/0 via-chart-1/0 to-chart-1/0 group-hover:from-chart-1/5 group-hover:via-chart-1/10',
        'chart-2': 'from-chart-2/0 via-chart-2/0 to-chart-2/0 group-hover:from-chart-2/5 group-hover:via-chart-2/10',
        'chart-3': 'from-chart-3/0 via-chart-3/0 to-chart-3/0 group-hover:from-chart-3/5 group-hover:via-chart-3/10',
        'chart-4': 'from-chart-4/0 via-chart-4/0 to-chart-4/0 group-hover:from-chart-4/5 group-hover:via-chart-4/10',
        'chart-5': 'from-chart-5/0 via-chart-5/0 to-chart-5/0 group-hover:from-chart-5/5 group-hover:via-chart-5/10',
      },
    },
    defaultVariants: {
      color: 'primary',
    },
  },
)

const iconBgVariants = cva(
  'flex h-14 w-14 shrink-0 items-center justify-center rounded bg-gradient-to-br shadow-lg',
  {
    variants: {
      color: {
        primary: 'from-primary to-primary',
        'chart-1': 'from-chart-1 to-chart-1',
        'chart-2': 'from-chart-2 to-chart-2',
        'chart-3': 'from-chart-3 to-chart-3',
        'chart-4': 'from-chart-4 to-chart-4',
        'chart-5': 'from-chart-5 to-chart-5',
      },
    },
    defaultVariants: {
      color: 'primary',
    },
  },
)

const textVariants = cva('', {
  variants: {
    color: {
      primary: 'text-primary',
      'chart-1': 'text-chart-1',
      'chart-2': 'text-chart-2',
      'chart-3': 'text-chart-3',
      'chart-4': 'text-chart-4',
      'chart-5': 'text-chart-5',
    },
  },
  defaultVariants: {
    color: 'primary',
  },
})

export interface AudienceCardProps {
  icon: ReactNode
  category: string
  title: string
  description: string
  features: string[]
  href: string
  ctaText: string
  color?: AudienceCardColor
  imageSlot?: ReactNode
  badgeSlot?: ReactNode
  decisionLabel?: string
  showGradient?: boolean
}

export function AudienceCard({
  icon,
  category,
  title,
  description,
  features,
  href,
  ctaText,
  color = 'primary',
  imageSlot,
  badgeSlot,
  decisionLabel,
  showGradient = false,
}: AudienceCardProps) {
  const descriptionId = `${title.toLowerCase().replace(/\s+/g, '-')}-description`

  return (
    <>
      {showGradient && (
        <div
          className="pointer-events-none absolute inset-x-0 top-0 h-[600px] opacity-40 -z-10"
          style={{
            background: 'radial-gradient(ellipse 80% 50% at 50% 0%, hsl(var(--primary) / 0.05), transparent)',
          }}
          aria-hidden="true"
        />
      )}
      <div className="flex h-full flex-col">
        {/* Optional decision label above card */}
        {decisionLabel && (
          <div className={cn('mb-3 text-sm font-medium font-sans', textVariants({ color }))}>{decisionLabel}</div>
        )}

        <Card className={cn(cardContainerVariants({ color }), 'flex flex-1 flex-col p-6')}>
          <div className={gradientVariants({ color })} />

          <CardContent className="flex flex-1 flex-col gap-0 p-0">
            {/* Icons side-by-side at top */}
            <div className="mb-6 flex items-top gap-3 min-h-[64px]">
              <div className={iconBgVariants({ color })}>
                <div className="size-6 text-primary-foreground" aria-hidden="true">
                  {icon}
                </div>
              </div>
              {imageSlot && (
                <div className="h-14 w-14 shrink-0 overflow-hidden rounded bg-card ring-1 ring-border">
                  {imageSlot}
                </div>
              )}
            </div>

            <div className={cn('mb-2 text-sm font-medium uppercase tracking-wider font-sans', textVariants({ color }))}>
              {category}
            </div>
            <h3 className="mb-3 min-h-[64px] text-2xl font-semibold text-card-foreground">{title}</h3>
            <p
              id={descriptionId}
              className="mb-6 min-h-[72px] text-sm leading-relaxed text-muted-foreground sm:text-base"
            >
              {parseInlineMarkdown(description)}
            </p>

            <ul className="mb-8 h-[120px] space-y-3">
              {features.map((feature, index) => (
                <BulletListItem key={index} title={feature} variant="arrow" color={color || 'primary'} />
              ))}
            </ul>

            {/* Spacer to push button to bottom */}
            <div className="flex-1" />
          </CardContent>

          <ButtonCardFooter
            variant="elevated"
            badgeSlot={badgeSlot}
            buttonText={ctaText}
            href={href}
            buttonColor={color}
            ariaDescribedBy={descriptionId}
            className="pt-0"
          />
        </Card>
      </div>
    </>
  )
}
