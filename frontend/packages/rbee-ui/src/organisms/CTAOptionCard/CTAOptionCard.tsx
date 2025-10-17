import { Badge, Card, CardContent, CardFooter } from '@rbee/ui/atoms'
import { IconCardHeader } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface CTAOptionCardProps {
  icon: ReactNode
  title: string
  body: string
  action: ReactNode
  tone?: 'primary' | 'outline'
  note?: string
  eyebrow?: string
  className?: string
}

export function CTAOptionCard({
  icon,
  title,
  body,
  action,
  tone = 'outline',
  note,
  eyebrow,
  className,
}: CTAOptionCardProps) {
  const titleId = `cta-option-${title.toLowerCase().replace(/\s+/g, '-')}`

  return (
    <Card
      className={cn(
        'group relative p-6 sm:p-7',
        // Entrance animation
        'animate-in fade-in-50 zoom-in-95 duration-300',
        // Interactive depth
        'hover:border-primary/40 hover:shadow-md focus-within:shadow-md transition-shadow',
        // Primary tone overrides
        tone === 'primary' && 'border-primary/40 bg-primary/5',
        className,
      )}
    >
      {/* Subtle radial highlight for primary tone */}
      {tone === 'primary' && (
        <span
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-8 -top-6 h-20 rounded-full bg-primary/10 blur-2xl"
        />
      )}

      <IconCardHeader
        icon={icon}
        title={title}
        titleId={titleId}
        iconSize="lg"
        iconTone="primary"
        titleClassName={cn(
          'text-2xl',
          tone === 'primary' ? 'text-primary' : 'text-foreground',
        )}
        align="center"
        className="flex-col items-center"
      />

      {/* Eyebrow label */}
      {eyebrow && (
        <div className="-mt-2 mb-4 flex justify-center">
          <Badge
            variant="outline"
            className="bg-primary/10 border-primary/30 text-primary text-[11px] font-medium"
          >
            {eyebrow}
          </Badge>
        </div>
      )}

      <CardContent className="p-0 text-center">
        <p className="font-sans text-sm leading-6 text-muted-foreground max-w-[80ch] mx-auto">
          {body}
        </p>
      </CardContent>

      <CardFooter className="flex-col p-0 pt-5">
        {action}
        {note && (
          <p className="mt-2 text-center font-sans text-[11px] text-muted-foreground">
            {note}
          </p>
        )}
      </CardFooter>
    </Card>
  )
}
