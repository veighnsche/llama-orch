import { Button, Card, CardContent } from '@rbee/ui/atoms'
import Link from 'next/link'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type AuditReadinessCTAProps = {
  heading: string
  description: string
  note: string
  noteAriaLabel: string
  buttons: Array<{
    text: string
    href: string
    variant?: 'default' | 'outline'
    ariaDescribedby?: string
  }>
  footnote: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function AuditReadinessCTA({
  heading,
  description,
  note,
  noteAriaLabel,
  buttons,
  footnote,
}: AuditReadinessCTAProps) {
  return (
    <Card className="animate-in fade-in-50 rounded-2xl border-primary/20 bg-primary/5 [animation-delay:200ms]">
      <CardContent className="p-8 text-center">
        <h3 className="mb-2 text-2xl font-semibold text-foreground">{heading}</h3>
        <p className="mb-2 text-foreground/85">{description}</p>
        <p
          id="compliance-pack-note"
          className="mb-6 text-sm text-muted-foreground"
          aria-label={noteAriaLabel}
        >
          {note}
        </p>
        <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
          {buttons.map((button, idx) => (
            <Button
              key={idx}
              size="lg"
              variant={button.variant}
              asChild
              aria-describedby={button.ariaDescribedby}
              className="transition-transform active:scale-[0.98]"
            >
              <Link href={button.href}>{button.text}</Link>
            </Button>
          ))}
        </div>
        <p className="mt-6 text-xs text-muted-foreground">{footnote}</p>
      </CardContent>
    </Card>
  )
}
