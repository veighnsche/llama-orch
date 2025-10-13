import { AlertTriangle, DollarSign, Lock } from 'lucide-react'
import { ReactNode } from 'react'

/**
 * ProblemItem represents a single problem card in the section.
 */
export type ProblemItem = {
  title: string
  body: string
  icon: React.ComponentType<{ className?: string }>
  tone?: 'primary' | 'destructive' | 'muted'
}

/**
 * ProblemSection displays a grid of problem cards with optional eyebrow, kicker, and illustration.
 * 
 * @example
 * ```tsx
 * <ProblemSection
 *   id="risk"
 *   eyebrow="Why this matters"
 *   title="The hidden risk of AI-assisted development"
 *   subtitle="You're building complex codebases with AI assistance. What happens when the provider changes the rules?"
 *   items={[
 *     { title: 'The model changes', body: '…', icon: AlertTriangle, tone: 'destructive' },
 *     { title: 'The price increases', body: '…', icon: DollarSign, tone: 'primary' },
 *     { title: 'The provider shuts down', body: '…', icon: Lock, tone: 'destructive' },
 *   ]}
 *   kicker={{ text: 'Heavy AI-built codebases are a ticking time bomb…', tone: 'destructive' }}
 * />
 * ```
 */
export type ProblemSectionProps = {
  eyebrow?: string
  title?: string
  subtitle?: string
  items?: ProblemItem[]
  kicker?: { text: string; tone?: 'primary' | 'destructive' | 'muted' }
  id?: string
  className?: string
  illustration?: ReactNode
}

const toneMap = {
  primary: {
    border: 'border-primary/50 hover:border-primary',
    bg: 'bg-primary/5',
    iconBg: 'bg-primary/10',
    iconText: 'text-primary',
  },
  destructive: {
    border: 'border-destructive/50 hover:border-destructive',
    bg: 'bg-destructive/10',
    iconBg: 'bg-destructive/10',
    iconText: 'text-destructive',
  },
  muted: {
    border: 'border-border hover:border-border/80',
    bg: 'bg-card',
    iconBg: 'bg-muted',
    iconText: 'text-muted-foreground',
  },
}

export function ProblemSection({
  eyebrow = 'Why this matters',
  title = 'The hidden risk of AI-assisted development',
  subtitle = "You're building complex codebases with AI assistance. What happens when the provider changes the rules?",
  items = [
    {
      title: 'The model changes',
      body: 'Your assistant updates overnight. Code generation breaks; workflows stall; your team is blocked.',
      icon: AlertTriangle,
      tone: 'destructive' as const,
    },
    {
      title: 'The price increases',
      body: '$20/month becomes $200/month—multiplied by your team. Infrastructure costs spiral.',
      icon: DollarSign,
      tone: 'primary' as const,
    },
    {
      title: 'The provider shuts down',
      body: 'APIs get deprecated. Your AI-built code becomes unmaintainable overnight.',
      icon: Lock,
      tone: 'destructive' as const,
    },
  ],
  kicker = {
    text: 'Heavy AI-built codebases are a ticking time bomb if you depend on external providers.',
    tone: 'destructive' as const,
  },
  id,
  className = '',
  illustration,
}: ProblemSectionProps) {
  const kickerTone = kicker?.tone || 'destructive'
  const kickerColorClass =
    kickerTone === 'primary'
      ? 'text-primary'
      : kickerTone === 'destructive'
        ? 'text-destructive'
        : 'text-muted-foreground'

  return (
    <section
      id={id}
      className={`border-b border-border bg-background py-24 animate-in fade-in-50 duration-500 ${className}`}
    >
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        {/* Heading block */}
        <div className="mx-auto max-w-2xl text-center">
          {eyebrow && (
            <div className="mb-4 inline-flex items-center rounded-full border border-border px-3 py-1 text-sm text-muted-foreground">
              {eyebrow}
            </div>
          )}
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            {title}
          </h2>
          {subtitle && (
            <p className="text-balance text-lg leading-relaxed text-muted-foreground">
              {subtitle}
            </p>
          )}
        </div>

        {/* Optional illustration */}
        {illustration && (
          <div className="mx-auto mt-12 max-w-4xl rounded-xl border border-border bg-card p-6">
            {illustration}
          </div>
        )}

        {/* Cards grid */}
        <div className="mx-auto mt-16 grid max-w-5xl gap-8 sm:grid-cols-3">
          {items.map((item, index) => {
            const tone = item.tone || 'muted'
            const styles = toneMap[tone]
            const Icon = item.icon

            return (
              <div
                key={index}
                className={`group relative overflow-hidden rounded-xl border p-8 transition-all focus:outline-none focus:ring-2 focus:ring-primary/40 animate-in fade-in slide-in-from-bottom-2 duration-500 ${styles.border} ${styles.bg}`}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div
                  className={`mb-4 inline-flex h-10 w-10 items-center justify-center rounded-lg ${styles.iconBg}`}
                >
                  <Icon className={`h-5 w-5 ${styles.iconText}`} aria-hidden="true" />
                </div>
                <h3 className="mb-3 text-xl font-semibold text-card-foreground">
                  {item.title}
                </h3>
                <p className="text-balance leading-relaxed text-muted-foreground">
                  {item.body}
                </p>
              </div>
            )
          })}
        </div>

        {/* Kicker */}
        {kicker && (
          <div className="mx-auto mt-12 max-w-2xl text-center">
            <p
              className={`text-balance text-lg font-medium leading-relaxed ${kickerColorClass}`}
            >
              {kicker.text}
            </p>
          </div>
        )}
      </div>
    </section>
  )
}
