import { Button } from '@rbee/ui/atoms/Button'
import { GitHubIcon } from '@rbee/ui/icons'
import Link from 'next/link'

export interface TechItem {
  name: string
  description: string
  ariaLabel: string
  delay?: number
}

export interface TechnologyStackProps {
  /**
   * Array of technology items to display
   */
  technologies: TechItem[]
  /**
   * Whether to show the open source CTA card
   * @default true
   */
  showOpenSourceCTA?: boolean
  /**
   * GitHub repository URL
   * @default "https://github.com/yourusername/rbee"
   */
  githubUrl?: string
  /**
   * License type to display
   * @default "MIT License"
   */
  license?: string
  /**
   * Whether to show the architecture docs link
   * @default true
   */
  showArchitectureLink?: boolean
  /**
   * Architecture docs URL
   * @default "/docs/architecture"
   */
  architectureUrl?: string
  /**
   * Optional className for the container
   */
  className?: string
}

const delayClasses = [
  'motion-safe:delay-0',
  'motion-safe:delay-100',
  'motion-safe:delay-200',
  'motion-safe:delay-300',
  'motion-safe:delay-400',
  'motion-safe:delay-500',
]

export function TechnologyStack({
  technologies,
  showOpenSourceCTA = true,
  githubUrl = 'https://github.com/yourusername/rbee',
  license = 'MIT License',
  showArchitectureLink = true,
  architectureUrl = '/docs/architecture',
  className = '',
}: TechnologyStackProps) {
  return (
    <div className={className}>
      <div className="text-xs tracking-wide uppercase text-muted-foreground mb-3 font-sans">Stack</div>
      <h3 className="text-2xl font-bold text-foreground mb-4">Technology Stack</h3>
      <div className="space-y-3">
        {/* Technology Cards */}
        {technologies.map((tech, index) => (
          <article
            key={tech.name}
            role="group"
            aria-label={tech.ariaLabel}
            className={`bg-muted/60 border border-border rounded-md p-4 hover:border-primary/40 transition-colors motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400 ${
              delayClasses[index % delayClasses.length]
            }`}
          >
            <div className="font-semibold text-foreground">{tech.name}</div>
            <div className="text-sm text-muted-foreground">{tech.description}</div>
          </article>
        ))}

        {/* Open Source CTA Card */}
        {showOpenSourceCTA && (
          <article
            role="group"
            aria-label="Open Source Information"
            className="bg-primary/10 border border-primary/30 rounded-md p-5 flex items-center justify-between motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400 motion-safe:delay-500"
          >
            <div>
              <div className="font-bold text-foreground">100% Open Source</div>
              <div className="text-sm text-muted-foreground">{license}</div>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="border-primary/30 bg-transparent"
              aria-label="View rbee source on GitHub"
              asChild
            >
              <a href={githubUrl} target="_blank" rel="noopener noreferrer">
                <GitHubIcon size={16} />
                View Source
              </a>
            </Button>
          </article>
        )}

        {/* Architecture Docs Link */}
        {showArchitectureLink && (
          <Link
            href={architectureUrl}
            className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Read Architecture →
          </Link>
        )}
      </div>
    </div>
  )
}
