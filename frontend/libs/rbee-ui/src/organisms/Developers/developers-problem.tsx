import { AlertTriangle, DollarSign, Lock } from 'lucide-react'
import { ProblemSection } from '@rbee/ui/organisms/ProblemSection'

/**
 * Backward-compatible wrapper for the developers page.
 * Re-exports the shared ProblemSection with developer-specific defaults.
 */
export function DevelopersProblem() {
  return (
    <ProblemSection
      id="risk"
      kicker="The Hidden Cost of Dependency"
      title="The Hidden Risk of AI-Assisted Development"
      subtitle="You're building complex codebases with AI assistance. But what happens when your provider changes the rules?"
      items={[
        {
          title: 'The Model Changes',
          body: 'Your AI assistant updates overnight. Suddenly, code generation breaks. Your workflow is destroyed. Your team is blocked.',
          icon: <AlertTriangle className="h-6 w-6" />,
          tone: 'destructive',
          tag: 'High risk',
        },
        {
          title: 'The Price Increases',
          body: '$20/month becomes $200/month. Multiply by your team size. Your AI infrastructure costs spiral out of control.',
          icon: <DollarSign className="h-6 w-6" />,
          tone: 'primary',
          tag: 'Cost increase: 10x',
        },
        {
          title: 'The Provider Shuts Down',
          body: 'API deprecated. Service discontinued. Your complex codebase—built with AI assistance—becomes unmaintainable overnight.',
          icon: <Lock className="h-6 w-6" />,
          tone: 'destructive',
          tag: 'Critical failure',
        },
      ]}
      ctaPrimary={{ label: 'Take Control', href: '/getting-started' }}
      ctaSecondary={{ label: 'View Documentation', href: '/docs' }}
      ctaCopy="Heavy, complicated codebases built with AI assistance are a ticking time bomb if you depend on external providers."
    />
  )
}
