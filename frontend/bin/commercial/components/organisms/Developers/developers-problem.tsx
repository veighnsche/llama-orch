import { AlertTriangle, DollarSign, Lock } from 'lucide-react'
import { ProblemSection } from '@/components/organisms/ProblemSection/ProblemSection'

/**
 * Backward-compatible wrapper for the developers page.
 * Re-exports the shared ProblemSection with developer-specific defaults.
 */
export function DevelopersProblem() {
  return (
    <ProblemSection
      id="risk"
      title="The Hidden Risk of AI-Assisted Development"
      subtitle="You're building complex codebases with AI assistance. But what happens when your provider changes the rules?"
      items={[
        {
          title: 'The Model Changes',
          body: 'Your AI assistant updates overnight. Suddenly, code generation breaks. Your workflow is destroyed. Your team is blocked.',
          icon: AlertTriangle,
          tone: 'destructive',
        },
        {
          title: 'The Price Increases',
          body: '$20/month becomes $200/month. Multiply by your team size. Your AI infrastructure costs spiral out of control.',
          icon: DollarSign,
          tone: 'primary',
        },
        {
          title: 'The Provider Shuts Down',
          body: 'API deprecated. Service discontinued. Your complex codebase—built with AI assistance—becomes unmaintainable overnight.',
          icon: Lock,
          tone: 'destructive',
        },
      ]}
      kicker={{
        text: 'Heavy, complicated codebases built with AI assistance are a ticking time bomb if you depend on external providers.',
        tone: 'destructive',
      }}
    />
  )
}
