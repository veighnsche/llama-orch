import {
  DevelopersHero,
  DevelopersProblem,
  DevelopersSolution,
  DevelopersHowItWorks,
  DevelopersFeatures,
  DevelopersUseCases,
  DevelopersCodeExamples,
} from '@rbee/ui/organisms/Developers'
import { EmailCapture } from '@rbee/ui/organisms/EmailCapture'
import { Footer } from '@rbee/ui/organisms/Footer'
import { PricingSection } from '@rbee/ui/organisms/PricingSection'
import { TestimonialsSection } from '@rbee/ui/organisms/SocialProofSection'
import { CTASection } from '@rbee/ui/organisms/CtaSection'
import { ArrowRight } from 'lucide-react'
import { GitHubIcon } from '@rbee/ui/atoms/GitHubIcon'

export default function DevelopersPage() {
  return (
    <main className="min-h-screen bg-slate-950">
      <DevelopersHero />
      <EmailCapture />
      <DevelopersProblem />
      <DevelopersSolution />
      <DevelopersHowItWorks />
      <DevelopersFeatures />
      <DevelopersUseCases />
      <DevelopersCodeExamples />
      <PricingSection variant="home" showKicker={false} showEditorialImage={false} />
      <TestimonialsSection
        title="Trusted by Developers Who Value Independence"
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
            role: 'DevOps Engineer',
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
      <CTASection
        title="Stop Depending on AI Providers. Start Building Today."
        subtitle="Join 500+ developers who've taken control of their AI infrastructure."
        primary={{ label: 'Get Started Free', href: '/getting-started', iconRight: ArrowRight }}
        secondary={{ label: 'View Documentation', href: '/docs', iconLeft: GitHubIcon, variant: 'outline' }}
        note="100% open source. No credit card required. Install in 15 minutes."
      />
    </main>
  )
}
