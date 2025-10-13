import { HeroSection } from '@/components/organisms/HeroSection/HeroSection'
import { WhatIsRbee } from '@/components/organisms/WhatIsRbee/WhatIsRbee'
import { AudienceSelector } from '@/components/organisms/AudienceSelector/AudienceSelector'
import { ProblemSection } from '@/components/organisms/ProblemSection/ProblemSection'
import { SolutionSection } from '@/components/organisms/SolutionSection/SolutionSection'
import { HowItWorksSection } from '@/components/organisms/HowItWorksSection/HowItWorksSection'
import { FeaturesSection } from '@/components/organisms/FeaturesSection/FeaturesSection'
import { UseCasesSection } from '@/components/organisms/UseCasesSection/UseCasesSection'
import { ComparisonSection } from '@/components/organisms/ComparisonSection/ComparisonSection'
import { PricingSection } from '@/components/organisms/PricingSection/PricingSection'
import { SocialProofSection } from '@/components/organisms/SocialProofSection/SocialProofSection'
import { TechnicalSection } from '@/components/organisms/TechnicalSection/TechnicalSection'
import { FAQSection } from '@/components/organisms/FaqSection/FaqSection'
import { CTASection } from '@/components/organisms/CtaSection/CtaSection'
import { EmailCapture } from '@/components/organisms/EmailCapture/EmailCapture'
import { Footer } from '@/components/organisms/Footer/Footer'
import { Anchor, DollarSign, Laptop, Shield, Building, Home as HomeIcon, Users, ArrowRight, BookOpen } from 'lucide-react'

export default function Home() {
  return (
    <main className="min-h-screen pt-16">
      <HeroSection />
      <WhatIsRbee />
      <AudienceSelector />
      <EmailCapture />
      <ProblemSection />
      <SolutionSection
        title="Your hardware. Your models. Your control."
        subtitle="rbee orchestrates inference across every GPU in your home network—workstations, gaming rigs, and Macs—turning idle hardware into a private, OpenAI-compatible AI platform."
        benefits={[
          {
            icon: <DollarSign className="h-6 w-6 text-primary" aria-hidden="true" />,
            title: 'Zero ongoing costs',
            body: 'Pay only for electricity. No API bills, no per-token surprises.',
          },
          {
            icon: <Shield className="h-6 w-6 text-primary" aria-hidden="true" />,
            title: 'Complete privacy',
            body: 'Code and data never leave your network. Audit-ready by design.',
          },
          {
            icon: <Anchor className="h-6 w-6 text-primary" aria-hidden="true" />,
            title: 'Locked to your rules',
            body: 'Models update only when you approve. No breaking changes.',
          },
          {
            icon: <Laptop className="h-6 w-6 text-primary" aria-hidden="true" />,
            title: 'Use all your hardware',
            body: 'CUDA, Metal, and CPU orchestrated as one pool.',
          },
        ]}
        topology={{
          mode: 'multi-host',
          hosts: [
            {
              hostLabel: 'Gaming PC',
              workers: [
                { id: 'w0', label: 'GPU 0', kind: 'cuda' },
                { id: 'w1', label: 'GPU 1', kind: 'cuda' },
              ],
            },
            {
              hostLabel: 'MacBook Pro',
              workers: [{ id: 'w2', label: 'GPU 0', kind: 'metal' }],
            },
            {
              hostLabel: 'Workstation',
              workers: [
                { id: 'w3', label: 'GPU 0', kind: 'cuda' },
                { id: 'w4', label: 'CPU 0', kind: 'cpu' },
              ],
            },
          ],
        }}
      />
      <HowItWorksSection />
      <FeaturesSection />
      <UseCasesSection
        title="Built for those who value independence"
        subtitle="Run serious AI on your own hardware. Keep costs at zero, keep control at 100%."
        items={[
          {
            icon: Laptop,
            title: 'The solo developer',
            scenario: 'Shipping a SaaS with AI features; wants control without vendor lock-in.',
            solution:
              'Run rbee on your gaming PC + spare workstation. Llama 70B for coding, SD for assets—local & fast.',
            outcome: '$0/month AI costs. Full control. No rate limits.',
          },
          {
            icon: Users,
            title: 'The small team',
            scenario: '5-person startup burning $500/mo on APIs.',
            solution:
              'Pool 3 workstations + 2 Macs into one rbee cluster. Shared models, faster inference, fewer blockers.',
            outcome: '$6,000+ saved per year. GDPR-friendly by design.',
          },
          {
            icon: HomeIcon,
            title: 'The homelab enthusiast',
            scenario: 'Four GPUs gathering dust.',
            solution: 'Spread workers across your LAN in minutes. Build agents: coder, doc generator, code reviewer.',
            outcome: 'Idle GPUs → productive. Auto-download models, clean shutdowns.',
          },
          {
            icon: Building,
            title: 'The enterprise',
            scenario: '50-dev org. Code cannot leave the premises.',
            solution:
              'On-prem rbee with audit trails and policy routing. Rhai-based rules for data residency & access.',
            outcome: 'EU-only compliance. Zero external dependencies.',
          },
        ]}
      />
      <ComparisonSection />
      <PricingSection />
      <SocialProofSection />
      <TechnicalSection />
      <FAQSection />
      <CTASection
        title="Stop depending on AI providers. Start building today."
        subtitle="Join 500+ developers who've taken control of their AI infrastructure."
        primary={{ label: 'Get started free', href: '/getting-started', iconRight: ArrowRight }}
        secondary={{ label: 'View documentation', href: '/docs', iconLeft: BookOpen, variant: 'outline' }}
        note="100% open source. No credit card required. Install in 15 minutes."
        emphasis="gradient"
      />
    </main>
  )
}
