import { DevelopersHero } from '@/components/organisms/Developers/developers-hero'
import { DevelopersProblem } from '@/components/organisms/Developers/developers-problem'
import { DevelopersSolution } from '@/components/organisms/Developers/developers-solution'
import { DevelopersHowItWorks } from '@/components/organisms/Developers/developers-how-it-works'
import { DevelopersFeatures } from '@/components/organisms/Developers/developers-features'
import { DevelopersUseCases } from '@/components/organisms/Developers/developers-use-cases'
import { DevelopersCodeExamples } from '@/components/organisms/Developers/developers-code-examples'
import { DevelopersPricing } from '@/components/organisms/Developers/developers-pricing'
import { DevelopersTestimonials } from '@/components/organisms/Developers/developers-testimonials'
import { DevelopersCTA } from '@/components/organisms/Developers/developers-cta'
import { EmailCapture } from '@/components/organisms/EmailCapture/EmailCapture'
import { Footer } from '@/components/organisms/Footer/Footer'

export default function DevelopersPage() {
  return (
    <main className="min-h-screen bg-slate-950 pt-16">
      <DevelopersHero />
      <EmailCapture />
      <DevelopersProblem />
      <DevelopersSolution />
      <DevelopersHowItWorks />
      <DevelopersFeatures />
      <DevelopersUseCases />
      <DevelopersCodeExamples />
      <DevelopersPricing />
      <DevelopersTestimonials />
      <DevelopersCTA />
      <Footer />
    </main>
  )
}
