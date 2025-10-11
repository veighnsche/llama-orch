import { DevelopersHero } from "@/components/developers/developers-hero"
import { DevelopersProblem } from "@/components/developers/developers-problem"
import { DevelopersSolution } from "@/components/developers/developers-solution"
import { DevelopersHowItWorks } from "@/components/developers/developers-how-it-works"
import { DevelopersFeatures } from "@/components/developers/developers-features"
import { DevelopersUseCases } from "@/components/developers/developers-use-cases"
import { DevelopersCodeExamples } from "@/components/developers/developers-code-examples"
import { DevelopersPricing } from "@/components/developers/developers-pricing"
import { DevelopersTestimonials } from "@/components/developers/developers-testimonials"
import { DevelopersCTA } from "@/components/developers/developers-cta"
import { EmailCapture } from "@/components/email-capture"
import { Footer } from "@/components/footer"

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
