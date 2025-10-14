import { ProvidersHero } from '@/components/organisms/Providers/providers-hero'
import { ProvidersProblem } from '@/components/organisms/Providers/providers-problem'
import { ProvidersSolution } from '@/components/organisms/Providers/providers-solution'
import { ProvidersHowItWorks } from '@/components/organisms/Providers/providers-how-it-works'
import { ProvidersFeatures } from '@/components/organisms/Providers/providers-features'
import { ProvidersUseCases } from '@/components/organisms/Providers/providers-use-cases'
import { ProvidersEarnings } from '@/components/organisms/Providers/providers-earnings'
import { ProvidersMarketplace } from '@/components/organisms/Providers/providers-marketplace'
import { ProvidersSecurity } from '@/components/organisms/Providers/providers-security'
import { ProvidersTestimonials } from '@/components/organisms/Providers/providers-testimonials'
import { ProvidersCTA } from '@/components/organisms/Providers/providers-cta'
import { EmailCapture } from '@/components/organisms/EmailCapture/EmailCapture'
import { Footer } from '@/components/organisms/Footer/Footer'

export default function GPUProvidersPage() {
  return (
    <main className="min-h-screen bg-background">
      <ProvidersHero />
      <EmailCapture />
      <ProvidersProblem />
      <ProvidersSolution />
      <ProvidersHowItWorks />
      <ProvidersFeatures />
      <ProvidersUseCases />
      <ProvidersEarnings />
      <ProvidersMarketplace />
      <ProvidersSecurity />
      <ProvidersTestimonials />
      <ProvidersCTA />
    </main>
  )
}
