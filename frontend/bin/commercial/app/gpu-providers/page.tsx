import {
  ProvidersHero,
  ProvidersProblem,
  ProvidersSolution,
  ProvidersHowItWorks,
  ProvidersFeatures,
  ProvidersUseCases,
  ProvidersEarnings,
  ProvidersMarketplace,
  ProvidersSecurity,
  ProvidersTestimonials,
  ProvidersCTA,
} from '@rbee/ui/organisms/Providers'
import { EmailCapture } from '@rbee/ui/organisms/EmailCapture'
import { Footer } from '@rbee/ui/organisms/Footer'

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
