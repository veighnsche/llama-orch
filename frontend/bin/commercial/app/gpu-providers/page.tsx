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
  EmailCapture,
  Footer,
} from '@rbee/ui/organisms'

export default function GPUProvidersPage() {
  return (
    <main className="min-h-screen bg-background">
      <ProvidersHero />
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
