'use client'

import {
  ProvidersCTA,
  ProvidersEarnings,
  ProvidersFeatures,
  ProvidersHero,
  ProvidersHowItWorks,
  ProvidersMarketplace,
  ProvidersProblem,
  ProvidersSecurity,
  ProvidersSolution,
  ProvidersTestimonials,
  ProvidersUseCases,
} from "@rbee/ui/organisms";

// ============================================================================
// Props Objects
// ============================================================================
// All organisms are self-contained with no props needed
// ============================================================================

export default function ProvidersPage() {
  return (
    <main>
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
  );
}
