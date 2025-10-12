<!-- Created by: TEAM-FE-009 -->
<!-- TEAM-FE-009: Implemented GPU Providers page assembly -->
<script setup lang="ts">
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
} from '~/stories'
</script>

<template>
  <main class="min-h-screen pt-16">
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
    <Footer />
  </main>
</template>
