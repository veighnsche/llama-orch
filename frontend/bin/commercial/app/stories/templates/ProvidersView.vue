<!-- Created by: TEAM-FE-000 -->
<!-- TEAM-FE-006: Implemented Providers page assembly -->
<!-- TEAM-DX-006: Changed from barrel imports to local relative imports to fix Vite build issues -->
<script setup lang="ts">
import ProvidersHero from '../organisms/ProvidersHero/ProvidersHero.vue'
import ProvidersProblem from '../organisms/ProvidersProblem/ProvidersProblem.vue'
import ProvidersSolution from '../organisms/ProvidersSolution/ProvidersSolution.vue'
import ProvidersHowItWorks from '../organisms/ProvidersHowItWorks/ProvidersHowItWorks.vue'
import ProvidersFeatures from '../organisms/ProvidersFeatures/ProvidersFeatures.vue'
import ProvidersUseCases from '../organisms/ProvidersUseCases/ProvidersUseCases.vue'
import ProvidersEarnings from '../organisms/ProvidersEarnings/ProvidersEarnings.vue'
import ProvidersMarketplace from '../organisms/ProvidersMarketplace/ProvidersMarketplace.vue'
import ProvidersSecurity from '../organisms/ProvidersSecurity/ProvidersSecurity.vue'
import ProvidersTestimonials from '../organisms/ProvidersTestimonials/ProvidersTestimonials.vue'
import ProvidersCTA from '../organisms/ProvidersCTA/ProvidersCTA.vue'
import EmailCapture from '../organisms/EmailCapture/EmailCapture.vue'
import Footer from '../organisms/Footer/Footer.vue'
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
