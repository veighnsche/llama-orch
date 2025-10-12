<!-- Created by: TEAM-FE-000 -->
<!-- TEAM-FE-005: Implemented Developers page assembly -->
<!-- TEAM-DX-006: Changed from barrel imports to local relative imports to fix Vite build issues -->
<script setup lang="ts">
import DevelopersHero from '../organisms/DevelopersHero/DevelopersHero.vue'
import DevelopersProblem from '../organisms/DevelopersProblem/DevelopersProblem.vue'
import DevelopersSolution from '../organisms/DevelopersSolution/DevelopersSolution.vue'
import DevelopersHowItWorks from '../organisms/DevelopersHowItWorks/DevelopersHowItWorks.vue'
import DevelopersFeatures from '../organisms/DevelopersFeatures/DevelopersFeatures.vue'
import DevelopersUseCases from '../organisms/DevelopersUseCases/DevelopersUseCases.vue'
import DevelopersCodeExamples from '../organisms/DevelopersCodeExamples/DevelopersCodeExamples.vue'
import DevelopersPricing from '../organisms/DevelopersPricing/DevelopersPricing.vue'
import DevelopersTestimonials from '../organisms/DevelopersTestimonials/DevelopersTestimonials.vue'
import DevelopersCTA from '../organisms/DevelopersCTA/DevelopersCTA.vue'
import EmailCapture from '../organisms/EmailCapture/EmailCapture.vue'
import Footer from '../organisms/Footer/Footer.vue'
</script>

<template>
  <main class="min-h-screen pt-16">
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
</template>
