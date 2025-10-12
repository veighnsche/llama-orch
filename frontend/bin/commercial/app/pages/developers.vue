<!-- Created by: TEAM-FE-009 -->
<!-- TEAM-FE-009: Implemented Developers page assembly -->
<!-- TEAM-FE-010: Adapted for Nuxt -->
<script setup lang="ts">
import {
  DevelopersHero,
  DevelopersProblem,
  DevelopersSolution,
  DevelopersHowItWorks,
  DevelopersFeatures,
  DevelopersUseCases,
  DevelopersCodeExamples,
  DevelopersPricing,
  DevelopersTestimonials,
  DevelopersCTA,
  EmailCapture,
  Footer,
} from '~/stories'
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
