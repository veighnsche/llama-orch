<!-- Created by: TEAM-FE-009 -->
<!-- TEAM-FE-009: Implemented Enterprise page assembly -->
<script setup lang="ts">
import {
  EnterpriseHero,
  EnterpriseProblem,
  EnterpriseSolution,
  EnterpriseCompliance,
  EnterpriseSecurity,
  EnterpriseHowItWorks,
  EnterpriseUseCases,
  EnterpriseComparison,
  EnterpriseFeatures,
  EnterpriseTestimonials,
  EnterpriseCTA,
  EmailCapture,
  Footer,
} from '~/stories'
</script>

<template>
  <main class="min-h-screen pt-16">
    <EnterpriseHero />
    <EmailCapture />
    <EnterpriseProblem />
    <EnterpriseSolution />
    <EnterpriseCompliance />
    <EnterpriseSecurity />
    <EnterpriseHowItWorks />
    <EnterpriseUseCases />
    <EnterpriseComparison />
    <EnterpriseFeatures />
    <EnterpriseTestimonials />
    <EnterpriseCTA />
    <Footer />
  </main>
</template>
