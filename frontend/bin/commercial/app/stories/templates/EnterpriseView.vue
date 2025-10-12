<!-- Created by: TEAM-FE-000 -->
<!-- TEAM-FE-007: Implemented Enterprise page assembly -->
<!-- TEAM-DX-006: Changed from barrel imports to local relative imports to fix Vite build issues -->
<script setup lang="ts">
import EnterpriseHero from '../organisms/EnterpriseHero/EnterpriseHero.vue'
import EnterpriseProblem from '../organisms/EnterpriseProblem/EnterpriseProblem.vue'
import EnterpriseSolution from '../organisms/EnterpriseSolution/EnterpriseSolution.vue'
import EnterpriseCompliance from '../organisms/EnterpriseCompliance/EnterpriseCompliance.vue'
import EnterpriseSecurity from '../organisms/EnterpriseSecurity/EnterpriseSecurity.vue'
import EnterpriseHowItWorks from '../organisms/EnterpriseHowItWorks/EnterpriseHowItWorks.vue'
import EnterpriseUseCases from '../organisms/EnterpriseUseCases/EnterpriseUseCases.vue'
import EnterpriseComparison from '../organisms/EnterpriseComparison/EnterpriseComparison.vue'
import EnterpriseFeatures from '../organisms/EnterpriseFeatures/EnterpriseFeatures.vue'
import EnterpriseTestimonials from '../organisms/EnterpriseTestimonials/EnterpriseTestimonials.vue'
import EnterpriseCTA from '../organisms/EnterpriseCTA/EnterpriseCTA.vue'
import EmailCapture from '../organisms/EmailCapture/EmailCapture.vue'
import Footer from '../organisms/Footer/Footer.vue'
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
