<!-- Created by: TEAM-FE-000 -->
<!-- TEAM-FE-009: Implemented Home page assembly -->
<!-- TEAM-FE-010: Adapted for Nuxt -->
<script setup lang="ts">
import {
  HeroSection,
  WhatIsRbee,
  AudienceSelector,
  ProblemSection,
  SolutionSection,
  HowItWorksSection,
  FeaturesSection,
  UseCasesSection,
  ComparisonSection,
  PricingSection,
  SocialProofSection,
  TechnicalSection,
  FAQSection,
  CTASection,
  EmailCapture,
  Footer,
} from '~/stories'
</script>

<template>
  <main class="min-h-screen pt-16">
    <HeroSection />
    <WhatIsRbee />
    <AudienceSelector />
    <EmailCapture />
    <ProblemSection />
    <SolutionSection />
    <HowItWorksSection />
    <FeaturesSection />
    <UseCasesSection />
    <ComparisonSection />
    <PricingSection />
    <SocialProofSection />
    <TechnicalSection />
    <FAQSection />
    <CTASection />
    <Footer />
  </main>
</template>
