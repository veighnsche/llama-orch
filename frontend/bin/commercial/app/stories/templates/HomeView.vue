<!-- Created by: TEAM-FE-000 -->
<!-- TEAM-FE-009: Implemented Home page assembly -->
<!-- TEAM-DX-006: Changed from barrel imports to local relative imports to fix Vite build issues -->
<script setup lang="ts">
import HeroSection from '../organisms/HeroSection/HeroSection.vue'
import WhatIsRbee from '../organisms/WhatIsRbee/WhatIsRbee.vue'
import AudienceSelector from '../organisms/AudienceSelector/AudienceSelector.vue'
import ProblemSection from '../organisms/ProblemSection/ProblemSection.vue'
import SolutionSection from '../organisms/SolutionSection/SolutionSection.vue'
import HowItWorksSection from '../organisms/HowItWorksSection/HowItWorksSection.vue'
import FeaturesSection from '../organisms/FeaturesSection/FeaturesSection.vue'
import UseCasesSection from '../organisms/UseCasesSection/UseCasesSection.vue'
import ComparisonSection from '../organisms/ComparisonSection/ComparisonSection.vue'
import PricingSection from '../organisms/PricingSection/PricingSection.vue'
import SocialProofSection from '../organisms/SocialProofSection/SocialProofSection.vue'
import TechnicalSection from '../organisms/TechnicalSection/TechnicalSection.vue'
import FAQSection from '../organisms/FAQSection/FAQSection.vue'
import CTASection from '../organisms/CTASection/CTASection.vue'
import EmailCapture from '../organisms/EmailCapture/EmailCapture.vue'
import Footer from '../organisms/Footer/Footer.vue'
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
