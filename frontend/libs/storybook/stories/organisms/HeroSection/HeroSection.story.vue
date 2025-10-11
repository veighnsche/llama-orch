<!-- Created by: TEAM-FE-003 -->
<script setup lang="ts">
import HeroSection from './HeroSection.vue'
</script>

<template>
  <Story title="organisms/HeroSection">
    <Variant title="Default">
      <HeroSection />
    </Variant>
    
    <Variant title="Custom Content">
      <HeroSection
        title="Custom AI Platform"
        subtitle="Your Way"
        description="Build amazing AI applications with complete control over your infrastructure."
        primary-button-text="Start Building"
        secondary-button-text="Learn More"
        github-stars="2,500+"
      />
    </Variant>
  </Story>
</template>
