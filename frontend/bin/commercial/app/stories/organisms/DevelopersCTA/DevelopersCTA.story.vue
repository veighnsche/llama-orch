<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-005: Added story variants -->
<script setup lang="ts">
import DevelopersCTA from './DevelopersCTA.vue'
</script>

<template>
  <Story title="organisms/DevelopersCTA">
    <Variant title="Default">
      <DevelopersCTA />
    </Variant>

    <Variant title="Custom Content">
      <DevelopersCTA
        title="Ready to Get Started?"
        subtitle="Join the community of developers building with rbee"
        primary-cta="Download Now"
        secondary-cta="Read the Docs"
        footer-text="Free forever. Open source. No strings attached."
      />
    </Variant>
  </Story>
</template>
