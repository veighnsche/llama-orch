<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import EnterpriseHero from './EnterpriseHero.vue'
</script>

<template>
  <Story title="organisms/EnterpriseHero">
    <Variant title="Default">
      <EnterpriseHero />
    </Variant>

    <Variant title="Without Audit Log">
      <EnterpriseHero :show-audit-log="false" />
    </Variant>

    <Variant title="Custom Stats">
      <EnterpriseHero
        stat1-value="99.99%"
        stat1-label="Uptime SLA"
        stat2-value="10 Years"
        stat2-label="Data Retention"
        stat3-value="50+"
        stat3-label="Event Types"
      />
    </Variant>

    <Variant title="Custom CTAs">
      <EnterpriseHero
        primary-cta="Get Started"
        secondary-cta="Learn More"
      />
    </Variant>
  </Story>
</template>
