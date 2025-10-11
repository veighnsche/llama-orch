<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import EnterpriseComparison from './EnterpriseComparison.vue'
</script>

<template>
  <Story title="organisms/EnterpriseComparison">
    <Variant title="Default">
      <EnterpriseComparison />
    </Variant>

    <Variant title="8 Features">
      <EnterpriseComparison
        :comparison-data="[
          { feature: 'Data Sovereignty', rbee: true, openai: false, azure: 'Partial' },
          { feature: 'EU-Only Deployment', rbee: true, openai: false, azure: 'Partial' },
          { feature: 'GDPR Compliant', rbee: true, openai: 'Partial', azure: 'Partial' },
          { feature: 'Immutable Audit Logs', rbee: true, openai: false, azure: false },
          { feature: '7-Year Audit Retention', rbee: true, openai: false, azure: false },
          { feature: 'SOC2 Type II Ready', rbee: true, openai: true, azure: true },
          { feature: 'On-Premises Deployment', rbee: true, openai: false, azure: false },
          { feature: 'Complete Control', rbee: true, openai: false, azure: 'Partial' }
        ]"
      />
    </Variant>

    <Variant title="Custom Title">
      <EnterpriseComparison
        title="Feature Comparison"
        subtitle="See how rbee compares to cloud providers."
      />
    </Variant>
  </Story>
</template>
