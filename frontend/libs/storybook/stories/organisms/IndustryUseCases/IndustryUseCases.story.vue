<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import IndustryUseCases from './IndustryUseCases.vue'
</script>

<template>
  <Story title="organisms/IndustryUseCases">
    <Variant title="Default">
      <IndustryUseCases />
    </Variant>

    <Variant title="3 Industries">
      <IndustryUseCases
        :industries="[
          {
            title: 'Financial Services',
            description: 'GDPR compliance, audit trails, data residency controls.'
          },
          {
            title: 'Healthcare',
            description: 'HIPAA-compliant infrastructure. Patient data never leaves your network.'
          },
          {
            title: 'Legal',
            description: 'Attorney-client privilege maintained. Complete confidentiality.'
          }
        ]"
      />
    </Variant>

    <Variant title="Custom Title">
      <IndustryUseCases
        title="Regulated Industries"
        subtitle="Built for compliance-first organizations."
      />
    </Variant>
  </Story>
</template>
