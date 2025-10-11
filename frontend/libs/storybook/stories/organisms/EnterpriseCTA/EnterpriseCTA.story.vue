<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import { Calendar, FileText } from 'lucide-vue-next'
import EnterpriseCTA from './EnterpriseCTA.vue'
</script>

<template>
  <Story title="organisms/EnterpriseCTA">
    <Variant title="Default">
      <EnterpriseCTA />
    </Variant>

    <Variant title="2 Options">
      <EnterpriseCTA
        :options="[
          {
            icon: Calendar,
            title: 'Schedule Demo',
            description: '30-minute demo with our compliance team.',
            buttonText: 'Book Demo',
            variant: 'default'
          },
          {
            icon: FileText,
            title: 'Compliance Pack',
            description: 'Download GDPR, SOC2, and ISO 27001 documentation.',
            buttonText: 'Download Docs',
            variant: 'outline'
          }
        ]"
      />
    </Variant>

    <Variant title="Custom Title">
      <EnterpriseCTA
        title="Get Started Today"
        subtitle="Choose your path to compliance."
      />
    </Variant>
  </Story>
</template>
