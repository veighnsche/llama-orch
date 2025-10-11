<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import { Globe, FileX } from 'lucide-vue-next'
import EnterpriseProblem from './EnterpriseProblem.vue'
</script>

<template>
  <Story title="organisms/EnterpriseProblem">
    <Variant title="Default">
      <EnterpriseProblem />
    </Variant>

    <Variant title="Custom Problems (2 items)">
      <EnterpriseProblem
        :problems="[
          {
            icon: Globe,
            title: 'Data Residency Issues',
            description: 'Your data crosses international borders without your control.'
          },
          {
            icon: FileX,
            title: 'No Audit Trail',
            description: 'Cannot prove compliance to auditors.'
          }
        ]"
      />
    </Variant>

    <Variant title="Custom Quote">
      <EnterpriseProblem
        quote-text="We need complete control over our AI infrastructure."
        quote-author="— CTO, Fortune 500 Company"
      />
    </Variant>
  </Story>
</template>
Can y