<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import { Laptop, Users, Home } from 'lucide-vue-next'
import UseCasesGrid from './UseCasesGrid.vue'
</script>

<template>
  <Story title="organisms/UseCasesGrid">
    <Variant title="Default">
      <UseCasesGrid />
    </Variant>

    <Variant title="3 Use Cases">
      <UseCasesGrid
        :use-cases="[
          {
            icon: Laptop,
            iconColor: 'text-blue-600 bg-blue-100',
            title: 'The Solo Developer',
            scenario: 'Building a SaaS with AI features.',
            solution: 'Runs rbee on gaming PC.',
            benefit: '$0/month AI costs.'
          },
          {
            icon: Users,
            iconColor: 'text-amber-600 bg-amber-100',
            title: 'The Small Team',
            scenario: '5-person startup cutting costs.',
            solution: 'Pools team hardware.',
            benefit: 'Saves $6,000/year.'
          },
          {
            icon: Home,
            iconColor: 'text-accent bg-accent/10',
            title: 'The Homelab Enthusiast',
            scenario: 'Has idle GPUs.',
            solution: 'Runs rbee across homelab.',
            benefit: 'Productive AI infrastructure.'
          }
        ]"
      />
    </Variant>

    <Variant title="Custom Title">
      <UseCasesGrid
        title="Who Uses rbee?"
      />
    </Variant>
  </Story>
</template>
