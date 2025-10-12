<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import { Shield, Database, Network, Terminal } from 'lucide-vue-next'
import AdditionalFeaturesGrid from './AdditionalFeaturesGrid.vue'
</script>

<template>
  <Story title="organisms/AdditionalFeaturesGrid">
    <Variant title="Default">
      <AdditionalFeaturesGrid />
    </Variant>

    <Variant title="4 Features">
      <AdditionalFeaturesGrid
        :features="[
          {
            icon: Shield,
            iconColor: 'text-blue-600 bg-blue-100',
            title: 'Cascading Shutdown',
            description: 'Press Ctrl+C once. Everything shuts down cleanly.'
          },
          {
            icon: Database,
            iconColor: 'text-accent bg-accent/10',
            title: 'Model Catalog',
            description: 'Automatic model provisioning from Hugging Face.'
          },
          {
            icon: Network,
            iconColor: 'text-primary bg-primary/10',
            title: 'Network Orchestration',
            description: 'Orchestrate across your entire home network.'
          },
          {
            icon: Terminal,
            iconColor: 'text-muted-foreground bg-muted',
            title: 'CLI & Web UI',
            description: 'Powerful CLI for automation and scripting.'
          }
        ]"
      />
    </Variant>

    <Variant title="Custom Title">
      <AdditionalFeaturesGrid
        title="Core Features"
      />
    </Variant>
  </Story>
</template>
