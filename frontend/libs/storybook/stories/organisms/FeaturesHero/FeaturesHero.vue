<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented FeaturesHero -->
<script setup lang="ts">
interface Props {
  title?: string
  highlight?: string
  description?: string
}

withDefaults(defineProps<Props>(), {
  title: 'Enterprise-Grade Features.',
  highlight: 'Homelab Simplicity.',
  description:
    'rbee combines the power of enterprise AI infrastructure with the simplicity of running on your own hardware. No cloud dependencies. No ongoing costs. Complete control.',
})
</script>

<template>
  <section class="py-24 bg-gradient-to-b from-background to-secondary">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center">
        <h1 class="text-5xl lg:text-6xl font-bold text-foreground mb-6 text-balance">
          {{ title }}
          <br />
          <span class="text-primary">{{ highlight }}</span>
        </h1>
        <p class="text-xl text-muted-foreground leading-relaxed">
          {{ description }}
        </p>
      </div>
    </div>
  </section>
</template>
