<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-002: Implemented PricingHero organism -->
<script setup lang="ts">
interface Props {
  title?: string
  subtitle?: string
  highlightText?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Start Free.',
  highlightText: 'Scale When Ready.',
  subtitle: 'All tiers include the full rbee orchestrator. No feature gates. No artificial limits. Just honest pricing for honest infrastructure.',
})
</script>

<template>
  <section class="py-24 bg-gradient-to-b from-slate-950 to-slate-900">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center">
        <h1 class="text-5xl lg:text-6xl font-bold text-white mb-6 text-balance">
          {{ title }}
          <br>
          <span class="text-amber-500">{{ highlightText }}</span>
        </h1>
        <p class="text-xl text-slate-300 leading-relaxed">
          {{ subtitle }}
        </p>
      </div>
    </div>
  </section>
</template>
