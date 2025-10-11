<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-009: Implemented UseCasesHero -->
<script setup lang="ts">
interface Props {
  title?: string
  highlight?: string
  description?: string
}

withDefaults(defineProps<Props>(), {
  title: 'Built for Those Who Value',
  highlight: 'Independence',
  description:
    'From solo developers to enterprises, rbee adapts to your needs. Own your AI infrastructure without compromising on power or flexibility.',
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
