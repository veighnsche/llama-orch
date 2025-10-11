<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-003: Implemented WhatIsRbee component -->
<script setup lang="ts">
interface Props {
  title?: string
  description?: string
  closingText?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'What is rbee?',
  description: 'rbee (pronounced "are-bee") is an open-source AI orchestration platform that turns all the computers in your home or office network into a unified AI infrastructure.',
  closingText: 'Whether you\'re a developer building AI tools, someone with idle GPUs to monetize, or an enterprise needing compliant AI infrastructure—rbee gives you complete control.',
})
</script>

<template>
  <section class="py-16 bg-secondary">
    <div class="container mx-auto px-4 max-w-4xl">
      <div class="text-center space-y-6">
        <h2 class="text-3xl md:text-4xl font-bold text-foreground">{{ title }}</h2>

        <p class="text-xl text-foreground leading-relaxed">
          <span class="font-semibold text-foreground">rbee</span> (pronounced "are-bee") is an{" "}
          <span class="font-semibold text-primary">open-source AI orchestration platform</span> that turns all
          the computers in your home or office network into a unified AI infrastructure.
        </p>

        <div class="grid md:grid-cols-3 gap-6 pt-8">
          <div class="bg-card p-6 rounded-lg border border-border">
            <div class="text-4xl font-bold text-primary mb-2">$0</div>
            <div class="text-sm text-muted-foreground">Monthly costs after setup. Just electricity.</div>
          </div>

          <div class="bg-card p-6 rounded-lg border border-border">
            <div class="text-4xl font-bold text-primary mb-2">100%</div>
            <div class="text-sm text-muted-foreground">Private. Your code and data never leave your network.</div>
          </div>

          <div class="bg-card p-6 rounded-lg border border-border">
            <div class="text-4xl font-bold text-primary mb-2">All</div>
            <div class="text-sm text-muted-foreground">Your GPUs working together—CUDA, Metal, CPU.</div>
          </div>
        </div>

        <p class="text-lg text-muted-foreground pt-4">
          {{ closingText }}
        </p>
      </div>
    </div>
  </section>
</template>
