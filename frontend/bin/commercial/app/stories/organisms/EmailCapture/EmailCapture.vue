<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-CONSOLIDATE: Implemented EmailCapture with design tokens -->
<script setup lang="ts">
import { ref } from 'vue'
import { Button, Input } from '~/stories'
import { Mail, CheckCircle2, Github } from 'lucide-vue-next'

interface Props {
  title?: string
  description?: string
  statusBadge?: string
  privacyText?: string
  githubText?: string
  githubUrl?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Be the First to Know',
  description: 'rbee is actively being built. Join our waitlist to get early access, updates on development progress, and exclusive launch benefits.',
  statusBadge: 'Currently in Development (M0 - 68% Complete)',
  privacyText: 'No spam. Unsubscribe anytime. We respect your privacy.',
  githubText: 'Star us on GitHub',
  githubUrl: 'https://github.com/veighnsche/llama-orch',
})

const email = ref('')
const submitted = ref(false)

const handleSubmit = (e: Event) => {
  e.preventDefault()
  // TODO: Wire up to actual email service (backend integration)
  console.log('Email submitted:', email.value)
  submitted.value = true
  setTimeout(() => {
    submitted.value = false
    email.value = ''
  }, 3000)
}
</script>

<template>
  <section class="py-24 bg-gradient-to-br from-background via-secondary to-background">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
      <div class="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 border border-primary/20 rounded-full text-primary text-sm font-medium mb-6">
        <span class="relative flex h-2 w-2">
          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
          <span class="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
        </span>
        {{ statusBadge }}
      </div>

      <h2 class="text-4xl md:text-5xl font-bold text-foreground mb-6">{{ title }}</h2>

      <p class="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed">
        {{ description }}
      </p>

      <form v-if="!submitted" @submit="handleSubmit" class="flex flex-col sm:flex-row gap-4 max-w-md mx-auto">
        <div class="flex-1 relative">
          <Mail class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
          <Input
            v-model="email"
            type="email"
            placeholder="your@email.com"
            required
            class="pl-10 h-12"
          />
        </div>
        <Button
          type="submit"
          size="lg"
          variant="default"
          class="h-12 px-8"
        >
          Join Waitlist
        </Button>
      </form>

      <div v-else class="flex items-center justify-center gap-2 text-primary text-lg">
        <CheckCircle2 :size="24" />
        <span>Thanks! We'll keep you updated.</span>
      </div>

      <p class="text-sm text-muted-foreground mt-6">{{ privacyText }}</p>

      <div class="mt-12 pt-12 border-t border-border">
        <p class="text-muted-foreground mb-4">Want to contribute or follow development?</p>
        <a
          :href="githubUrl"
          target="_blank"
          rel="noopener noreferrer"
          class="inline-flex items-center gap-2 text-primary hover:text-primary/80 transition-colors font-medium"
        >
          <Github :size="20" />
          {{ githubText }}
        </a>
      </div>
    </div>
  </section>
</template>
