<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-006: Implemented CTASection component -->
<script setup lang="ts">
import { ArrowRight, BookOpen, MessageCircle } from 'lucide-vue-next'
import { Button } from '~/stories'

interface Props {
  title?: string
  highlightText?: string
  description?: string
  footerText?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Take Control of Your',
  highlightText: 'AI Infrastructure Today.',
  description: 'Join hundreds of users, providers, and enterprises who\'ve chosen independence.',
  footerText: '100% open source. No credit card required. Install in 15 minutes.',
})
</script>

<template>
  <section class="py-24 bg-gradient-to-br from-background via-secondary to-accent">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center space-y-8">
        <h2 class="text-4xl lg:text-6xl font-bold text-foreground mb-6 text-balance">
          {{ title }}
          <br />
          <span class="text-primary">{{ highlightText }}</span>
        </h2>

        <p class="text-xl text-muted-foreground leading-relaxed">
          {{ description }}
        </p>

        <div class="flex flex-col sm:flex-row gap-4 justify-center pt-4">
          <Button
            size="lg"
            class="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-lg h-14 px-8"
          >
            Get Started Free
            <ArrowRight class="ml-2 h-5 w-5" />
          </Button>
          <Button
            size="lg"
            variant="outline"
            class="h-14 px-8"
          >
            <BookOpen class="mr-2 h-5 w-5" />
            View Documentation
          </Button>
          <Button
            size="lg"
            variant="outline"
            class="h-14 px-8"
          >
            <MessageCircle class="mr-2 h-5 w-5" />
            Join Discord
          </Button>
        </div>

        <p class="text-muted-foreground pt-4">{{ footerText }}</p>
      </div>
    </div>
  </section>
</template>
