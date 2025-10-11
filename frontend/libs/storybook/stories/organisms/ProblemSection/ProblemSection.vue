<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-003: Implemented ProblemSection component -->
<script setup lang="ts">
import { AlertTriangle, DollarSign, Lock } from 'lucide-vue-next'

interface Props {
  title?: string
  closingText?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'The Hidden Cost of AI Dependency',
  closingText: 'Whether you\'re building with AI, monetizing hardware, or ensuring compliance—dependency on external providers creates risk you can\'t afford.',
})
</script>

<template>
  <section class="py-24 bg-gradient-to-br from-destructive/10 via-background to-background dark:from-destructive/20">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
          {{ title }}
        </h2>
      </div>

      <div class="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <!-- Card 1 -->
        <div class="bg-card/50 border border-destructive/30 rounded-lg p-8 space-y-4 hover:border-destructive/50 transition-colors">
          <div class="h-12 w-12 rounded-lg bg-destructive/10 flex items-center justify-center">
            <AlertTriangle class="h-6 w-6 text-destructive" />
          </div>
          <h3 class="text-xl font-bold text-foreground">Loss of Control</h3>
          <p class="text-muted-foreground leading-relaxed">
            Providers change models, deprecate APIs, or shut down entirely. Your infrastructure, workflows, and
            business continuity depend on decisions you can't control.
          </p>
        </div>

        <!-- Card 2 -->
        <div class="bg-card/50 border border-destructive/30 rounded-lg p-8 space-y-4 hover:border-destructive/50 transition-colors">
          <div class="h-12 w-12 rounded-lg bg-destructive/10 flex items-center justify-center">
            <DollarSign class="h-6 w-6 text-destructive" />
          </div>
          <h3 class="text-xl font-bold text-foreground">Unpredictable Costs</h3>
          <p class="text-muted-foreground leading-relaxed">
            Pricing changes without warning. Usage scales, costs spiral. What starts affordable becomes unsustainable
            as your needs grow.
          </p>
        </div>

        <!-- Card 3 -->
        <div class="bg-card/50 border border-destructive/30 rounded-lg p-8 space-y-4 hover:border-destructive/50 transition-colors">
          <div class="h-12 w-12 rounded-lg bg-destructive/10 flex items-center justify-center">
            <Lock class="h-6 w-6 text-destructive" />
          </div>
          <h3 class="text-xl font-bold text-foreground">Privacy & Compliance Risks</h3>
          <p class="text-muted-foreground leading-relaxed">
            Sensitive data leaves your network. Compliance requirements clash with cloud dependencies. Audit trails
            are incomplete. Regulatory exposure grows.
          </p>
        </div>
      </div>

      <div class="max-w-3xl mx-auto text-center mt-12">
        <p class="text-xl text-muted-foreground leading-relaxed text-balance">
          {{ closingText }}
        </p>
      </div>
    </div>
  </section>
</template>
