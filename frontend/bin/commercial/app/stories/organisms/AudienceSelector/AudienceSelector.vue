<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-004: Implemented AudienceSelector organism -->
<script setup lang="ts">
import { Card, Button } from '~/stories'
import { ArrowRight, Code2, Server, Shield } from 'lucide-vue-next'

interface Props {
  title?: string
  subtitle?: string
  description?: string
}

withDefaults(defineProps<Props>(), {
  title: 'Choose Your Path',
  subtitle: 'Where do you want to start?',
  description: 'rbee adapts to your needs. Whether you\'re building, providing, or securing—we have a path designed for you.',
})
</script>

<template>
  <section class="relative bg-gradient-to-b from-background via-secondary to-background py-24 sm:py-32">
    <div class="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/20 to-transparent" />

    <div class="mx-auto max-w-7xl px-6 lg:px-8">
      <div class="mx-auto mb-16 max-w-3xl text-center">
        <p class="mb-4 font-sans text-sm font-medium uppercase tracking-wider text-primary">
          {{ title }}
        </p>

        <h2 class="mb-6 font-sans text-3xl font-semibold tracking-tight text-foreground sm:text-4xl lg:text-5xl">
          {{ subtitle }}
        </h2>
        <p class="font-sans text-lg leading-relaxed text-muted-foreground">
          {{ description }}
        </p>
      </div>

      <div class="mx-auto grid max-w-6xl gap-6 lg:grid-cols-3 lg:gap-8">
        <!-- Developers Card -->
        <Card class="group relative overflow-hidden border-border/50 bg-card/30 p-8 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:border-blue-500/50 hover:bg-card/60 hover:shadow-2xl hover:shadow-blue-500/20">
          <div class="absolute inset-0 -z-10 bg-gradient-to-br from-blue-500/0 via-blue-500/0 to-blue-500/0 opacity-0 transition-all duration-500 group-hover:from-blue-500/5 group-hover:via-blue-500/10 group-hover:to-transparent group-hover:opacity-100" />

          <div class="mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 shadow-lg shadow-blue-500/30 transition-all duration-300 group-hover:scale-110 group-hover:shadow-blue-500/50">
            <Code2 :size="28" class="text-white" />
          </div>

          <div class="mb-2 text-sm font-medium uppercase tracking-wider text-blue-400">For Developers</div>
          <h3 class="mb-3 font-sans text-2xl font-semibold text-foreground">Build on Your Hardware</h3>

          <p class="mb-6 font-sans leading-relaxed text-muted-foreground">
            Use your homelab GPUs to power AI coding tools. OpenAI-compatible API works with Zed, Cursor, and any tool
            you already use.
          </p>

          <ul class="mb-8 space-y-3 text-sm text-card-foreground">
            <li class="flex items-start gap-2">
              <span class="mt-1 text-blue-400">→</span>
              <span>Zero API costs, unlimited usage</span>
            </li>
            <li class="flex items-start gap-2">
              <span class="mt-1 text-blue-400">→</span>
              <span>Complete privacy, no data leaves your network</span>
            </li>
            <li class="flex items-start gap-2">
              <span class="mt-1 text-blue-400">→</span>
              <span>Build custom AI agents with TypeScript</span>
            </li>
          </ul>

          <a href="/developers" class="block">
            <Button class="w-full bg-blue-600 font-medium text-white transition-all hover:bg-blue-700 hover:shadow-lg hover:shadow-blue-500/30">
              Explore Developer Path
              <ArrowRight :size="16" class="ml-2 transition-transform group-hover:translate-x-1" />
            </Button>
          </a>
        </Card>

        <!-- GPU Providers Card -->
        <Card class="group relative overflow-hidden border-border/50 bg-card/30 p-8 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:border-green-500/50 hover:bg-card/60 hover:shadow-2xl hover:shadow-green-500/20">
          <div class="absolute inset-0 -z-10 bg-gradient-to-br from-green-500/0 via-green-500/0 to-green-500/0 opacity-0 transition-all duration-500 group-hover:from-green-500/5 group-hover:via-green-500/10 group-hover:to-transparent group-hover:opacity-100" />

          <div class="mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br from-green-500 to-green-600 shadow-lg shadow-green-500/30 transition-all duration-300 group-hover:scale-110 group-hover:shadow-green-500/50">
            <Server :size="28" class="text-white" />
          </div>

          <div class="mb-2 text-sm font-medium uppercase tracking-wider text-green-400">For GPU Owners</div>
          <h3 class="mb-3 font-sans text-2xl font-semibold text-foreground">Monetize Your Hardware</h3>

          <p class="mb-6 font-sans leading-relaxed text-muted-foreground">
            Turn idle GPUs into revenue. Join the rbee marketplace and earn from your gaming PC, workstation, or
            server farm.
          </p>

          <ul class="mb-8 space-y-3 text-sm text-card-foreground">
            <li class="flex items-start gap-2">
              <span class="mt-1 text-green-400">→</span>
              <span>Set your own pricing and availability</span>
            </li>
            <li class="flex items-start gap-2">
              <span class="mt-1 text-green-400">→</span>
              <span>Secure platform with audit trails</span>
            </li>
            <li class="flex items-start gap-2">
              <span class="mt-1 text-green-400">→</span>
              <span>Passive income from existing hardware</span>
            </li>
          </ul>

          <a href="/gpu-providers" class="block">
            <Button class="w-full bg-green-600 font-medium text-white transition-all hover:bg-green-700 hover:shadow-lg hover:shadow-green-500/30">
              Become a Provider
              <ArrowRight :size="16" class="ml-2 transition-transform group-hover:translate-x-1" />
            </Button>
          </a>
        </Card>

        <!-- Enterprise Card -->
        <Card class="group relative overflow-hidden border-border/50 bg-card/30 p-8 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:border-primary/50 hover:bg-card/60 hover:shadow-2xl hover:shadow-primary/20">
          <div class="absolute inset-0 -z-10 bg-gradient-to-br from-primary/0 via-primary/0 to-primary/0 opacity-0 transition-all duration-500 group-hover:from-primary/5 group-hover:via-primary/10 group-hover:to-transparent group-hover:opacity-100" />

          <div class="mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-accent shadow-lg shadow-primary/30 transition-all duration-300 group-hover:scale-110 group-hover:shadow-primary/50">
            <Shield :size="28" class="text-white" />
          </div>

          <div class="mb-2 text-sm font-medium uppercase tracking-wider text-primary">For Enterprise</div>
          <h3 class="mb-3 font-sans text-2xl font-semibold text-foreground">Compliance & Security</h3>

          <p class="mb-6 font-sans leading-relaxed text-muted-foreground">
            Deploy AI infrastructure with EU compliance, comprehensive audit trails, and enterprise-grade security
            built-in from day one.
          </p>

          <ul class="mb-8 space-y-3 text-sm text-card-foreground">
            <li class="flex items-start gap-2">
              <span class="mt-1 text-primary">→</span>
              <span>GDPR-compliant with 7-year audit retention</span>
            </li>
            <li class="flex items-start gap-2">
              <span class="mt-1 text-primary">→</span>
              <span>SOC2 and ISO 27001 aligned</span>
            </li>
            <li class="flex items-start gap-2">
              <span class="mt-1 text-primary">→</span>
              <span>On-premises or private cloud deployment</span>
            </li>
          </ul>

          <a href="/enterprise" class="block">
            <Button class="w-full bg-primary font-medium text-primary-foreground transition-all hover:bg-accent hover:shadow-lg hover:shadow-primary/30">
              Enterprise Solutions
              <ArrowRight :size="16" class="ml-2 transition-transform group-hover:translate-x-1" />
            </Button>
          </a>
        </Card>
      </div>

      <div class="mx-auto mt-12 max-w-2xl text-center">
        <p class="font-sans text-sm leading-relaxed text-muted-foreground/70">
          Not sure which path fits? Keep scrolling to explore the full platform.
        </p>
      </div>
    </div>
  </section>
</template>
