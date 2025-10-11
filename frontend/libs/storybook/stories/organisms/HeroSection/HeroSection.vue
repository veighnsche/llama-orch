<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-003: Implemented HeroSection component -->
<script setup lang="ts">
import { Button } from 'rbee-storybook/stories'
import { ArrowRight, Github, Star } from 'lucide-vue-next'

interface Props {
  title?: string
  subtitle?: string
  description?: string
  primaryButtonText?: string
  secondaryButtonText?: string
  githubStars?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'AI Infrastructure.',
  subtitle: 'On Your Terms.',
  description: 'Orchestrate AI inference across any hardware—your GPUs, your network, your rules. Build with AI, monetize idle hardware, or ensure compliance. Zero vendor lock-in.',
  primaryButtonText: 'Get Started Free',
  secondaryButtonText: 'View Documentation',
  githubStars: '1,200+',
})
</script>

<template>
  <section class="relative min-h-screen flex items-center bg-gradient-to-br from-background via-secondary to-muted dark:from-background dark:via-card dark:to-secondary">
    <div class="container mx-auto px-4 py-20">
      <div class="grid lg:grid-cols-2 gap-12 items-center">
        <!-- Left: Messaging -->
        <div class="space-y-8">
          <div class="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 border border-primary/20 rounded-full text-primary text-sm">
            <span class="relative flex h-2 w-2">
              <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
              <span class="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
            </span>
            100% Open Source • MIT License
          </div>

          <h1 class="text-5xl lg:text-7xl font-bold text-foreground leading-tight text-balance">
            {{ title }}
            <br />
            <span class="text-primary">{{ subtitle }}</span>
          </h1>

          <p class="text-xl text-muted-foreground leading-relaxed text-pretty">
            {{ description }}
          </p>

          <div class="flex flex-col sm:flex-row gap-4">
            <Button
              size="lg"
              class="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-lg h-14 px-8"
            >
              {{ primaryButtonText }}
              <ArrowRight class="ml-2 h-5 w-5" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              class="border-border text-foreground hover:bg-secondary h-14 px-8 bg-transparent"
            >
              {{ secondaryButtonText }}
            </Button>
          </div>

          <!-- Trust Indicators -->
          <div class="flex flex-wrap gap-6 pt-4">
            <div class="flex items-center gap-2 text-muted-foreground">
              <Github class="h-5 w-5" />
              <span class="text-sm">Open Source</span>
            </div>
            <div class="flex items-center gap-2 text-muted-foreground">
              <Star class="h-5 w-5 fill-primary text-primary" />
              <span class="text-sm">{{ githubStars }} Stars</span>
            </div>
            <div class="flex items-center gap-2 text-muted-foreground">
              <div class="h-5 w-5 flex items-center justify-center text-xs font-bold border border-border rounded">
                API
              </div>
              <span class="text-sm">OpenAI-Compatible</span>
            </div>
            <div class="flex items-center gap-2 text-muted-foreground">
              <div class="h-5 w-5 flex items-center justify-center text-xs font-bold border border-border rounded">
                $0
              </div>
              <span class="text-sm">No Cloud Required</span>
            </div>
          </div>
        </div>

        <!-- Right: Terminal Visual -->
        <div class="relative">
          <div class="bg-card border border-border rounded-lg overflow-hidden shadow-2xl">
            <div class="flex items-center gap-2 px-4 py-3 bg-secondary border-b border-border">
              <div class="flex gap-2">
                <div class="h-3 w-3 rounded-full bg-red-500"></div>
                <div class="h-3 w-3 rounded-full bg-amber-500"></div>
                <div class="h-3 w-3 rounded-full bg-green-500"></div>
              </div>
              <span class="text-muted-foreground text-sm ml-2 font-mono">rbee-keeper</span>
            </div>
            <div class="p-6 font-mono text-sm space-y-3">
              <div class="text-muted-foreground">
                <span class="text-green-400">$</span> rbee-keeper infer --model llama-3.1-70b
              </div>
              <div class="text-foreground pl-4">
                <span class="text-primary">→</span> Loading model across 3 GPUs...
              </div>
              <div class="text-foreground pl-4">
                <span class="text-green-400">✓</span> Model ready (2.3s)
              </div>
              <div class="text-slate-400 pl-4">
                <span class="text-blue-400">Prompt:</span> Generate REST API for user management
              </div>
              <div class="text-slate-300 pl-4 leading-relaxed">
                <span class="text-amber-400 animate-pulse">▊</span> Generating code...
              </div>

              <!-- GPU Utilization -->
              <div class="pt-4 space-y-2">
                <div class="text-muted-foreground text-xs">GPU Utilization:</div>
                <div class="space-y-1">
                  <div class="flex items-center gap-2">
                    <span class="text-muted-foreground text-xs w-24">workstation</span>
                    <div class="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                      <div class="h-full bg-primary w-[85%]"></div>
                    </div>
                    <span class="text-slate-400 text-xs">85%</span>
                  </div>
                  <div class="flex items-center gap-2">
                    <span class="text-muted-foreground text-xs w-24">mac-studio</span>
                    <div class="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                      <div class="h-full bg-primary w-[72%]"></div>
                    </div>
                    <span class="text-slate-400 text-xs">72%</span>
                  </div>
                  <div class="flex items-center gap-2">
                    <span class="text-muted-foreground text-xs w-24">gaming-pc</span>
                    <div class="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                      <div class="h-full bg-primary w-[91%]"></div>
                    </div>
                    <span class="text-slate-400 text-xs">91%</span>
                  </div>
                </div>
              </div>

              <!-- Cost Counter -->
              <div class="pt-2 flex items-center justify-between text-xs">
                <span class="text-muted-foreground">Cost:</span>
                <span class="text-green-400 font-bold">$0.00</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
