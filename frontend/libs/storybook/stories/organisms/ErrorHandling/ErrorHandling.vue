<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented ErrorHandling -->
<script setup lang="ts">
import { Network, AlertTriangle, Database, Activity, CheckCircle2 } from 'lucide-vue-next'
</script>

<template>
  <section class="py-24 bg-background">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
          Comprehensive Error Handling
        </h2>
        <p class="text-xl text-muted-foreground leading-relaxed">
          19+ error scenarios with clear messages and actionable suggestions. No cryptic failures.
        </p>
      </div>

      <div class="max-w-5xl mx-auto">
        <div class="grid md:grid-cols-2 gap-6">
          <!-- Network & Connectivity -->
          <div class="bg-card border border-border rounded-lg p-6">
            <div class="flex items-center gap-3 mb-4">
              <div class="h-10 w-10 rounded-lg bg-red-100 flex items-center justify-center">
                <Network class="h-5 w-5 text-red-600" />
              </div>
              <h3 class="text-lg font-bold text-foreground">Network & Connectivity</h3>
            </div>
            <ul class="space-y-2 text-sm text-muted-foreground">
              <li class="flex items-start gap-2">
                <span class="text-destructive mt-1">•</span>
                <span>SSH connection timeout (3 retries with backoff)</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-destructive mt-1">•</span>
                <span>SSH authentication failure with fix suggestions</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-destructive mt-1">•</span>
                <span>HTTP connection failures with retry logic</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-destructive mt-1">•</span>
                <span>Connection loss during inference (partial results saved)</span>
              </li>
            </ul>
          </div>

          <!-- Resource Errors -->
          <div class="bg-card border border-border rounded-lg p-6">
            <div class="flex items-center gap-3 mb-4">
              <div class="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <AlertTriangle class="h-5 w-5 text-primary" />
              </div>
              <h3 class="text-lg font-bold text-foreground">Resource Errors</h3>
            </div>
            <ul class="space-y-2 text-sm text-muted-foreground">
              <li class="flex items-start gap-2">
                <span class="text-primary mt-1">•</span>
                <span>Insufficient RAM with model size suggestions</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-primary mt-1">•</span>
                <span>VRAM exhausted (FAIL FAST, no CPU fallback)</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-primary mt-1">•</span>
                <span>Disk space checks before downloads</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-primary mt-1">•</span>
                <span>OOM detection during model loading</span>
              </li>
            </ul>
          </div>

          <!-- Model & Backend -->
          <div class="bg-card border border-border rounded-lg p-6">
            <div class="flex items-center gap-3 mb-4">
              <div class="h-10 w-10 rounded-lg bg-blue-100 flex items-center justify-center">
                <Database class="h-5 w-5 text-blue-600" />
              </div>
              <h3 class="text-lg font-bold text-foreground">Model & Backend</h3>
            </div>
            <ul class="space-y-2 text-sm text-muted-foreground">
              <li class="flex items-start gap-2">
                <span class="text-blue-500 mt-1">•</span>
                <span>Model not found (404) with Hugging Face link</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-blue-500 mt-1">•</span>
                <span>Private model (403) with auth token suggestion</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-blue-500 mt-1">•</span>
                <span>Download failures with resume support</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-blue-500 mt-1">•</span>
                <span>Backend not available with alternatives</span>
              </li>
            </ul>
          </div>

          <!-- Process Lifecycle -->
          <div class="bg-card border border-border rounded-lg p-6">
            <div class="flex items-center gap-3 mb-4">
              <div class="h-10 w-10 rounded-lg bg-accent/10 flex items-center justify-center">
                <Activity class="h-5 w-5 text-accent" />
              </div>
              <h3 class="text-lg font-bold text-foreground">Process Lifecycle</h3>
            </div>
            <ul class="space-y-2 text-sm text-muted-foreground">
              <li class="flex items-start gap-2">
                <span class="text-accent mt-1">•</span>
                <span>Worker binary not found with install instructions</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-accent mt-1">•</span>
                <span>Worker crashes during startup (log suggestions)</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-accent mt-1">•</span>
                <span>Graceful shutdown with active requests</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-accent mt-1">•</span>
                <span>Force-kill after 30s timeout</span>
              </li>
            </ul>
          </div>
        </div>

        <div class="mt-8 bg-accent/10 border border-accent/30 rounded-lg p-6">
          <div class="flex items-start gap-3">
            <CheckCircle2 class="h-6 w-6 text-accent flex-shrink-0" />
            <div>
              <div class="font-bold text-accent-foreground mb-2">Exponential Backoff with Jitter</div>
              <div class="text-muted-foreground text-sm leading-relaxed">
                All retries use exponential backoff with random jitter (0.5-1.5x) to avoid thundering herd problems.
                SSH: 3 attempts. HTTP: 3 attempts. Downloads: 6 attempts with resume support.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
