<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-007: Implemented ProvidersSecurity component -->
<script setup lang="ts">
import { Shield, Lock, Eye, FileCheck } from 'lucide-vue-next'
</script>

<template>
  <section class="border-b border-border bg-background px-6 py-24">
    <div class="mx-auto max-w-7xl">
      <div class="mb-16 text-center">
        <h2 class="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">Your Security Is Our Priority</h2>
        <p class="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
          Enterprise-grade security protects your hardware, data, and earnings.
        </p>
      </div>

      <div class="grid gap-8 md:grid-cols-2">
        <div class="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
          <div class="mb-6 flex items-center gap-4">
            <div class="flex h-14 w-14 items-center justify-center rounded-xl bg-green-500/10">
              <Shield :size="28" class="text-green-400" />
            </div>
            <div>
              <h3 class="text-xl font-bold text-foreground">Sandboxed Execution</h3>
              <div class="text-sm text-muted-foreground">Complete isolation</div>
            </div>
          </div>
          <p class="mb-4 text-pretty leading-relaxed text-muted-foreground">
            All jobs run in isolated sandboxes with no access to your files, network, or personal data. Your system stays completely secure.
          </p>
          <ul class="space-y-2 text-sm text-muted-foreground">
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              No file system access
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              No network access
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              No access to personal data
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              Automatic cleanup after jobs
            </li>
          </ul>
        </div>

        <div class="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
          <div class="mb-6 flex items-center gap-4">
            <div class="flex h-14 w-14 items-center justify-center rounded-xl bg-green-500/10">
              <Lock :size="28" class="text-green-400" />
            </div>
            <div>
              <h3 class="text-xl font-bold text-foreground">Encrypted Communication</h3>
              <div class="text-sm text-muted-foreground">End-to-end encryption</div>
            </div>
          </div>
          <p class="mb-4 text-pretty leading-relaxed text-muted-foreground">
            All communication between your GPU and the marketplace is encrypted. Your earnings data and job details are protected.
          </p>
          <ul class="space-y-2 text-sm text-muted-foreground">
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              TLS 1.3 encryption
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              Secure payment processing
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              Protected earnings data
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              Private job details
            </li>
          </ul>
        </div>

        <div class="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
          <div class="mb-6 flex items-center gap-4">
            <div class="flex h-14 w-14 items-center justify-center rounded-xl bg-green-500/10">
              <Eye :size="28" class="text-green-400" />
            </div>
            <div>
              <h3 class="text-xl font-bold text-foreground">Malware Scanning</h3>
              <div class="text-sm text-muted-foreground">Automatic protection</div>
            </div>
          </div>
          <p class="mb-4 text-pretty leading-relaxed text-muted-foreground">
            Every job is automatically scanned for malware before execution. Suspicious jobs are blocked and reported.
          </p>
          <ul class="space-y-2 text-sm text-muted-foreground">
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              Real-time malware detection
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              Automatic job blocking
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              Threat intelligence updates
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              Customer vetting process
            </li>
          </ul>
        </div>

        <div class="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
          <div class="mb-6 flex items-center gap-4">
            <div class="flex h-14 w-14 items-center justify-center rounded-xl bg-green-500/10">
              <FileCheck :size="28" class="text-green-400" />
            </div>
            <div>
              <h3 class="text-xl font-bold text-foreground">Hardware Protection</h3>
              <div class="text-sm text-muted-foreground">Warranty-safe operation</div>
            </div>
          </div>
          <p class="mb-4 text-pretty leading-relaxed text-muted-foreground">
            Temperature monitoring, automatic cooldown periods, and power limits protect your hardware and maintain warranty coverage.
          </p>
          <ul class="space-y-2 text-sm text-muted-foreground">
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              Temperature monitoring
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              Automatic cooldown periods
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              Power consumption limits
            </li>
            <li class="flex items-center gap-2">
              <div class="h-1.5 w-1.5 rounded-full bg-green-400" />
              Hardware health monitoring
            </li>
          </ul>
        </div>
      </div>

      <div class="mt-12 rounded-2xl border border-green-500/50 bg-green-500/20 p-8 text-center">
        <p class="text-balance text-lg font-medium text-green-300">
          Plus: €1M insurance coverage included for all providers. Your hardware is protected.
        </p>
      </div>
    </div>
  </section>
</template>
