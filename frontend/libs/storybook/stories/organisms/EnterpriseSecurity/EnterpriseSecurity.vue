<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented EnterpriseSecurity -->
<script setup lang="ts">
import { Shield, Lock, Eye, Server, Clock } from 'lucide-vue-next'

interface SecurityCrate {
  icon: any
  title: string
  subtitle: string
  description: string
  features: string[]
}

interface Guarantee {
  value: string
  label: string
}

interface Props {
  title?: string
  subtitle?: string
  securityCrates?: SecurityCrate[]
  guarantees?: Guarantee[]
}

withDefaults(defineProps<Props>(), {
  title: 'Enterprise-Grade Security',
  subtitle:
    'Five specialized security crates provide defense-in-depth protection against the most sophisticated attacks.',
  securityCrates: () => [
    {
      icon: Lock,
      title: 'auth-min: Zero-Trust Authentication',
      subtitle: 'The Trickster Guardians',
      description:
        'Timing-safe token comparison prevents CWE-208 attacks. Token fingerprinting for safe logging. Bind policy enforcement prevents accidental exposure.',
      features: [
        'Timing-safe comparison (constant-time)',
        'Token fingerprinting (SHA-256)',
        'Bearer token parsing (RFC 6750)',
        'Bind policy enforcement',
      ],
    },
    {
      icon: Eye,
      title: 'audit-logging: Compliance Engine',
      subtitle: 'Legally Defensible Proof',
      description:
        'Immutable audit trail with 32 event types. Tamper detection via blockchain-style hash chains. 7-year retention for GDPR compliance.',
      features: [
        'Immutable audit trail (append-only)',
        '32 event types across 7 categories',
        'Tamper detection (hash chains)',
        '7-year retention (GDPR)',
      ],
    },
    {
      icon: Shield,
      title: 'input-validation: First Line of Defense',
      subtitle: 'Trust No Input',
      description:
        'Prevents injection attacks and resource exhaustion. Validates identifiers, model references, prompts, and paths before processing.',
      features: [
        'SQL injection prevention',
        'Command injection prevention',
        'Path traversal prevention',
        'Resource exhaustion prevention',
      ],
    },
    {
      icon: Server,
      title: 'secrets-management: Credential Guardian',
      subtitle: 'Never in Environment',
      description:
        'File-based secrets with memory zeroization. Systemd credentials support. Timing-safe verification prevents memory dump attacks.',
      features: [
        'File-based loading (not env vars)',
        'Memory zeroization on drop',
        'Permission validation (0600)',
        'Timing-safe verification',
      ],
    },
    {
      icon: Clock,
      title: 'deadline-propagation: Performance Enforcer',
      subtitle: 'Every Millisecond Counts',
      description:
        'Ensures rbee never wastes cycles on doomed work. Deadline propagation from client to worker. Aborts immediately when deadline exceeded.',
      features: [
        'Deadline propagation (client → worker)',
        'Remaining time calculation',
        'Deadline enforcement (abort if insufficient)',
        'Timeout responses (504 Gateway Timeout)',
      ],
    },
  ],
  guarantees: () => [
    { value: '< 10%', label: 'Timing variance (constant-time)' },
    { value: '100%', label: 'Token fingerprinting (no raw tokens)' },
    { value: 'Zero', label: 'Memory leaks (zeroization on drop)' },
  ],
})
</script>

<template>
  <section class="border-b border-border bg-background px-6 py-24">
    <div class="mx-auto max-w-7xl">
      <div class="mb-16 text-center">
        <h2 class="mb-4 text-4xl font-bold text-foreground">{{ title }}</h2>
        <p class="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
          {{ subtitle }}
        </p>
      </div>

      <div class="grid gap-8 lg:grid-cols-2">
        <div
          v-for="(crate, i) in securityCrates"
          :key="i"
          :class="[
            'rounded-lg border border-border bg-card p-8',
            i === securityCrates.length - 1 ? 'lg:col-span-2' : '',
          ]"
        >
          <div class="mb-4 flex items-center gap-3">
            <div class="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <component :is="crate.icon" class="h-6 w-6 text-primary" />
            </div>
            <div>
              <h3 class="text-xl font-bold text-foreground">{{ crate.title }}</h3>
              <p class="text-sm text-muted-foreground">{{ crate.subtitle }}</p>
            </div>
          </div>

          <p class="mb-4 leading-relaxed text-muted-foreground">
            {{ crate.description }}
          </p>

          <div :class="i === securityCrates.length - 1 ? 'grid gap-4 md:grid-cols-2' : 'space-y-2'">
            <div v-for="(feature, j) in crate.features" :key="j" class="flex items-start gap-2 text-sm text-muted-foreground">
              <span class="text-accent">✓</span>
              <span>{{ feature }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Security Guarantees -->
      <div class="mt-12 rounded-lg border border-primary/20 bg-primary/5 p-8">
        <h3 class="mb-6 text-center text-2xl font-semibold text-foreground">Security Guarantees</h3>
        <div class="grid gap-4 md:grid-cols-3">
          <div v-for="(guarantee, i) in guarantees" :key="i" class="text-center">
            <div class="mb-2 text-3xl font-bold text-primary">{{ guarantee.value }}</div>
            <div class="text-sm text-muted-foreground">{{ guarantee.label }}</div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
