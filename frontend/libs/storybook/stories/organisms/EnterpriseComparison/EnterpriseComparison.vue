<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented EnterpriseComparison -->
<script setup lang="ts">
import { Check, X } from 'lucide-vue-next'

interface ComparisonRow {
  feature: string
  rbee: boolean | string
  openai: boolean | string
  azure: boolean | string
}

interface Props {
  title?: string
  subtitle?: string
  comparisonData?: ComparisonRow[]
  disclaimer?: string
}

withDefaults(defineProps<Props>(), {
  title: 'Why Enterprises Choose rbee',
  subtitle: "Compare rbee's compliance and security features against external AI providers.",
  comparisonData: () => [
    { feature: 'Data Sovereignty', rbee: true, openai: false, azure: 'Partial' },
    { feature: 'EU-Only Deployment', rbee: true, openai: false, azure: 'Partial' },
    { feature: 'GDPR Compliant', rbee: true, openai: 'Partial', azure: 'Partial' },
    { feature: 'Immutable Audit Logs', rbee: true, openai: false, azure: false },
    { feature: '7-Year Audit Retention', rbee: true, openai: false, azure: false },
    { feature: 'SOC2 Type II Ready', rbee: true, openai: true, azure: true },
    { feature: 'ISO 27001 Aligned', rbee: true, openai: true, azure: true },
    { feature: 'Zero US Cloud Dependencies', rbee: true, openai: false, azure: false },
    { feature: 'On-Premises Deployment', rbee: true, openai: false, azure: false },
    { feature: 'Complete Control', rbee: true, openai: false, azure: 'Partial' },
    { feature: 'Custom SLAs', rbee: true, openai: false, azure: true },
    { feature: 'White-Label Option', rbee: true, openai: false, azure: false },
  ],
  disclaimer: '* Comparison based on publicly available information as of October 2025',
})

const renderCell = (value: boolean | string) => {
  if (value === true) return { type: 'check', class: 'text-accent' }
  if (value === false) return { type: 'x', class: 'text-destructive' }
  return { type: 'text', text: value, class: 'text-muted-foreground' }
}
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

      <div class="overflow-x-auto">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-border">
              <th class="p-4 text-left text-sm font-semibold text-muted-foreground">Feature</th>
              <th class="bg-primary/5 p-4 text-center text-sm font-semibold text-primary">
                rbee (Self-Hosted)
              </th>
              <th class="p-4 text-center text-sm font-semibold text-muted-foreground">OpenAI / Anthropic</th>
              <th class="p-4 text-center text-sm font-semibold text-muted-foreground">Azure OpenAI</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, i) in comparisonData" :key="i" class="border-b border-border">
              <td class="p-4 text-sm text-muted-foreground">{{ row.feature }}</td>
              <td class="bg-primary/5 p-4 text-center">
                <Check v-if="renderCell(row.rbee).type === 'check'" class="mx-auto h-5 w-5" :class="renderCell(row.rbee).class" />
                <X v-else-if="renderCell(row.rbee).type === 'x'" class="mx-auto h-5 w-5" :class="renderCell(row.rbee).class" />
                <span v-else class="text-sm" :class="renderCell(row.rbee).class">{{ renderCell(row.rbee).text }}</span>
              </td>
              <td class="p-4 text-center">
                <Check v-if="renderCell(row.openai).type === 'check'" class="mx-auto h-5 w-5" :class="renderCell(row.openai).class" />
                <X v-else-if="renderCell(row.openai).type === 'x'" class="mx-auto h-5 w-5" :class="renderCell(row.openai).class" />
                <span v-else class="text-sm" :class="renderCell(row.openai).class">{{ renderCell(row.openai).text }}</span>
              </td>
              <td class="p-4 text-center">
                <Check v-if="renderCell(row.azure).type === 'check'" class="mx-auto h-5 w-5" :class="renderCell(row.azure).class" />
                <X v-else-if="renderCell(row.azure).type === 'x'" class="mx-auto h-5 w-5" :class="renderCell(row.azure).class" />
                <span v-else class="text-sm" :class="renderCell(row.azure).class">{{ renderCell(row.azure).text }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="mt-8 text-center text-sm text-muted-foreground">
        {{ disclaimer }}
      </div>
    </div>
  </section>
</template>
