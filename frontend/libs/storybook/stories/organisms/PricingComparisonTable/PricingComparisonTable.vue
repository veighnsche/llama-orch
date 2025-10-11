<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-002: Implemented PricingComparisonTable organism -->
<script setup lang="ts">
import { Check, X } from 'lucide-vue-next'

interface Props {
  title?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Detailed Feature Comparison',
})
</script>

<template>
  <section class="py-24 bg-secondary">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
          {{ title }}
        </h2>
      </div>

      <div class="max-w-5xl mx-auto overflow-x-auto">
        <table class="w-full bg-card border border-border rounded-lg">
          <thead>
            <tr class="border-b border-border">
              <th class="text-left p-4 font-semibold text-foreground">
                Feature
              </th>
              <th class="text-center p-4 font-semibold text-foreground">
                Home/Lab
              </th>
              <th class="text-center p-4 font-semibold text-foreground bg-primary/10">
                Team
              </th>
              <th class="text-center p-4 font-semibold text-foreground">
                Enterprise
              </th>
            </tr>
          </thead>
          <tbody>
            <tr class="border-b border-border">
              <td class="p-4 text-muted-foreground">Number of GPUs</td>
              <td class="text-center p-4 text-foreground">Unlimited</td>
              <td class="text-center p-4 text-foreground bg-primary/10">Unlimited</td>
              <td class="text-center p-4 text-foreground">Unlimited</td>
            </tr>
            <tr class="border-b border-border">
              <td class="p-4 text-muted-foreground">OpenAI-compatible API</td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
              <td class="text-center p-4 bg-primary/10">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-border">
              <td class="p-4 text-muted-foreground">Multi-GPU orchestration</td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
              <td class="text-center p-4 bg-primary/10">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-border">
              <td class="p-4 text-muted-foreground">Rhai scheduler</td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
              <td class="text-center p-4 bg-primary/10">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-border">
              <td class="p-4 text-muted-foreground">CLI access</td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
              <td class="text-center p-4 bg-primary/10">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-border">
              <td class="p-4 text-muted-foreground">Web UI</td>
              <td class="text-center p-4">
                <X class="h-5 w-5 text-muted-foreground/50 mx-auto" />
              </td>
              <td class="text-center p-4 bg-primary/10">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-border">
              <td class="p-4 text-muted-foreground">Team collaboration</td>
              <td class="text-center p-4">
                <X class="h-5 w-5 text-muted-foreground/50 mx-auto" />
              </td>
              <td class="text-center p-4 bg-primary/10">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-border">
              <td class="p-4 text-muted-foreground">Support</td>
              <td class="text-center p-4 text-muted-foreground">Community</td>
              <td class="text-center p-4 text-foreground bg-primary/10">Priority Email</td>
              <td class="text-center p-4 text-foreground">Dedicated</td>
            </tr>
            <tr class="border-b border-border">
              <td class="p-4 text-muted-foreground">SLA</td>
              <td class="text-center p-4">
                <X class="h-5 w-5 text-muted-foreground/50 mx-auto" />
              </td>
              <td class="text-center p-4 bg-primary/10">
                <X class="h-5 w-5 text-muted-foreground/50 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-border">
              <td class="p-4 text-muted-foreground">White-label</td>
              <td class="text-center p-4">
                <X class="h-5 w-5 text-muted-foreground/50 mx-auto" />
              </td>
              <td class="text-center p-4 bg-primary/10">
                <X class="h-5 w-5 text-muted-foreground/50 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
            </tr>
            <tr>
              <td class="p-4 text-muted-foreground">Professional services</td>
              <td class="text-center p-4">
                <X class="h-5 w-5 text-muted-foreground/50 mx-auto" />
              </td>
              <td class="text-center p-4 bg-primary/10">
                <X class="h-5 w-5 text-muted-foreground/50 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>
</template>
