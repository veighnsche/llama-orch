<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-006: Implemented ComparisonSection component -->
<script setup lang="ts">
import { Check, X } from 'lucide-vue-next'

interface Props {
  title?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Why Developers Choose rbee',
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

      <div class="max-w-6xl mx-auto overflow-x-auto">
        <table class="w-full bg-card border border-border rounded-lg">
          <thead>
            <tr class="border-b border-border">
              <th class="text-left p-4 font-bold text-card-foreground">Feature</th>
              <th class="p-4 font-bold text-primary bg-accent">rbee</th>
              <th class="p-4 font-medium text-muted-foreground">OpenAI/Anthropic</th>
              <th class="p-4 font-medium text-muted-foreground">Ollama</th>
              <th class="p-4 font-medium text-muted-foreground">Runpod/Vast.ai</th>
            </tr>
          </thead>
          <tbody>
            <tr class="border-b border-border">
              <td class="p-4 font-medium text-card-foreground">Cost</td>
              <td class="p-4 text-center bg-accent">
                <div class="text-sm font-medium text-foreground">$0</div>
                <div class="text-xs text-muted-foreground">(your hardware)</div>
              </td>
              <td class="p-4 text-center text-sm text-muted-foreground">$20-100/mo per dev</td>
              <td class="p-4 text-center text-sm text-muted-foreground">$0</td>
              <td class="p-4 text-center text-sm text-muted-foreground">$0.50-2/hr</td>
            </tr>
            <tr class="border-b border-border">
              <td class="p-4 font-medium text-card-foreground">Privacy</td>
              <td class="p-4 text-center bg-accent">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
                <div class="text-xs text-muted-foreground mt-1">Complete</div>
              </td>
              <td class="p-4 text-center">
                <X class="h-5 w-5 text-red-500 mx-auto" />
                <div class="text-xs text-muted-foreground mt-1">Limited</div>
              </td>
              <td class="p-4 text-center">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
                <div class="text-xs text-muted-foreground mt-1">Complete</div>
              </td>
              <td class="p-4 text-center">
                <X class="h-5 w-5 text-red-500 mx-auto" />
                <div class="text-xs text-muted-foreground mt-1">Limited</div>
              </td>
            </tr>
            <tr class="border-b border-border">
              <td class="p-4 font-medium text-card-foreground">Multi-GPU</td>
              <td class="p-4 text-center bg-accent">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
                <div class="text-xs text-muted-foreground mt-1">Orchestrated</div>
              </td>
              <td class="p-4 text-center text-sm text-muted-foreground">N/A</td>
              <td class="p-4 text-center">
                <div class="text-sm text-muted-foreground">Limited</div>
              </td>
              <td class="p-4 text-center">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-border">
              <td class="p-4 font-medium text-card-foreground">OpenAI API</td>
              <td class="p-4 text-center bg-accent">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
              <td class="p-4 text-center">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
              <td class="p-4 text-center">
                <div class="text-sm text-muted-foreground">Partial</div>
              </td>
              <td class="p-4 text-center">
                <X class="h-5 w-5 text-red-500 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-border">
              <td class="p-4 font-medium text-card-foreground">Custom Routing</td>
              <td class="p-4 text-center bg-accent">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
                <div class="text-xs text-muted-foreground mt-1">Rhai scripts</div>
              </td>
              <td class="p-4 text-center">
                <X class="h-5 w-5 text-red-500 mx-auto" />
              </td>
              <td class="p-4 text-center">
                <X class="h-5 w-5 text-red-500 mx-auto" />
              </td>
              <td class="p-4 text-center">
                <X class="h-5 w-5 text-red-500 mx-auto" />
              </td>
            </tr>
            <tr>
              <td class="p-4 font-medium text-card-foreground">Rate Limits</td>
              <td class="p-4 text-center bg-accent">
                <div class="text-sm font-medium text-green-600">None</div>
              </td>
              <td class="p-4 text-center">
                <div class="text-sm text-red-600">Yes</div>
              </td>
              <td class="p-4 text-center">
                <div class="text-sm font-medium text-green-600">None</div>
              </td>
              <td class="p-4 text-center">
                <div class="text-sm text-red-600">Yes</div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>
</template>
