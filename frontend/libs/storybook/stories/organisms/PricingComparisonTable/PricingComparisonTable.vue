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
  <section class="py-24 bg-slate-50">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-slate-900 mb-6 text-balance">
          {{ title }}
        </h2>
      </div>

      <div class="max-w-5xl mx-auto overflow-x-auto">
        <table class="w-full bg-white border border-slate-200 rounded-lg">
          <thead>
            <tr class="border-b border-slate-200">
              <th class="text-left p-4 font-semibold text-slate-900">
                Feature
              </th>
              <th class="text-center p-4 font-semibold text-slate-900">
                Home/Lab
              </th>
              <th class="text-center p-4 font-semibold text-slate-900 bg-amber-50">
                Team
              </th>
              <th class="text-center p-4 font-semibold text-slate-900">
                Enterprise
              </th>
            </tr>
          </thead>
          <tbody>
            <tr class="border-b border-slate-200">
              <td class="p-4 text-slate-600">Number of GPUs</td>
              <td class="text-center p-4 text-slate-900">Unlimited</td>
              <td class="text-center p-4 text-slate-900 bg-amber-50">Unlimited</td>
              <td class="text-center p-4 text-slate-900">Unlimited</td>
            </tr>
            <tr class="border-b border-slate-200">
              <td class="p-4 text-slate-600">OpenAI-compatible API</td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
              <td class="text-center p-4 bg-amber-50">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-slate-200">
              <td class="p-4 text-slate-600">Multi-GPU orchestration</td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
              <td class="text-center p-4 bg-amber-50">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-slate-200">
              <td class="p-4 text-slate-600">Rhai scheduler</td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
              <td class="text-center p-4 bg-amber-50">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-slate-200">
              <td class="p-4 text-slate-600">CLI access</td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
              <td class="text-center p-4 bg-amber-50">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-slate-200">
              <td class="p-4 text-slate-600">Web UI</td>
              <td class="text-center p-4">
                <X class="h-5 w-5 text-slate-300 mx-auto" />
              </td>
              <td class="text-center p-4 bg-amber-50">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-slate-200">
              <td class="p-4 text-slate-600">Team collaboration</td>
              <td class="text-center p-4">
                <X class="h-5 w-5 text-slate-300 mx-auto" />
              </td>
              <td class="text-center p-4 bg-amber-50">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-slate-200">
              <td class="p-4 text-slate-600">Support</td>
              <td class="text-center p-4 text-slate-600">Community</td>
              <td class="text-center p-4 text-slate-900 bg-amber-50">Priority Email</td>
              <td class="text-center p-4 text-slate-900">Dedicated</td>
            </tr>
            <tr class="border-b border-slate-200">
              <td class="p-4 text-slate-600">SLA</td>
              <td class="text-center p-4">
                <X class="h-5 w-5 text-slate-300 mx-auto" />
              </td>
              <td class="text-center p-4 bg-amber-50">
                <X class="h-5 w-5 text-slate-300 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
            </tr>
            <tr class="border-b border-slate-200">
              <td class="p-4 text-slate-600">White-label</td>
              <td class="text-center p-4">
                <X class="h-5 w-5 text-slate-300 mx-auto" />
              </td>
              <td class="text-center p-4 bg-amber-50">
                <X class="h-5 w-5 text-slate-300 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
            </tr>
            <tr>
              <td class="p-4 text-slate-600">Professional services</td>
              <td class="text-center p-4">
                <X class="h-5 w-5 text-slate-300 mx-auto" />
              </td>
              <td class="text-center p-4 bg-amber-50">
                <X class="h-5 w-5 text-slate-300 mx-auto" />
              </td>
              <td class="text-center p-4">
                <Check class="h-5 w-5 text-green-600 mx-auto" />
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>
</template>
