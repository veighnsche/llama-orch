<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-007: Implemented ProvidersEarnings component -->
<script setup lang="ts">
import { ref, computed } from 'vue'
import { Button } from '~/stories'
import { SliderRoot, SliderTrack, SliderRange, SliderThumb } from 'radix-vue'

interface GPUModel {
  name: string
  baseRate: number
  vram: number
}

const gpuModels: GPUModel[] = [
  { name: 'RTX 4090', baseRate: 0.45, vram: 24 },
  { name: 'RTX 4080', baseRate: 0.35, vram: 16 },
  { name: 'RTX 4070 Ti', baseRate: 0.28, vram: 12 },
  { name: 'RTX 3090', baseRate: 0.32, vram: 24 },
  { name: 'RTX 3080', baseRate: 0.25, vram: 10 },
  { name: 'RTX 3070', baseRate: 0.18, vram: 8 }
]

const selectedGPU = ref(gpuModels[0])
const utilization = ref([80])
const hoursPerDay = ref([20])

const hourlyRate = computed(() => selectedGPU.value.baseRate)
const dailyEarnings = computed(() => hourlyRate.value * hoursPerDay.value[0] * (utilization.value[0] / 100))
const monthlyEarnings = computed(() => dailyEarnings.value * 30)
const yearlyEarnings = computed(() => monthlyEarnings.value * 12)
</script>

<template>
  <section id="earnings-calculator" class="border-b border-border bg-background px-6 py-24">
    <div class="mx-auto max-w-7xl">
      <div class="mb-16 text-center">
        <h2 class="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">
          Calculate Your Potential Earnings
        </h2>
        <p class="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
          See how much you could earn based on your GPU model, availability, and utilization.
        </p>
      </div>

      <div class="mx-auto max-w-4xl">
        <div class="grid gap-8 lg:grid-cols-2">
          <!-- Calculator Inputs -->
          <div class="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <h3 class="mb-6 text-xl font-bold text-foreground">Your Configuration</h3>

            <div class="space-y-6">
              <!-- GPU Selection -->
              <div>
                <label class="mb-3 block text-sm font-medium text-muted-foreground">Select Your GPU</label>
                <div class="grid gap-2">
                  <button
                    v-for="gpu in gpuModels"
                    :key="gpu.name"
                    @click="selectedGPU = gpu"
                    :class="[
                      'rounded-lg border p-3 text-left transition-all',
                      selectedGPU.name === gpu.name
                        ? 'border-primary bg-primary/10'
                        : 'border-border bg-background/50 hover:border-border/70'
                    ]"
                  >
                    <div class="flex items-center justify-between">
                      <div>
                        <div class="font-medium text-foreground">{{ gpu.name }}</div>
                        <div class="text-xs text-muted-foreground">{{ gpu.vram }}GB VRAM</div>
                      </div>
                      <div class="text-sm text-primary">€{{ gpu.baseRate }}/hr</div>
                    </div>
                  </button>
                </div>
              </div>

              <!-- Hours Per Day -->
              <div>
                <div class="mb-3 flex items-center justify-between">
                  <label class="text-sm font-medium text-muted-foreground">Hours Available Per Day</label>
                  <span class="text-lg font-bold text-primary">{{ hoursPerDay[0] }}h</span>
                </div>
                <SliderRoot
                  v-model="hoursPerDay"
                  :min="1"
                  :max="24"
                  :step="1"
                  class="relative flex w-full touch-none items-center select-none"
                >
                  <SliderTrack class="bg-muted relative h-1.5 w-full grow overflow-hidden rounded-full">
                    <SliderRange class="bg-primary absolute h-full" />
                  </SliderTrack>
                  <SliderThumb class="border-primary ring-ring/50 block size-4 shrink-0 rounded-full border bg-white shadow-sm transition-[color,box-shadow] hover:ring-4 focus-visible:ring-4 focus-visible:outline-hidden" />
                </SliderRoot>
                <div class="mt-2 flex justify-between text-xs text-muted-foreground">
                  <span>1h</span>
                  <span>24h</span>
                </div>
              </div>

              <!-- Utilization -->
              <div>
                <div class="mb-3 flex items-center justify-between">
                  <label class="text-sm font-medium text-muted-foreground">Expected Utilization</label>
                  <span class="text-lg font-bold text-primary">{{ utilization[0] }}%</span>
                </div>
                <SliderRoot
                  v-model="utilization"
                  :min="10"
                  :max="100"
                  :step="5"
                  class="relative flex w-full touch-none items-center select-none"
                >
                  <SliderTrack class="bg-muted relative h-1.5 w-full grow overflow-hidden rounded-full">
                    <SliderRange class="bg-primary absolute h-full" />
                  </SliderTrack>
                  <SliderThumb class="border-primary ring-ring/50 block size-4 shrink-0 rounded-full border bg-white shadow-sm transition-[color,box-shadow] hover:ring-4 focus-visible:ring-4 focus-visible:outline-hidden" />
                </SliderRoot>
                <div class="mt-2 flex justify-between text-xs text-muted-foreground">
                  <span>10%</span>
                  <span>100%</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Earnings Display -->
          <div class="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <h3 class="mb-6 text-xl font-bold text-foreground">Your Potential Earnings</h3>

            <div class="space-y-6">
              <div class="rounded-xl border border-primary/20 bg-primary/10 p-6">
                <div class="mb-2 text-sm text-primary">Monthly Earnings</div>
                <div class="text-5xl font-bold text-foreground">€{{ monthlyEarnings.toFixed(0) }}</div>
                <div class="mt-2 text-sm text-muted-foreground">
                  Based on {{ hoursPerDay[0] }}h/day at {{ utilization[0] }}% utilization
                </div>
              </div>

              <div class="grid gap-4 sm:grid-cols-2">
                <div class="rounded-lg border border-border bg-background/50 p-4">
                  <div class="mb-1 text-sm text-muted-foreground">Daily</div>
                  <div class="text-2xl font-bold text-foreground">€{{ dailyEarnings.toFixed(2) }}</div>
                </div>
                <div class="rounded-lg border border-border bg-background/50 p-4">
                  <div class="mb-1 text-sm text-muted-foreground">Yearly</div>
                  <div class="text-2xl font-bold text-foreground">€{{ yearlyEarnings.toFixed(0) }}</div>
                </div>
              </div>

              <div class="space-y-3 rounded-lg border border-border bg-background/50 p-4">
                <div class="text-sm font-medium text-muted-foreground">Breakdown</div>
                <div class="flex justify-between text-sm">
                  <span class="text-muted-foreground">Hourly rate:</span>
                  <span class="text-foreground">€{{ hourlyRate.toFixed(2) }}/hr</span>
                </div>
                <div class="flex justify-between text-sm">
                  <span class="text-muted-foreground">Hours per month:</span>
                  <span class="text-foreground">{{ hoursPerDay[0] * 30 }}h</span>
                </div>
                <div class="flex justify-between text-sm">
                  <span class="text-muted-foreground">Utilization:</span>
                  <span class="text-foreground">{{ utilization[0] }}%</span>
                </div>
                <div class="border-t border-border pt-3">
                  <div class="flex justify-between text-sm">
                    <span class="text-muted-foreground">rbee commission (15%):</span>
                    <span class="text-foreground">-€{{ (monthlyEarnings * 0.15).toFixed(0) }}</span>
                  </div>
                  <div class="mt-2 flex justify-between font-medium">
                    <span class="text-foreground">Your take-home:</span>
                    <span class="text-primary">€{{ (monthlyEarnings * 0.85).toFixed(0) }}</span>
                  </div>
                </div>
              </div>

              <Button class="w-full bg-primary text-primary-foreground hover:bg-primary/90">Start Earning Now</Button>
            </div>
          </div>
        </div>

        <div class="mt-8 rounded-lg border border-border bg-card/50 p-6 text-center">
          <p class="text-sm text-muted-foreground">
            Earnings are estimates based on current market rates and may vary. Actual earnings depend on demand, your pricing, and availability.
          </p>
        </div>
      </div>
    </div>
  </section>
</template>
