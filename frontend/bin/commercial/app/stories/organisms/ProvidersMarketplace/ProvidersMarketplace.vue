<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-007: Implemented ProvidersMarketplace component -->
<script setup lang="ts">
import { TrendingUp, Users, Globe, Shield } from 'lucide-vue-next'

interface Props {
  title?: string
  subtitle?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'How the rbee Marketplace Works',
  subtitle: 'A fair, transparent marketplace that connects GPU providers with developers who need compute power.'
})
</script>

<template>
  <section class="border-b border-border bg-gradient-to-b from-background to-secondary px-6 py-24">
    <div class="mx-auto max-w-7xl">
      <div class="mb-16 text-center">
        <h2 class="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">
          {{ title }}
        </h2>
        <p class="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
          {{ subtitle }}
        </p>
      </div>

      <div class="mb-16 grid gap-8 md:grid-cols-2 lg:grid-cols-4">
        <div class="text-center">
          <div class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
            <TrendingUp :size="32" class="text-primary" />
          </div>
          <h3 class="mb-2 text-lg font-bold text-foreground">Dynamic Pricing</h3>
          <p class="text-pretty text-sm leading-relaxed text-muted-foreground">
            Set your own rates or use automatic pricing based on market demand
          </p>
        </div>

        <div class="text-center">
          <div class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
            <Users :size="32" class="text-primary" />
          </div>
          <h3 class="mb-2 text-lg font-bold text-foreground">Growing Demand</h3>
          <p class="text-pretty text-sm leading-relaxed text-muted-foreground">
            Thousands of developers need GPU power for AI development
          </p>
        </div>

        <div class="text-center">
          <div class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
            <Globe :size="32" class="text-primary" />
          </div>
          <h3 class="mb-2 text-lg font-bold text-foreground">Global Reach</h3>
          <p class="text-pretty text-sm leading-relaxed text-muted-foreground">
            Your GPUs are discoverable by developers worldwide
          </p>
        </div>

        <div class="text-center">
          <div class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
            <Shield :size="32" class="text-primary" />
          </div>
          <h3 class="mb-2 text-lg font-bold text-foreground">Fair Commission</h3>
          <p class="text-pretty text-sm leading-relaxed text-muted-foreground">
            Only 15% commission. You keep 85% of all earnings
          </p>
        </div>
      </div>

      <div class="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-12">
        <div class="grid gap-12 lg:grid-cols-2">
          <div>
            <h3 class="mb-6 text-2xl font-bold text-foreground">Marketplace Features</h3>
            <div class="space-y-4">
              <div class="flex gap-4">
                <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
                  <div class="h-2 w-2 rounded-full bg-primary" />
                </div>
                <div>
                  <div class="mb-1 font-medium text-foreground">Automatic Matching</div>
                  <div class="text-sm leading-relaxed text-muted-foreground">
                    Our algorithm matches your GPUs with the right jobs based on requirements and your pricing.
                  </div>
                </div>
              </div>

              <div class="flex gap-4">
                <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
                  <div class="h-2 w-2 rounded-full bg-primary" />
                </div>
                <div>
                  <div class="mb-1 font-medium text-foreground">Rating System</div>
                  <div class="text-sm leading-relaxed text-muted-foreground">
                    Build your reputation with customer ratings. Higher ratings = more jobs and better rates.
                  </div>
                </div>
              </div>

              <div class="flex gap-4">
                <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
                  <div class="h-2 w-2 rounded-full bg-primary" />
                </div>
                <div>
                  <div class="mb-1 font-medium text-foreground">Guaranteed Payments</div>
                  <div class="text-sm leading-relaxed text-muted-foreground">
                    Customers pay upfront. You're guaranteed to get paid for every job completed.
                  </div>
                </div>
              </div>

              <div class="flex gap-4">
                <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
                  <div class="h-2 w-2 rounded-full bg-primary" />
                </div>
                <div>
                  <div class="mb-1 font-medium text-foreground">Dispute Resolution</div>
                  <div class="text-sm leading-relaxed text-muted-foreground">
                    Fair dispute resolution process protects both providers and customers.
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h3 class="mb-6 text-2xl font-bold text-foreground">Commission Structure</h3>
            <div class="space-y-4">
              <div class="rounded-lg border border-border bg-background/50 p-6">
                <div class="mb-4 flex items-center justify-between">
                  <div class="text-sm text-muted-foreground">Standard Commission</div>
                  <div class="text-2xl font-bold text-primary">15%</div>
                </div>
                <div class="text-sm leading-relaxed text-muted-foreground">
                  We take 15% to cover marketplace operations, payment processing, and customer support.
                </div>
              </div>

              <div class="rounded-lg border border-primary/20 bg-primary/10 p-6">
                <div class="mb-4 flex items-center justify-between">
                  <div class="text-sm text-primary">You Keep</div>
                  <div class="text-2xl font-bold text-primary">85%</div>
                </div>
                <div class="text-sm leading-relaxed text-primary">
                  You keep 85% of all earnings. No hidden fees. No surprise deductions.
                </div>
              </div>

              <div class="space-y-2 text-sm">
                <div class="flex justify-between">
                  <span class="text-muted-foreground">Example: €100 job</span>
                  <span class="text-foreground">€100.00</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-muted-foreground">rbee commission (15%)</span>
                  <span class="text-foreground">-€15.00</span>
                </div>
                <div class="border-t border-border pt-2">
                  <div class="flex justify-between font-medium">
                    <span class="text-foreground">Your earnings</span>
                    <span class="text-primary">€85.00</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
