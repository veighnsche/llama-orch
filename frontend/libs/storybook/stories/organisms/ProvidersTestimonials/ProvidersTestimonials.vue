<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-007: Implemented ProvidersTestimonials component -->
<script setup lang="ts">
import { Star } from 'lucide-vue-next'
</script>

<template>
  <section class="border-b border-border bg-gradient-to-b from-background to-secondary px-6 py-24">
    <div class="mx-auto max-w-7xl">
      <div class="mb-16 text-center">
        <h2 class="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">What Providers Are Saying</h2>
        <p class="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
          Real earnings from real GPU providers on the rbee marketplace.
        </p>
      </div>

      <div class="mb-12 grid gap-8 md:grid-cols-3">
        <div class="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
          <div class="mb-4 flex gap-1">
            <Star v-for="i in 5" :key="i" :size="20" class="fill-primary text-primary" />
          </div>
          <p class="mb-6 text-pretty leading-relaxed text-muted-foreground">
            "My RTX 4090 was sitting idle 20 hours a day. Now it earns me €160/month. Literally passive income while I sleep. Setup took 10 minutes."
          </p>
          <div class="flex items-center gap-3">
            <div class="h-10 w-10 rounded-full bg-gradient-to-br from-primary to-accent" />
            <div>
              <div class="font-medium text-foreground">Marcus T.</div>
              <div class="text-sm text-muted-foreground">Gaming PC Owner • €160/mo</div>
            </div>
          </div>
        </div>

        <div class="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
          <div class="mb-4 flex gap-1">
            <Star v-for="i in 5" :key="i" :size="20" class="fill-primary text-primary" />
          </div>
          <p class="mb-6 text-pretty leading-relaxed text-muted-foreground">
            "I have 4 GPUs in my homelab. They were mostly idle. Now they earn €420/month combined. Pays for my entire homelab electricity plus profit."
          </p>
          <div class="flex items-center gap-3">
            <div class="h-10 w-10 rounded-full bg-gradient-to-br from-primary to-accent" />
            <div>
              <div class="font-medium text-foreground">Sarah K.</div>
              <div class="text-sm text-muted-foreground">Homelab Enthusiast • €420/mo</div>
            </div>
          </div>
        </div>

        <div class="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
          <div class="mb-4 flex gap-1">
            <Star v-for="i in 5" :key="i" :size="20" class="fill-primary text-primary" />
          </div>
          <p class="mb-6 text-pretty leading-relaxed text-muted-foreground">
            "After Ethereum went proof-of-stake, my mining rig was useless. Now it earns more with rbee than it ever did mining. Better margins, too."
          </p>
          <div class="flex items-center gap-3">
            <div class="h-10 w-10 rounded-full bg-gradient-to-br from-primary to-accent" />
            <div>
              <div class="font-medium text-foreground">David L.</div>
              <div class="text-sm text-muted-foreground">Former Miner • €780/mo</div>
            </div>
          </div>
        </div>
      </div>

      <div class="grid gap-8 md:grid-cols-4">
        <div class="text-center">
          <div class="mb-2 text-4xl font-bold text-primary">500+</div>
          <div class="text-sm text-muted-foreground">Active Providers</div>
        </div>
        <div class="text-center">
          <div class="mb-2 text-4xl font-bold text-primary">2,000+</div>
          <div class="text-sm text-muted-foreground">GPUs Earning</div>
        </div>
        <div class="text-center">
          <div class="mb-2 text-4xl font-bold text-primary">€180K+</div>
          <div class="text-sm text-muted-foreground">Paid to Providers</div>
        </div>
        <div class="text-center">
          <div class="mb-2 text-4xl font-bold text-primary">4.8/5</div>
          <div class="text-sm text-muted-foreground">Average Rating</div>
        </div>
      </div>
    </div>
  </section>
</template>
