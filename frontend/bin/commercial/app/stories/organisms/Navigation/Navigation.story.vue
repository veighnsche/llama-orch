<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-CONSOLIDATE: Updated with interactive variants and design tokens -->
<script setup lang="ts">
import Navigation from './Navigation.vue'
</script>

<template>
  <Story title="organisms/Navigation">
    <Variant title="Default">
      <div class="min-h-screen bg-background">
        <Navigation />
        <div class="pt-24 px-4">
          <div class="max-w-7xl mx-auto">
            <h1 class="text-3xl font-bold text-foreground mb-4">Navigation Component</h1>
            <p class="text-muted-foreground">
              Fixed top navigation bar with logo, desktop menu, mobile hamburger menu, and CTA button.
            </p>
            <div class="mt-8 space-y-4 text-muted-foreground">
              <p>✅ Responsive design (desktop/mobile)</p>
              <p>✅ Mobile menu toggle</p>
              <p>✅ GitHub link</p>
              <p>✅ Join Waitlist CTA</p>
              <p>✅ Nuxt routing integration</p>
              <p>✅ Design tokens (no hardcoded colors)</p>
            </div>
          </div>
        </div>
      </div>
    </Variant>

    <Variant title="With Content">
      <div class="min-h-screen bg-background">
        <Navigation />
        <div class="pt-24 px-4 pb-16">
          <div class="max-w-7xl mx-auto space-y-8">
            <section>
              <h1 class="text-5xl font-bold text-foreground mb-4">Welcome to rbee</h1>
              <p class="text-xl text-muted-foreground">
                Private LLM orchestration for your infrastructure
              </p>
            </section>
            <section class="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div class="bg-card p-6 rounded-lg border border-border">
                <h3 class="text-xl font-semibold text-card-foreground mb-2">Feature 1</h3>
                <p class="text-muted-foreground">Sample content to show navigation with page content</p>
              </div>
              <div class="bg-card p-6 rounded-lg border border-border">
                <h3 class="text-xl font-semibold text-card-foreground mb-2">Feature 2</h3>
                <p class="text-muted-foreground">Sample content to show navigation with page content</p>
              </div>
              <div class="bg-card p-6 rounded-lg border border-border">
                <h3 class="text-xl font-semibold text-card-foreground mb-2">Feature 3</h3>
                <p class="text-muted-foreground">Sample content to show navigation with page content</p>
              </div>
            </section>
          </div>
        </div>
      </div>
    </Variant>
  </Story>
</template>
