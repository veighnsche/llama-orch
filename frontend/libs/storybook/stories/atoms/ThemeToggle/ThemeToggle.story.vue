<!-- Created by: TEAM-FE-010 -->
<script setup lang="ts">
import ThemeToggle from './ThemeToggle.vue'
</script>

<template>
  <Story title="atoms/ThemeToggle">
    <Variant title="Default (Icon)">
      <div class="p-8 bg-background">
        <ThemeToggle />
      </div>
    </Variant>

    <Variant title="Outline Variant">
      <div class="p-8 bg-background">
        <ThemeToggle variant="outline" />
      </div>
    </Variant>

    <Variant title="Default Variant">
      <div class="p-8 bg-background">
        <ThemeToggle variant="default" />
      </div>
    </Variant>

    <Variant title="Small Size">
      <div class="p-8 bg-background">
        <ThemeToggle size="icon-sm" />
      </div>
    </Variant>

    <Variant title="Large Size">
      <div class="p-8 bg-background">
        <ThemeToggle size="icon-lg" />
      </div>
    </Variant>

    <Variant title="In Navigation Context">
      <div class="min-h-screen bg-background">
        <nav class="fixed top-0 left-0 right-0 z-50 bg-background/95 backdrop-blur-sm border-b border-border">
          <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex items-center justify-between h-16">
              <div class="flex items-center gap-2">
                <span class="text-2xl">🐝</span>
                <span class="text-xl font-semibold text-primary">rbee</span>
              </div>
              <div class="flex items-center gap-4">
                <ThemeToggle />
                <button class="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium">
                  Join Waitlist
                </button>
              </div>
            </div>
          </div>
        </nav>
        <div class="pt-24 px-4">
          <div class="max-w-7xl mx-auto">
            <h1 class="text-3xl font-bold text-foreground mb-4">Theme Toggle Component</h1>
            <p class="text-muted-foreground">
              Click the theme toggle button in the navigation to switch between light and dark modes.
            </p>
            <div class="mt-8 p-6 bg-card border border-border rounded-lg">
              <h2 class="text-xl font-semibold text-card-foreground mb-2">Sample Card</h2>
              <p class="text-muted-foreground">
                This card demonstrates how the theme changes affect all design tokens.
              </p>
            </div>
          </div>
        </div>
      </div>
    </Variant>
  </Story>
</template>
