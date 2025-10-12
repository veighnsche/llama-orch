<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-CONSOLIDATE: Ported from reference/v0 navigation component with design tokens -->
<!-- TEAM-FE-010: Added ThemeToggle button from shared storybook -->
<script setup lang="ts">
import { ref } from 'vue'
import { Button, ThemeToggle } from 'rbee-storybook/stories'
import { Menu, X, Github } from 'lucide-vue-next'

const mobileMenuOpen = ref(false)

const toggleMobileMenu = () => {
  mobileMenuOpen.value = !mobileMenuOpen.value
}

const closeMobileMenu = () => {
  mobileMenuOpen.value = false
}
</script>

<template>
  <nav class="fixed top-0 left-0 right-0 z-50 bg-background/95 backdrop-blur-sm border-b border-border">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <!-- Logo -->
        <NuxtLink
          to="/"
          class="flex items-center gap-2 text-xl font-semibold text-primary hover:text-primary/80 transition-colors"
        >
          <span class="text-2xl">🐝</span>
          <span>rbee</span>
        </NuxtLink>

        <!-- Desktop Navigation -->
        <div class="hidden md:flex items-center gap-8">
          <NuxtLink to="/features" class="text-muted-foreground hover:text-foreground transition-colors">
            Features
          </NuxtLink>
          <NuxtLink to="/use-cases" class="text-muted-foreground hover:text-foreground transition-colors">
            Use Cases
          </NuxtLink>
          <NuxtLink to="/pricing" class="text-muted-foreground hover:text-foreground transition-colors">
            Pricing
          </NuxtLink>
          <NuxtLink to="/developers" class="text-muted-foreground hover:text-foreground transition-colors">
            For Developers
          </NuxtLink>
          <NuxtLink to="/providers" class="text-muted-foreground hover:text-foreground transition-colors">
            For Providers
          </NuxtLink>
          <NuxtLink to="/enterprise" class="text-muted-foreground hover:text-foreground transition-colors">
            For Enterprise
          </NuxtLink>

          <a
            href="https://github.com/veighnsche/llama-orch"
            target="_blank"
            rel="noopener noreferrer"
            class="text-muted-foreground hover:text-foreground transition-colors"
          >
            <Github :size="20" />
          </a>

          <ThemeToggle />

          <Button size="sm" variant="default">
            Join Waitlist
          </Button>
        </div>

        <!-- Mobile Menu Button -->
        <button
          @click="toggleMobileMenu"
          class="md:hidden text-muted-foreground hover:text-foreground"
        >
          <X v-if="mobileMenuOpen" :size="24" />
          <Menu v-else :size="24" />
        </button>
      </div>

      <!-- Mobile Menu -->
      <div v-if="mobileMenuOpen" class="md:hidden py-4 space-y-4 border-t border-border">
        <NuxtLink
          to="/features"
          class="block text-muted-foreground hover:text-foreground transition-colors"
          @click="closeMobileMenu"
        >
          Features
        </NuxtLink>
        <NuxtLink
          to="/use-cases"
          class="block text-muted-foreground hover:text-foreground transition-colors"
          @click="closeMobileMenu"
        >
          Use Cases
        </NuxtLink>
        <NuxtLink
          to="/pricing"
          class="block text-muted-foreground hover:text-foreground transition-colors"
          @click="closeMobileMenu"
        >
          Pricing
        </NuxtLink>
        <NuxtLink
          to="/developers"
          class="block text-muted-foreground hover:text-foreground transition-colors"
          @click="closeMobileMenu"
        >
          For Developers
        </NuxtLink>
        <NuxtLink
          to="/providers"
          class="block text-muted-foreground hover:text-foreground transition-colors"
          @click="closeMobileMenu"
        >
          For Providers
        </NuxtLink>
        <NuxtLink
          to="/enterprise"
          class="block text-muted-foreground hover:text-foreground transition-colors"
          @click="closeMobileMenu"
        >
          For Enterprise
        </NuxtLink>
        <a
          href="https://github.com/veighnsche/llama-orch"
          target="_blank"
          rel="noopener noreferrer"
          class="block text-muted-foreground hover:text-foreground transition-colors"
        >
          GitHub
        </a>
        <div class="flex items-center gap-2">
          <ThemeToggle />
          <span class="text-sm text-muted-foreground">Toggle theme</span>
        </div>
        <Button size="sm" variant="default" class="w-full">
          Join Waitlist
        </Button>
      </div>
    </div>
  </nav>
</template>
