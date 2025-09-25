<script setup lang="ts">
  import { ref, watch } from 'vue'
  import { useRoute, type RouteLocationRaw } from 'vue-router'
  import NavbarShell from './navbar-shell.vue'
  import NavbarBrand from './navbar-brand.vue'
  import NavbarToggle from './navbar-toggle.vue'
  import NavbarLinks from './navbar-links.vue'
  import NavbarDrawer from './navbar-drawer.vue'

  type Brand = {
    label: string
    to?: RouteLocationRaw
    href?: string
    ariaLabel?: string
    showGlyph?: boolean
  }

  type NavItem = {
    label: string
    to?: RouteLocationRaw
    href?: string
    ariaLabel?: string
  }

  const props = withDefaults(
    defineProps<{
      brand?: Brand
      links?: NavItem[]
      navAriaLabel?: string
      menuButtonLabels?: { open?: string; close?: string; srOnly?: string }
    }>(),
    {
      brand: () => ({ label: 'Brand', to: '/', ariaLabel: 'Home', showGlyph: true }),
      links: () => [],
      navAriaLabel: 'Primary navigation',
      menuButtonLabels: () => ({ open: 'Open menu', close: 'Close menu', srOnly: 'Menu' }),
    },
  )

  const open = ref(false)
  const route = useRoute()
  watch(
    () => route.fullPath,
    () => {
      open.value = false
    },
  )

</script>

<template>
  <NavbarShell :nav-aria-label="navAriaLabel">
    <template #brand>
      <slot name="brand" :brand="brand">
        <NavbarBrand :brand="brand" />
      </slot>
    </template>

    <template #toggle>
      <NavbarToggle :open="open" :labels="menuButtonLabels" @toggle="open = !open" />
    </template>

    <template #links>
      <NavbarLinks :items="links" />
    </template>

    <template #right>
      <slot name="right" />
    </template>

    <template #drawer>
      <NavbarDrawer :open="open" :items="links" @close="open = false">
        <template #ops>
          <slot name="drawer-ops" :close="() => (open = false)" />
        </template>
      </NavbarDrawer>
    </template>
  </NavbarShell>
</template>

