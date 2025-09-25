<script setup lang="ts">
  import { reactive, computed } from 'vue'
  import Navbar from './navbar.vue'
  import Button from './button.vue'
  import NavbarShell from './navbar-shell.vue'
  import NavbarBrand from './navbar-brand.vue'
  import NavbarToggle from './navbar-toggle.vue'
  import NavbarLinks from './navbar-links.vue'
  import NavbarDrawer from './navbar-drawer.vue'

  const state = reactive({
    brandLabel: 'Orchyra',
    showGlyph: true,
    navLabel: 'Primary navigation',
    links: [
      { label: 'Home', to: '/' },
      { label: 'About', to: '/about' },
      { label: 'Contact', to: '/contact' },
    ] as Array<{ label: string; to?: string; href?: string }>,
  })

  const brand = computed(() => ({
    label: state.brandLabel,
    to: '/',
    ariaLabel: `${state.brandLabel} — home`,
    showGlyph: state.showGlyph,
  }))
</script>

<template>
  <Story title="UI/Navbar" :layout="{ type: 'single', iframe: false }">
    <Variant title="Playground">
      <div>
        <Navbar :brand="brand" :links="state.links" :nav-aria-label="state.navLabel">
          <template #right>
            <Button as="router-link" :to="'/contact'" size="sm" variant="primary">Contact us</Button>
          </template>
          <template #drawer-ops>
            <Button as="router-link" :to="'/contact'" variant="primary">Contact us</Button>
          </template>
        </Navbar>
      </div>
    </Variant>

    <Variant title="Minimal (brand only)">
      <Navbar :brand="{ label: 'Brand', to: '/' }" />
    </Variant>

    <Variant title="With external links">
      <Navbar
        :brand="{ label: 'Brand', to: '/' }"
        :links="[
          { label: 'Docs', href: 'https://example.com' },
          { label: 'About', to: '/about' },
        ]"
      />
    </Variant>

    <Variant title="Composable (shell + parts)">
      <ComposableNavbar :brand="brand" :links="state.links" />
    </Variant>

    <template #controls>
      <HstText v-model="state.brandLabel" title="Brand label" />
      <HstCheckbox v-model="state.showGlyph" title="Show glyph" />
      <HstText v-model="state.navLabel" title="ARIA label" />
      <HstJson v-model="state.links" title="Links (array of {label, to|href})" />
    </template>
  </Story>
</template>

<script lang="ts">
  import { defineComponent, ref, watch } from 'vue'
  import { useRoute } from 'vue-router'
  export default defineComponent({
    name: 'ComposableNavbarWrapper',
    components: { NavbarShell, NavbarBrand, NavbarToggle, NavbarLinks, NavbarDrawer, Button },
    props: {
      brand: { type: Object, required: true },
      links: { type: Array, required: true },
    },
    setup() {
      const open = ref(false)
      const route = useRoute()
      watch(
        () => route.fullPath,
        () => {
          open.value = false
        },
      )
      return { open }
    },
    template: `
      <NavbarShell nav-aria-label="Primary navigation">
        <template #brand>
          <NavbarBrand :brand="brand" />
        </template>
        <template #toggle>
          <NavbarToggle :open="open" @toggle="open = !open" />
        </template>
        <template #links>
          <NavbarLinks :items="links" />
        </template>
        <template #right>
          <Button as="router-link" to="/contact" size="sm" variant="primary">Contact us</Button>
        </template>
        <template #drawer>
          <NavbarDrawer :open="open" :items="links" @close="open = false">
            <template #ops>
              <Button as="router-link" to="/contact" variant="primary">Contact us</Button>
            </template>
          </NavbarDrawer>
        </template>
      </NavbarShell>
    `,
  })
</script>
