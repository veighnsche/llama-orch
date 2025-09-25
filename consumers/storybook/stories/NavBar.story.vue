<script setup lang="ts">
import { reactive, computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import Button from './Button.vue'
import NavBarShell from './NavBarShell.vue'
import Brand from './Brand.vue'
import MenuToggle from './MenuToggle.vue'
import NavLinks from './NavLinks.vue'
import Drawer from './Drawer.vue'

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

// Separate open states for each variant to avoid interference
const openPlayground = ref(false)
const openComposable = ref(false)
const route = useRoute()
watch(
  () => route.fullPath,
  () => {
    openPlayground.value = false
    openComposable.value = false
  },
)
</script>

<template>
  <Story title="UI/Navbar" :layout="{ type: 'single', iframe: false }">
    <Variant title="Playground">
      <div>
        <NavBarShell :nav-aria-label="state.navLabel">
          <template #brand>
            <Brand :brand="brand" />
          </template>
          <template #toggle>
            <MenuToggle :open="openPlayground" @toggle="openPlayground = !openPlayground" />
          </template>
          <template #links>
            <NavLinks :items="state.links" />
          </template>
          <template #right>
            <Button as="router-link" :to="'/contact'" size="sm" variant="primary">Contact us</Button>
          </template>
          <template #drawer>
            <Drawer :open="openPlayground" :items="state.links" @close="openPlayground = false">
              <template #ops>
                <Button as="router-link" :to="'/contact'" variant="primary">Contact us</Button>
              </template>
            </Drawer>
          </template>
        </NavBarShell>
      </div>
    </Variant>

    <Variant title="Minimal (brand only)">
      <NavBarShell :nav-aria-label="state.navLabel">
        <template #brand>
          <Brand :brand="{ label: 'Brand', to: '/', showGlyph: false }" />
        </template>
      </NavBarShell>
    </Variant>

    <Variant title="With external links">
      <NavBarShell :nav-aria-label="state.navLabel">
        <template #brand>
          <Brand :brand="{ label: 'Brand', to: '/' }" />
        </template>
        <template #toggle>
          <MenuToggle :open="openComposable" @toggle="openComposable = !openComposable" />
        </template>
        <template #links>
          <NavLinks :items="[{ label: 'Docs', href: 'https://example.com' }, { label: 'About', to: '/about' }]" />
        </template>
        <template #drawer>
          <Drawer :open="openComposable" :items="[{ label: 'Docs', href: 'https://example.com' }, { label: 'About', to: '/about' }]" @close="openComposable = false" />
        </template>
      </NavBarShell>
    </Variant>

    <Variant title="Composable (shell + parts)">
      <NavBarShell :nav-aria-label="state.navLabel">
        <template #brand>
          <Brand :brand="brand" />
        </template>
        <template #toggle>
          <MenuToggle :open="openComposable" @toggle="openComposable = !openComposable" />
        </template>
        <template #links>
          <NavLinks :items="state.links" />
        </template>
        <template #right>
          <Button as="router-link" :to="'/contact'" size="sm" variant="primary">Contact us</Button>
        </template>
        <template #drawer>
          <Drawer :open="openComposable" :items="state.links" @close="openComposable = false">
            <template #ops>
              <Button as="router-link" :to="'/contact'" variant="primary">Contact us</Button>
            </template>
          </Drawer>
        </template>
      </NavBarShell>
    </Variant>

    <template #controls>
      <HstText v-model="state.brandLabel" title="Brand label" />
      <HstCheckbox v-model="state.showGlyph" title="Show glyph" />
      <HstText v-model="state.navLabel" title="ARIA label" />
      <HstJson v-model="state.links" title="Links (array of {label, to|href})" />
    </template>
  </Story>
</template>
