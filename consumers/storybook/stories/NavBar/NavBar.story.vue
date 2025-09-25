<script setup lang="ts">
  import { reactive, computed, ref, watch } from 'vue'
  import { useRoute } from 'vue-router'
  import Button from '../Button/Button.vue'
  import NavBarShell from './NavBarShell.vue'
  import Brand from '../Brand/Brand.vue'
  import NavLinks from './NavLinks.vue'
  import Drawer from '../Drawer/Drawer.vue'
  import DrawerTrigger from '../Drawer/DrawerTrigger.vue'
  import DrawerPanel from '../Drawer/DrawerPanel.vue'

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

  // Separate open states and ids for each variant to avoid interference
  const openPlayground = ref(false)
  const openComposable = ref(false)
  const drawerIdPlayground = 'nav-drawer-playground'
  const drawerIdComposable = 'nav-drawer-composable'
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
  <Story title="Composite/Navbar" :layout="{ type: 'single', iframe: false, width: 980 }">
    <Variant title="Playground">
      <div
        style="
          max-width: 980px;
          border: 1px solid var(--surface-muted);
          border-radius: var(--radius-md);
          overflow: hidden;
        "
      >
        <Drawer :id="drawerIdPlayground" v-model="openPlayground" :hide-on-desktop="false">
          <NavBarShell :nav-aria-label="state.navLabel">
            <template #brand>
              <Brand :brand="brand" />
            </template>
            <template #toggle>
              <DrawerTrigger
                class="menu-toggle"
                :as="Button"
                variant="ghost"
                size="sm"
                icon-only
                :aria-controls="drawerIdPlayground"
              >
                <svg v-if="!openPlayground" viewBox="0 0 24 24" aria-hidden="true">
                  <path
                    d="M4 7h16M4 12h16M4 17h16"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                  />
                </svg>
                <svg v-else viewBox="0 0 24 24" aria-hidden="true">
                  <path
                    d="M6 6l12 12M18 6l-12 12"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                  />
                </svg>
              </DrawerTrigger>
            </template>
            <template #links>
              <NavLinks :items="state.links" />
            </template>
            <template #right>
              <Button as="router-link" :to="'/contact'" size="sm" variant="primary">
                Contact us
              </Button>
            </template>
            <template #drawer>
              <DrawerPanel :items="state.links">
                <template #ops>
                  <Button as="router-link" :to="'/contact'" variant="primary"> Contact us </Button>
                </template>
              </DrawerPanel>
            </template>
          </NavBarShell>
        </Drawer>
      </div>
    </Variant>

    <Variant title="Minimal (brand only)">
      <NavBarShell :nav-aria-label="state.navLabel">
        <template #brand>
          <Brand :brand="{ label: 'Brand', to: '/', showGlyph: false }" />
        </template>
      </NavBarShell>
    </Variant>

    <Variant title="External links">
      <Drawer :id="drawerIdComposable" v-model="openComposable" :hide-on-desktop="false">
        <NavBarShell :nav-aria-label="state.navLabel">
          <template #brand>
            <Brand :brand="{ label: 'Brand', to: '/' }" />
          </template>
          <template #toggle>
            <DrawerTrigger
              class="menu-toggle"
              :as="Button"
              variant="ghost"
              size="sm"
              icon-only
              :aria-controls="drawerIdComposable"
            >
              <svg v-if="!openComposable" viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M4 7h16M4 12h16M4 17h16"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                />
              </svg>
              <svg v-else viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M6 6l12 12M18 6l-12 12"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                />
              </svg>
            </DrawerTrigger>
          </template>
          <template #links>
            <NavLinks
              :items="[
                { label: 'Docs', href: 'https://example.com' },
                { label: 'About', to: '/about' },
              ]"
            />
          </template>
          <template #drawer>
            <DrawerPanel
              :items="[
                { label: 'Docs', href: 'https://example.com' },
                { label: 'About', to: '/about' },
              ]"
            />
          </template>
        </NavBarShell>
      </Drawer>
    </Variant>

    <Variant title="Full">
      <Drawer :id="drawerIdComposable" v-model="openComposable" :hide-on-desktop="false">
        <NavBarShell :nav-aria-label="state.navLabel">
          <template #brand>
            <Brand :brand="brand" />
          </template>
          <template #toggle>
            <DrawerTrigger
              class="menu-toggle"
              :as="Button"
              variant="ghost"
              size="sm"
              icon-only
              :aria-controls="drawerIdComposable"
            >
              <svg v-if="!openComposable" viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M4 7h16M4 12h16M4 17h16"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                />
              </svg>
              <svg v-else viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M6 6l12 12M18 6l-12 12"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                />
              </svg>
            </DrawerTrigger>
          </template>
          <template #links>
            <NavLinks :items="state.links" />
          </template>
          <template #right>
            <Button as="router-link" :to="'/contact'" size="sm" variant="primary">
              Contact us
            </Button>
          </template>
          <template #drawer>
            <DrawerPanel :items="state.links">
              <template #ops>
                <Button as="router-link" :to="'/contact'" variant="primary"> Contact us </Button>
              </template>
            </DrawerPanel>
          </template>
        </NavBarShell>
      </Drawer>
    </Variant>

    <template #controls>
      <HstText v-model="state.brandLabel" title="Brand label" />
      <HstCheckbox v-model="state.showGlyph" title="Show glyph" />
      <HstText v-model="state.navLabel" title="ARIA label" />
      <HstJson v-model="state.links" title="Links (array of {label, to|href})" />
    </template>
  </Story>
</template>
