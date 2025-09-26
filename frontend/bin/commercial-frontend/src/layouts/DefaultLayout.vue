<template>
  <div class="layout">
    <Drawer :id="drawerId" v-model="open" :hide-on-desktop="false">
      <NavbarShell :nav-aria-label="$t('a11y.navPrimary')">
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
            :aria-controls="drawerId"
          >
            <svg v-if="!open" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
            </svg>
            <svg v-else viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6 6l12 12M18 6l-12 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
            </svg>
          </DrawerTrigger>
        </template>
        <template #links>
          <NavLinks :items="links" />
        </template>
        <template #right>
          <LanguageSwitcher />
          <Button as="router-link" to="/service-menu" variant="primary" size="sm">
            {{ $t('nav.serviceMenu', 'Service menu') }}
          </Button>
        </template>
        <template #drawer>
          <DrawerPanel :items="links">
            <template #ops>
              <LanguageSwitcher />
              <Button as="router-link" to="/service-menu" variant="primary">
                {{ $t('nav.serviceMenu', 'Service menu') }}
              </Button>
            </template>
          </DrawerPanel>
        </template>
      </NavbarShell>
    </Drawer>
    <main class="content bp-grid">
      <RouterView />
    </main>
    <SiteFooter />
  </div>
</template>

<script setup lang="ts">
  import { RouterView } from 'vue-router'
  import { computed, ref } from 'vue'
  import { useI18n } from 'vue-i18n'
  import { NavbarShell, Brand, NavLinks, Drawer, DrawerTrigger, DrawerPanel, Button } from 'orchyra-storybook/stories'
  import LanguageSwitcher from '@/components/LanguageSwitcher.vue'
  import SiteFooter from '@/components/SiteFooter.vue'

  const { t } = useI18n()

  const open = ref(false)
  const drawerId = 'nav-drawer-app'

  const brand = computed(() => ({
    label: 'Orchyra',
    to: '/',
    ariaLabel: 'Orchyra — home',
    showGlyph: true,
  }))

  const links = computed(() => [
    { label: t('nav.publicTap'), to: '/public-tap' },
    { label: t('nav.privateTap'), to: '/private-tap' },
    { label: t('nav.toolkit'), to: '/toolkit' },
    { label: t('nav.pricing'), to: '/pricing' },
    { label: t('nav.proof'), to: '/proof' },
    { label: t('nav.faqs'), to: '/faqs' },
    { label: t('nav.about'), to: '/about' },
    { label: t('nav.contact'), to: '/contact' },
  ])
</script>

<style scoped>
  .layout {
    display: flex;
    min-height: 100vh;
    flex-direction: column;
  }
  .content {
    flex: 1 1 auto;
  }
</style>
