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
          <NavLinks :items="links" />
        </template>
        <template #right>
          <LanguageSwitcher />
          <ThemeSwitcher />
          <Button class="login-btn" as="router-link" to="/service-menu" variant="primary" size="sm">
            {{ $t('nav.login', 'Log in || Sign up') }}
          </Button>
        </template>
        <template #drawer>
          <DrawerPanel :items="links">
            <template #ops>
              <LanguageSwitcher />
              <ThemeSwitcher />
              <Button as="router-link" to="/service-menu" variant="primary">
                {{ $t('nav.serviceMenu', 'Service menu') }}
              </Button>
            </template>
          </DrawerPanel>
        </template>
      </NavbarShell>
      <!-- Mobile-only dev utility bar above the logo -->
      <DevUtilityBar class="mobile-devbar" :compact="true" :append-utm="true" />
      <!-- Mobile-only brand above content -->
      <div class="mobile-brand">
        <Brand :brand="brand" />
      </div>
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
  import {
    NavbarShell,
    Brand,
    NavLinks,
    Drawer,
    DrawerTrigger,
    DrawerPanel,
    Button,
  } from 'orchyra-storybook/stories'
  import LanguageSwitcher from '@/components/LanguageSwitcher.vue'
  import ThemeSwitcher from '@/components/ThemeSwitcher.vue'
  import SiteFooter from '@/components/SiteFooter.vue'
  import DevUtilityBar from '@/components/DevUtilityBar.vue'

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
  /* Mobile-only dev utility bar (shown above logo) */
  .mobile-devbar {
    display: block;
  }
  @media (min-width: 920px) {
    .mobile-devbar {
      display: none;
    }
  }
  /* Show brand above hero only on mobile; hide on desktop */
  .mobile-brand {
    display: flex;
    justify-content: center;
    padding: 0.6rem 1rem 0.25rem; /* align with navbar, add a bit more breathing room */
  }
  /* Make the brand bigger and centered in the mobile header band */
  .mobile-brand :deep(.brand) {
    gap: 0.6rem;
    padding: 0 0.5rem;
  }
  .mobile-brand :deep(.brand-glyph) {
    width: 40px;
    height: 40px;
  }
  .mobile-brand :deep(.brand-glyph) svg {
    width: 26px;
    height: 26px;
  }
  .mobile-brand :deep(.brand-word) {
    font-size: 1.28rem;
  }
  @media (min-width: 920px) {
    .mobile-brand {
      display: none;
    }
  }
</style>
