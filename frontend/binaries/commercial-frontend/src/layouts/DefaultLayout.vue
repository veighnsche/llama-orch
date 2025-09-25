<template>
  <div class="layout">
    <NavBar :brand="brand" :links="links" :nav-aria-label="$t('a11y.navPrimary')">
      <template #right>
        <LanguageSwitcher />
        <Button as="router-link" to="/service-menu" variant="primary" size="sm">
          {{ $t('nav.serviceMenu', 'Service menu') }}
        </Button>
      </template>
      <template #drawer-ops>
        <LanguageSwitcher />
        <Button as="router-link" to="/service-menu" variant="primary">
          {{ $t('nav.serviceMenu', 'Service menu') }}
        </Button>
      </template>
    </NavBar>
    <main class="content bp-grid">
      <RouterView />
    </main>
    <SiteFooter />
  </div>
</template>

<script setup lang="ts">
  import { RouterView } from 'vue-router'
  import { computed } from 'vue'
  import { useI18n } from 'vue-i18n'
  import NavBar from 'orchyra-storybook/stories/navbar.vue'
  import Button from 'orchyra-storybook/stories/button.vue'
  import LanguageSwitcher from '@/components/LanguageSwitcher.vue'
  import SiteFooter from '@/components/SiteFooter.vue'

  const { t } = useI18n()

  const brand = computed(() => ({
    label: 'Orchyra',
    to: '/',
    ariaLabel: 'Orchyra — home',
    showGlyph: true,
  }))

  const links = computed(() => [
    { label: t('nav.publicTap'), to: '/public-tap' },
    { label: t('nav.privateTap'), to: '/private-tap' },
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
    padding: 1rem 0;
  }
</style>
