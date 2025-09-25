<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'
import { useI18n } from 'vue-i18n'
import { useRoute, RouterLink } from 'vue-router'

const { t, tm, locale } = useI18n()
const route = useRoute()

useMeta({
  title: () => t('seoTitle.about'),
  description: () => t('seoDesc.about'),
  keywords: () => tm('seo.about') as string[],
  canonical: () => `${window.location.origin}${route.fullPath}`,
  alternates: () => {
    const href = `${window.location.origin}${route.fullPath}`
    return [
      { hrefLang: 'nl', href },
      { hrefLang: 'en', href },
      { hrefLang: 'x-default', href },
    ]
  },
  watchSources: [() => locale.value, () => route.fullPath],
})
</script>

<template>
  <main class="page">
    <h1>{{ $t('about.h1') }}</h1>

    <section>
      <h2>{{ $t('about.identity') }}</h2>
      <p>{{ $t('about.identityP') }}</p>
    </section>

    <section>
      <h2>{{ $t('about.usp') }}</h2>
      <ul>
        <li>{{ $t('about.usp1') }}</li>
        <li>{{ $t('about.usp2') }}</li>
        <li>{{ $t('about.usp3') }}</li>
        <li>{{ $t('about.usp4') }}</li>
      </ul>
    </section>

    <section>
      <h2>{{ $t('about.approach') }}</h2>
      <p>{{ $t('about.approachP') }}</p>
    </section>

    <section>
      <RouterLink class="cta primary" to="/contact">{{ $t('about.cta') }}</RouterLink>
    </section>
  </main>
</template>

<style scoped>
.page { display: grid; gap: 1rem; }
.cta.primary {
  text-decoration: none;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--acc-cyan);
  border-radius: 6px;
  background: var(--acc-cyan);
  color: white;
}
</style>
