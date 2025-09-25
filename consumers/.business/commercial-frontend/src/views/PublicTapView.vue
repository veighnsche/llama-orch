<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'

const { t, tm, locale } = useI18n()
const route = useRoute()

useMeta({
  title: () => t('seoTitle.publicTap'),
  description: () => t('seoDesc.publicTap'),
  keywords: () => tm('seo.publicTap') as string[],
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
    <h1>{{ $t('publicTap.h1') }}</h1>

    <section>
      <h2>{{ $t('publicTap.what') }}</h2>
      <p>{{ $t('publicTap.whatP') }}</p>
    </section>

    <section>
      <h2>{{ $t('publicTap.who') }}</h2>
      <ul>
        <li>{{ $t('publicTap.bWho1') }}</li>
        <li>{{ $t('publicTap.bWho2') }}</li>
        <li>{{ $t('publicTap.bWho3') }}</li>
      </ul>
    </section>

    <section>
      <h2>{{ $t('publicTap.why') }}</h2>
      <ul>
        <li>{{ $t('publicTap.bWhy1') }}</li>
        <li>{{ $t('publicTap.bWhy2') }}</li>
        <li>{{ $t('publicTap.bWhy3') }}</li>
      </ul>
    </section>

    <section>
      <h2>{{ $t('publicTap.terms') }}</h2>
      <ul>
        <li>{{ $t('publicTap.bTerms1') }}</li>
        <li>{{ $t('publicTap.bTerms2') }}</li>
      </ul>
    </section>

    <section>
      <h2>{{ $t('publicTap.pricing') }}</h2>
      <ul>
        <li>{{ $t('publicTap.bPrice1') }}</li>
        <li>{{ $t('publicTap.bPrice2') }}</li>
        <li>{{ $t('publicTap.bPrice3') }}</li>
        <li>{{ $t('publicTap.bPrice4') }}</li>
      </ul>
      <p class="note">{{ $t('publicTap.note') }}</p>
    </section>
  </main>
</template>

<style scoped>
.page {
  display: grid;
  gap: 1rem;
}
.note { color: var(--muted); }
</style>
