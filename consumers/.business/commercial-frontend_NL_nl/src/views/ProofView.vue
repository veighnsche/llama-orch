<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'

const { t, tm, locale } = useI18n()
const route = useRoute()

useMeta({
  title: () => `${t('proof.h1')} | Orchyra`,
  description: () => t('seoDesc.proof'),
  keywords: () => tm('seo.proof') as string[],
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
    <h1>{{ $t('proof.h1') }}</h1>

    <section>
      <h2>{{ $t('proof.artifacts') }}</h2>
      <ul>
        <li>{{ $t('proof.b1') }}</li>
        <li>{{ $t('proof.b2') }}</li>
        <li>{{ $t('proof.b3') }}</li>
        <li>{{ $t('proof.b4') }}</li>
      </ul>
      <p class="microcopy">{{ $t('proof.microcopy') }}</p>
    </section>

    <section>
      <h2>{{ $t('proof.visuals') }}</h2>
      <ul>
        <li>{{ $t('proof.v1') }}</li>
        <li>{{ $t('proof.v2') }}</li>
      </ul>
    </section>
  </main>
</template>

<style scoped>
.page { display: grid; gap: 1rem; }
.microcopy { color: #374151; }
</style>
