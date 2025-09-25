<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'

const { t, tm, locale } = useI18n()
const route = useRoute()

useMeta({
  title: () => t('seoTitle.privateTap'),
  description: () => t('seoDesc.privateTap'),
  keywords: () => tm('seo.privateTap') as string[],
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
    <h1>{{ $t('privateTap.h1') }}</h1>

    <section>
      <h2>{{ $t('privateTap.what') }}</h2>
      <p>{{ $t('privateTap.whatP') }}</p>
    </section>

    <section>
      <h2>{{ $t('privateTap.value') }}</h2>
      <ul>
        <li>{{ $t('privateTap.bVal1') }}</li>
        <li>{{ $t('privateTap.bVal2') }}</li>
        <li>{{ $t('privateTap.bVal3') }}</li>
      </ul>
    </section>

    <section>
      <h2>{{ $t('privateTap.pricing') }}</h2>
      <ul>
        <li>{{ $t('privateTap.bPrice1') }}</li>
        <li>{{ $t('privateTap.bPrice2') }}</li>
        <li>{{ $t('privateTap.bPrice3') }}</li>
      </ul>
      <p class="note">
        {{ $t('privateTap.note') }}
      </p>
    </section>
  </main>
</template>

<style scoped>
.page {
  display: grid;
  gap: 1rem;
}
.note {
  color: var(--muted);
}
</style>
