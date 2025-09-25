<script setup lang="ts">
  import { useMeta } from '@/composables/useMeta'
  import { useI18n } from 'vue-i18n'
  import { useRoute } from 'vue-router'

  const { t, tm, locale } = useI18n()
  const route = useRoute()

  useMeta({
    title: () => t('seoTitle.pricing'),
    description: () => t('seoDesc.pricing'),
    keywords: () => tm('seo.pricing') as string[],
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
    <h1>{{ $t('pricing.h1') }}</h1>

    <section>
      <h2>{{ $t('pricing.credits') }}</h2>
      <ul>
        <li>{{ $t('publicTap.bPrice1') }}</li>
        <li>{{ $t('publicTap.bPrice2') }}</li>
        <li>{{ $t('publicTap.bPrice3') }}</li>
        <li>{{ $t('publicTap.bPrice4') }}</li>
      </ul>
      <p class="note">
        {{ $t('pricing.note1') }}
      </p>
    </section>

    <section>
      <h2>{{ $t('pricing.gpu') }}</h2>
      <ul>
        <li>{{ $t('privateTap.bPrice1') }}</li>
        <li>{{ $t('privateTap.bPrice2') }}</li>
        <li>{{ $t('privateTap.bPrice3') }}</li>
      </ul>
      <p class="note">
        {{ $t('pricing.note2') }}
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
