<script setup lang="ts">
  import { useMeta } from '@/composables/useMeta'
  import { useI18n } from 'vue-i18n'
  import { useRoute } from 'vue-router'

  const { t, tm, locale } = useI18n()
  const route = useRoute()

  useMeta({
    title: () => t('seoTitle.toolkit'),
    description: () => t('seoDesc.toolkit'),
    keywords: () => (tm('seo.toolkit') as string[]) ?? [],
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
    <h1>{{ $t('toolkit.h1') }}</h1>

    <section>
      <h2>{{ $t('toolkit.what') }}</h2>
      <p>{{ $t('toolkit.whatP') }}</p>
    </section>

    <section>
      <h2>{{ $t('toolkit.lang') }}</h2>
      <ul>
        <li>{{ $t('toolkit.bLang1') }}</li>
        <li>{{ $t('toolkit.bLang2') }}</li>
        <li>{{ $t('toolkit.bLang3') }}</li>
      </ul>
    </section>

    <section>
      <h2>{{ $t('toolkit.applets') }}</h2>
      <ul>
        <li>{{ $t('toolkit.bApp1') }}</li>
        <li>{{ $t('toolkit.bApp2') }}</li>
        <li>{{ $t('toolkit.bApp3') }}</li>
        <li>{{ $t('toolkit.bApp4') }}</li>
        <li>{{ $t('toolkit.bApp5') }}</li>
      </ul>
    </section>

    <section>
      <h2>{{ $t('toolkit.control') }}</h2>
      <ul>
        <li>{{ $t('toolkit.bCtl1') }}</li>
        <li>{{ $t('toolkit.bCtl2') }}</li>
        <li>{{ $t('toolkit.bCtl3') }}</li>
      </ul>
      <p class="note">{{ $t('toolkit.note') }}</p>
    </section>

    <p>
      <router-link to="/contact" class="cta">{{ $t('toolkit.cta') }}</router-link>
    </p>
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
  .cta {
    text-decoration: none;
    font-weight: 600;
  }
</style>
