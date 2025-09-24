<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'

const { t, tm, locale } = useI18n()
const route = useRoute()

useMeta({
  title: () => `${t('contact.h1')} | Orchyra`,
  description: () => t('seoDesc.contact'),
  keywords: () => tm('seo.contact') as string[],
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
    <h1>{{ $t('contact.h1') }}</h1>

    <section>
      <h2>{{ $t('contact.contact') }}</h2>
      <ul>
        <li>{{ $t('contact.email') }}: <a href="mailto:info@example.com">info@example.com</a></li>
        <li>{{ $t('contact.linkedin') }}: <a href="#" rel="nofollow noopener">{{ $t('a11y.addLink') }}</a></li>
        <li>{{ $t('contact.github') }}: <a href="#" rel="nofollow noopener">{{ $t('a11y.addLink') }}</a></li>
      </ul>
    </section>

    <section>
      <h2>{{ $t('contact.legal') }}</h2>
      <ul>
        <li>{{ $t('contact.l1') }}</li>
        <li>{{ $t('contact.l2') }}</li>
        <li>{{ $t('contact.l3') }}</li>
      </ul>
      <p class="note">{{ $t('contact.note') }}</p>
    </section>

    <section>
      <h2>{{ $t('contact.dataLogs') }}</h2>
      <p>
        {{ $t('contact.dataLogsP') }}
      </p>
    </section>
  </main>
</template>

<style scoped>
.page { display: grid; gap: 1rem; }
.note { color: #6b7280; }
</style>
