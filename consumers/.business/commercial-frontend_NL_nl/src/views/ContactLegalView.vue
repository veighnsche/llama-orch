<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'

const { t, tm, locale } = useI18n()
const route = useRoute()

const email = import.meta.env.VITE_CONTACT_EMAIL || 'info@example.com'
const linkedin = import.meta.env.VITE_LINKEDIN_URL || '#'
const github = import.meta.env.VITE_GITHUB_URL || '#'

useMeta({
  title: () => t('seoTitle.contact'),
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
        <li>{{ $t('contact.email') }}: <a :href="`mailto:${email}`">{{ email }}</a></li>
        <li>{{ $t('contact.linkedin') }}: <a :href="linkedin" rel="nofollow noopener">{{ linkedin }}</a></li>
        <li>{{ $t('contact.github') }}: <a :href="github" rel="nofollow noopener">{{ github }}</a></li>
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
