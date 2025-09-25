<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import Hero from '@/components/home/Hero.vue'
import Why from '@/components/home/Why.vue'
import Three from '@/components/home/Three.vue'
import Public from '@/components/home/PublicTap.vue'
import Private from '@/components/home/PrivateTap.vue'
import Proof from '@/components/home/Proof.vue'
import Audience from '@/components/home/Audience.vue'
import More from '@/components/home/More.vue'

const { t, tm, locale } = useI18n()
const route = useRoute()

// Configure production URLs via Vite env; fall back to current origin/placeholders
const siteUrl = import.meta.env.VITE_SITE_URL || window.location.origin
const githubUrl = import.meta.env.VITE_GITHUB_URL || ''

useMeta({
  title: () => `Orchyra — ${t('home.hero.h2')}`,
  description: () => t('seoDesc.home'),
  keywords: () => tm('seo.home') as string[],
  jsonLdId: 'ld-json-home',
  jsonLd: () => {
    const data: Record<string, unknown> = {
      '@context': 'https://schema.org',
      '@type': 'ProfessionalService',
      name: 'Orchyra',
      alternateName: t('home.hero.h2'),
      description: t('home.hero.sub'),
      areaServed: 'NL, EU',
      url: siteUrl,
      offers: [
        {
          '@type': 'Offer',
          name: t('publicTap.h1'),
          priceCurrency: 'EUR',
          price: '50',
          description: t('publicTap.whatP'),
        },
        {
          '@type': 'Offer',
          name: t('privateTap.h1'),
          priceCurrency: 'EUR',
          price: '1.80',
          unitText: t('a11y.perGpuHour'),
          priceValidUntil: '2026-12-31',
        },
      ],
    }
    if (githubUrl && /^https?:\/\//.test(githubUrl)) {
      ;(data as any).sameAs = [githubUrl]
    }
    return data
  },
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
  <main class="home">
    <Hero />
    <Why />
    <Three />
    <Public />
    <Private />
    <Proof />
    <Audience />
    <More />
  </main>
</template>

