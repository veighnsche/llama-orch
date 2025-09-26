<script setup lang="ts">
  import { defineAsyncComponent } from 'vue'
  import { useMeta } from '@/composables/useMeta'
  import { useRoute } from 'vue-router'
  import { useI18n } from 'vue-i18n'

  // Above-the-fold: keep eager
  import DevUtilityBar from '@/components/DevUtilityBar.vue'
  import Hero from '@/components/home/Hero.vue'
  import WhyPlumbing from '@/components/home/WhyPlumbing.vue'

  // Below-the-fold: lazy for perf (simple code-splitting)
  const ThreeTools = defineAsyncComponent(() => import('@/components/home/ThreeTools.vue'))
  const PublicTap = defineAsyncComponent(() => import('@/components/home/PublicTap.vue'))
  const PrivateTap = defineAsyncComponent(() => import('@/components/home/PrivateTap.vue'))
  const Proof = defineAsyncComponent(() => import('@/components/home/Proof.vue'))
  const Audience = defineAsyncComponent(() => import('@/components/home/Audience.vue'))
  const More = defineAsyncComponent(() => import('@/components/home/More.vue'))

  const { t, tm, locale } = useI18n()
  const route = useRoute()

  // Prefer configured SITE URL; avoid direct window.* in SSR
  const siteUrl = import.meta.env.VITE_SITE_URL ?? ''
  const githubUrl = import.meta.env.VITE_GITHUB_URL || ''

  // Helper to build absolute URLs safely
  const abs = (p: string) => (siteUrl && p.startsWith('/') ? siteUrl + p : p)

  // Localized paths for hreflang (assumes /nl and /en routing; tweak if you use i18n routing differently)
  const currentPath = route.fullPath
  const altFor = (lang: string) =>
    siteUrl
      ? `${siteUrl}/${lang}${currentPath.startsWith('/') ? currentPath : `/${currentPath}`}`
      : currentPath

  useMeta({
    title: () => t('footer.brandLine'),
    description: () => t('seoDesc.home'),
    keywords: () => (tm('seo.home') as string[]) ?? [],
    canonical: () => (siteUrl ? `${siteUrl}${currentPath}` : currentPath),
    alternates: () => [
      { hrefLang: 'nl', href: altFor('nl') },
      { hrefLang: 'en', href: altFor('en') },
      { hrefLang: 'x-default', href: siteUrl ? `${siteUrl}${currentPath}` : currentPath },
    ],
    jsonLdId: 'ld-json-home',
    jsonLd: () => {
      const offers = [
        {
          '@type': 'Offer',
          name: t('publicTap.h1'),
          url: abs('/public-tap'),
          availability: 'https://schema.org/InStock',
          priceSpecification: {
            '@type': 'PriceSpecification',
            priceCurrency: 'EUR',
            price: '50',
          },
          description: t('publicTap.whatP'),
        },
        {
          '@type': 'Offer',
          name: t('privateTap.h1'),
          url: abs('/private-tap'),
          availability: 'https://schema.org/PreOrder',
          priceSpecification: {
            '@type': 'UnitPriceSpecification',
            priceCurrency: 'EUR',
            price: '1.80',
            unitCode: 'HUR', // per UNECE Rec 20 — per GPU hour
          },
          priceValidUntil: '2026-12-31',
          description: t('privateTap.whatP') ?? undefined,
        },
      ]

      const data: Record<string, unknown> = {
        '@context': 'https://schema.org',
        '@type': 'ProfessionalService',
        name: 'Orchyra',
        alternateName: t('footer.brandLine'),
        description: t('jsonld.desc'),
        url: siteUrl || undefined,
        inLanguage: locale.value,
        image: abs('/assets/hero_pipes.png'),
        areaServed: [
          { '@type': 'Place', name: 'Netherlands' },
          { '@type': 'Place', name: 'European Union' },
        ],
        serviceType: 'Private LLM hosting — design, deployment, and support',
        brand: { '@type': 'Brand', name: 'Orchyra' },
        offers,
        sameAs: githubUrl && /^https?:\/\//.test(githubUrl) ? [githubUrl] : undefined,
        // Add a stable @id for dedup in SERP
        '@id': siteUrl ? `${siteUrl}#orchyra-service` : undefined,
      }
      return data
    },
    watchSources: [() => locale.value, () => route.fullPath],
  })
</script>

<template>
  <main id="main" class="home">
    <DevUtilityBar />
    <Hero />
    <WhyPlumbing />
    <!-- Lazy chunks render naturally when reached; optional <Suspense> if you want placeholders -->
    <ThreeTools />
    <PublicTap />
    <PrivateTap />
    <Proof />
    <Audience />
    <More />
  </main>
</template>

<style scoped>
  /* page layout baseline */
  .home section {
    max-width: 1120px; /* zelfde breedte als nav/hero */
    margin-left: auto;
    margin-right: auto;
    padding: 2rem 1rem; /* ademruimte links/rechts */
  }

  /* meer adem tussen secties */
  .home section + section {
    margin-top: 2.5rem;
    border-top: 1px solid var(--surface-muted); /* optioneel scheiding */
    padding-top: 2.5rem;
  }

  /* headings consistenter */
  .home section h2,
  .home section h3 {
    font-weight: 800;
    font-size: 1.5rem;
    margin-bottom: 1rem;
    color: var(--text);
  }

  .home section.fullbleed {
    max-width: none;
    margin: 0;
    padding: 3rem 0;
    background: var(--surface); /* licht contrast */
  }

  /* fullbleed if needed */
  .home section.fullbleed {
    max-width: none;
    margin: 0;
    padding: 3rem 0;
    background: var(--surface); /* licht contrast */
  }
</style>
