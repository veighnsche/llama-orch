<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'
import { RouterLink, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'

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
    <section class="hero">
      <h1>{{ $t('home.hero.h1') }}</h1>
      <h2 class="tag">{{ $t('home.hero.h2') }}</h2>
      <p class="sub">{{ $t('home.hero.sub') }}</p>
      <div class="ctas">
        <RouterLink class="cta primary" to="/public-tap">{{ $t('home.hero.ctaPublic') }}</RouterLink>
        <RouterLink class="cta" to="/private-tap">{{ $t('home.hero.ctaPrivate') }}</RouterLink>
      </div>
      <ul class="trust">
        <li>{{ $t('home.hero.trust1') }}</li>
        <li>{{ $t('home.hero.trust2') }}</li>
        <li>{{ $t('home.hero.trust3') }}</li>
      </ul>
      <figure class="hero-media">
        <!-- Image placeholder — place your generated image at public/assets/hero_pipes.png -->
        <img
          src="/assets/hero_pipes.png"
          width="1440"
          height="560"
          alt="flat vector blueprint diagram of clean data pipes feeding a dedicated tap, subtle cyan/teal accents, sturdy industrial lines, customer-facing, no text"
          loading="eager"
          fetchpriority="high"
        />
      </figure>
    </section>

    <section class="why">
      <h3>{{ $t('home.why.title') }}</h3>
      <ul>
        <li>{{ $t('home.why.b1') }}</li>
        <li>{{ $t('home.why.b2') }}</li>
        <li>{{ $t('home.why.b3') }}</li>
      </ul>
    </section>

    <section class="three">
      <h3>{{ $t('home.three.title') }}</h3>
      <ul>
        <li>{{ $t('home.three.i1') }}</li>
        <li>{{ $t('home.three.i2') }}</li>
        <li>{{ $t('home.three.i3') }}</li>
      </ul>
    </section>

    <section class="public">
      <h3>{{ $t('home.public.title') }}</h3>
      <p>{{ $t('home.public.p1') }}</p>
      <ul>
        <li>{{ $t('home.public.b1') }}</li>
        <li>{{ $t('home.public.b2') }}</li>
        <li>{{ $t('home.public.b3') }}</li>
      </ul>
    </section>

    <section class="private">
      <h3>{{ $t('home.private.title') }}</h3>
      <p>{{ $t('home.private.p1') }}</p>
      <ul>
        <li>{{ $t('home.private.b1') }}</li>
        <li>{{ $t('home.private.b2') }}</li>
        <li>{{ $t('home.private.b3') }}</li>
      </ul>
    </section>

    <section class="proof">
      <h3>{{ $t('home.proof.title') }}</h3>
      <ul>
        <li>{{ $t('home.proof.b1') }}</li>
        <li>{{ $t('home.proof.b2') }}</li>
        <li>{{ $t('home.proof.b3') }}</li>
        <li>{{ $t('home.proof.b4') }}</li>
      </ul>
    </section>

    <section class="audience">
      <h3>{{ $t('home.audience.title') }}</h3>
      <ul>
        <li>{{ $t('home.audience.b1') }}</li>
        <li>{{ $t('home.audience.b2') }}</li>
        <li>{{ $t('home.audience.b3') }}</li>
      </ul>
    </section>

    <section class="more">
      <RouterLink to="/faqs">{{ $t('home.more.faqs') }}</RouterLink>
      ·
      <RouterLink to="/about">{{ $t('home.more.about') }}</RouterLink>
    </section>
  </main>
</template>

<style scoped>
.hero {
  display: grid;
  gap: 0.5rem;
  margin-bottom: 1.25rem;
}
.hero-media {
  margin: 0.5rem 0 0 0;
}
.hero-media img {
  display: block;
  width: 100%;
  height: auto;
  max-width: 100%;
  border-radius: 6px;
}
.tag {
  font-weight: 600;
}
.sub {
  color: #374151;
}
.ctas {
  display: flex;
  gap: 0.75rem;
  margin: 0.5rem 0 0.25rem 0;
}
.cta {
  text-decoration: none;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: 6px;
}
.cta.primary {
  background: #0ea5e9;
  color: white;
  border-color: #0ea5e9;
}
.trust {
  display: flex;
  gap: 1rem;
  list-style: none;
  padding: 0;
  margin: 0.25rem 0 0 0;
  color: #374151;
}
section {
  margin: 1.25rem 0;
}
ul {
  padding-left: 1.1rem;
}
</style>
