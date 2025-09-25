<script setup lang="ts">
import { RouterLink } from 'vue-router'
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
</script>

<template>
  <section class="hero" aria-labelledby="hero-title">
    <div class="container">
      <!-- Left: copy -->
      <div class="copy">
        <p class="eyebrow">{{ t('home.hero.eyebrow', 'AI plumbing, done right') }}</p>
        <h1 id="hero-title" class="title">{{ t('home.hero.h1') }}</h1>
        <p class="sub">{{ t('home.hero.sub') }}</p>

        <div class="ctas" role="group" aria-label="primary actions">
          <RouterLink class="cta primary" to="/service-menu">
            {{ t('home.hero.ctaMenu', 'View service menu') }}
          </RouterLink>
          <RouterLink class="cta ghost" to="/proofs">
            {{ t('home.hero.ctaProofs', 'Inspect proofs') }}
          </RouterLink>
        </div>

        <!-- Proof badges (replaces trusted-by) -->
        <ul class="badges" aria-label="proofs and certifications">
          <li class="badge">
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2l3 3h4v4l3 3-3 3v4h-4l-3 3-3-3H5v-4l-3-3 3-3V5h4l3-3z" fill="currentColor" opacity=".12"/><path d="M12 4.5l2.1 2.1h2.95V9.55L19.5 11.7l-2.45 2.15V16.8H14.1L12 18.9l-2.1-2.1H6.95v-3L4.5 11.7 6.95 9.55V6.6H9.9L12 4.5z" fill="currentColor"/></svg>
            <span>{{ t('home.hero.badge.oss', 'Open source') }}</span>
          </li>
          <li class="badge">
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="currentColor" opacity=".12"/><path d="M8 12l2 2 6-6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
            <span>{{ t('home.hero.badge.gpl', 'GPL licensed') }}</span>
          </li>
          <li class="badge">
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="14" rx="2" fill="currentColor" opacity=".12"/><path d="M7 8h10M7 12h6M7 16h4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
            <span>{{ t('home.hero.badge.proofs', 'Proof-driven') }}</span>
          </li>
          <li class="badge">
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2l7 4v6c0 5-7 10-7 10S5 17 5 12V6l7-4z" fill="currentColor" opacity=".12"/><path d="M12 22s7-5 7-10V6l-7-4-7 4v6c0 5 7 10 7 10z" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="10" r="2" fill="currentColor"/></svg>
            <span>{{ t('home.hero.badge.local', 'Local first') }}</span>
          </li>
        </ul>
      </div>

      <!-- Right: media -->
      <figure class="hero-media">
        <img
          src="/assets/hero_pipes.png"
          width="1440"
          height="560"
          alt="Blueprint-style diagram of clean data pipes feeding a dedicated tap; industrial, cyan/teal accents."
          loading="eager"
          fetchpriority="high"
        />
      </figure>
    </div>
  </section>
</template>

<style scoped>
:root {
  --hero-bg: radial-gradient(1200px 600px at 20% -10%, #e0f2fe 0%, transparent 60%),
              radial-gradient(800px 400px at 120% 50%, #ccfbf1 0%, transparent 55%);
  --ink: #0f172a;         /* slate-900 */
  --muted: #475569;       /* slate-600 */
  --brand: #0ea5e9;       /* cyan-500 */
  --ring: rgba(14,165,233,.35);
  --border: #e5e7eb;      /* gray-200 */
  --chip: #f1f5f9;        /* slate-100 */
}

.hero {
  background: var(--hero-bg);
  padding: clamp(2rem, 4vw, 3.5rem) 1rem;
}

.container {
  max-width: 1120px;
  margin: 0 auto;
  display: grid;
  align-items: center;
  gap: 2rem;
}

.copy { max-width: 640px; }

.eyebrow {
  display: inline-block;
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: var(--brand);
  background: rgba(14,165,233,.08);
  border: 1px solid rgba(14,165,233,.2);
  padding: .25rem .6rem;
  border-radius: 999px;
  margin-bottom: .75rem;
}

.title {
  color: var(--ink);
  line-height: 1.05;
  font-weight: 900;
  font-size: clamp(2rem, 4.5vw, 3.25rem);
  letter-spacing: -.02em;
  margin: 0 0 .6rem 0;
}

.sub {
  color: var(--muted);
  font-size: clamp(1rem, 1.35vw, 1.125rem);
  margin: 0 0 1rem 0;
}

.ctas {
  display: flex;
  flex-wrap: wrap;
  gap: .75rem;
  margin: 1rem 0 1.25rem 0;
}

.cta {
  text-decoration: none;
  padding: .65rem .95rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  font-weight: 700;
  outline: none;
  transition: transform .06s ease, box-shadow .06s ease, border-color .2s ease, background .2s ease;
}

.cta:focus-visible { box-shadow: 0 0 0 4px var(--ring); }

.cta.primary {
  background: var(--brand);
  color: #fff;
  border-color: var(--brand);
}
.cta.primary:hover { transform: translateY(-1px); }

.cta.ghost {
  background: #fff;
  color: var(--ink);
}
.cta.ghost:hover { border-color: #cbd5e1; transform: translateY(-1px); }

.badges {
  display: flex;
  flex-wrap: wrap;
  gap: .5rem .75rem;
  list-style: none;
  padding: 0;
  margin: .25rem 0 0 0;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  padding: .35rem .55rem;
  background: var(--chip);
  border: 1px solid #e2e8f0; /* slate-200 */
  border-radius: 999px;
  color: #0f172a;
  font-size: .9rem;
  font-weight: 600;
}

.badge svg { color: #0ea5e9; }

.hero-media {
  margin: 0;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 12px 32px rgba(2, 6, 23, 0.12);
}
.hero-media img {
  display: block;
  width: 100%;
  height: auto;
}

@media (min-width: 920px) {
  .container { grid-template-columns: 1.1fr 1fr; }
}
</style>
