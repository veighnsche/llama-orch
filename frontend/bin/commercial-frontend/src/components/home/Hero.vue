<script setup lang="ts">
  import { RouterLink } from 'vue-router'
  import { useI18n } from 'vue-i18n'
  const { t } = useI18n()
  import { Button, Badge } from '../../../../../libs/storybook/stories'
</script>

<template>
  <section class="hero" aria-labelledby="hero-title">
    <div class="container">
      <div class="copy">
        <p class="eyebrow">
          {{ t('home.hero.eyebrow', 'AI plumbing, done right') }}
        </p>

        <h1 id="hero-title" class="title">
          {{ t('home.hero.h1', 'Robust AI, without the leaks.') }}
        </h1>

        <p class="sub">
          {{
            t(
              'home.hero.sub',
              'I install and certify agentic AI pipelines. Proofs, logs, and documentation included. EU data, no lock-in.',
            )
          }}
        </p>

        <div class="ctas" role="group" aria-label="primary actions">
          <Button as="router-link" to="/service-menu" variant="primary">
            {{ t('home.hero.ctaMenu', 'View service menu') }}
          </Button>
          <Button as="router-link" to="/proofs" variant="ghost">
            {{ t('home.hero.ctaProofs', 'Inspect proofs') }}
          </Button>
          <div class="inline-links" aria-label="quick links">
            <RouterLink to="/public-tap">
              {{ t('home.hero.quickPublic', 'Public Tap') }}
            </RouterLink>
            <span aria-hidden="true">·</span>
            <RouterLink to="/private-tap">
              {{ t('home.hero.quickPrivate', 'Private Tap') }}
            </RouterLink>
          </div>
        </div>

        <ul class="badges" aria-label="proofs and certifications">
          <li>
            <Badge variant="green">
              <template #icon>
                <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                  <rect
                    x="3"
                    y="4"
                    width="18"
                    height="14"
                    rx="2"
                    fill="currentColor"
                    opacity=".12"
                  />
                  <path
                    d="M7 8h10M7 12h6M14.5 12l3.5 3.5"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                  />
                </svg>
              </template>
              {{ t('home.hero.badge.proofs', 'Proof-driven') }}
            </Badge>
          </li>
          <li>
            <Badge variant="cyan">
              <template #icon>
                <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                  <circle cx="12" cy="12" r="9" fill="currentColor" opacity=".12" />
                  <path
                    d="M8 12l2 2 6-6"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                  />
                </svg>
              </template>
              {{ t('home.hero.badge.eu', 'EU data') }}
            </Badge>
          </li>
          <li>
            <Badge variant="slate">
              <template #icon>
                <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                  <path
                    d="M12 2l3 3h4v4l3 3-3 3v4h-4l-3 3-3-3H5v-4l-3-3 3-3V5h4l3-3z"
                    fill="currentColor"
                    opacity=".12"
                  />
                  <path
                    d="M12 4.5l2.1 2.1h3V9.6l2.4 2.1-2.4 2.1v3.1h-3L12 19.4l-2.1-2.1H6.9v-3.1L4.5 11.7l2.4-2.1V6.6h3L12 4.5z"
                    fill="currentColor"
                  />
                </svg>
              </template>
              {{ t('home.hero.badge.oss', 'Open source') }}
            </Badge>
          </li>
          <li>
            <Badge variant="purple">
              <template #icon>
                <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                  <path
                    d="M12 2l7 4v6c0 5-7 10-7 10S5 17 5 12V6l7-4z"
                    fill="currentColor"
                    opacity=".12"
                  />
                  <circle cx="12" cy="10" r="2" fill="currentColor" />
                </svg>
              </template>
              {{ t('home.hero.badge.local', 'Local first') }}
            </Badge>
          </li>
        </ul>
      </div>

      <figure class="hero-media">
        <img
          src="/assets/hero_pipes.png"
          width="1440"
          height="560"
          alt="Blueprint diagram of clean data pipes feeding a dedicated tap; cyan/teal industrial accents."
          loading="eager"
          fetchpriority="high"
        />
      </figure>
    </div>
  </section>
</template>

<style scoped>
  :root {
    --hero-bg:
      radial-gradient(
        1100px 520px at 18% -8%,
        color-mix(in srgb, var(--acc-cyan) 25%, var(--surface-alt)) 0%,
        transparent 60%
      ),
      radial-gradient(
        760px 360px at 115% 45%,
        color-mix(in srgb, var(--acc-teal) 20%, var(--surface-alt)) 0%,
        transparent 55%
      );
    --ink: var(--text);
    --brand: var(--acc-cyan);
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
    grid-template-columns: 1fr; /* mobile */
  }
  @media (min-width: 920px) {
    .container {
      grid-template-columns: 1.1fr 1fr;
    }
  }

  .copy {
    max-width: 640px;
  }
  .eyebrow {
    display: inline-block;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--brand);
    background: color-mix(in srgb, var(--acc-cyan) 8%, transparent);
    border: 1px solid color-mix(in srgb, var(--acc-cyan) 20%, transparent);
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    margin: 0 0 0.75rem;
  }
  .title {
    color: var(--ink);
    line-height: 1.05;
    font-weight: 900;
    letter-spacing: -0.02em;
    font-size: clamp(2.1rem, 4.8vw, 3.5rem);
    margin: 0 0 0.6rem;
  }
  .sub {
    color: var(--muted);
    font-size: clamp(1rem, 1.35vw, 1.125rem);
    margin: 0 0 1rem;
  }

  .ctas {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin: 1rem 0 1rem;
    align-items: center;
  }

  .inline-links {
    display: flex;
    gap: 0.5rem;
    margin-left: 0.25rem;
  }
  .inline-links a {
    color: var(--muted);
    text-decoration: none;
    font-weight: 600;
  }
  .inline-links a:hover {
    text-decoration: underline;
  }

  .badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 0.75rem;
    list-style: none;
    padding: 0;
    margin: 0.25rem 0 0 0;
  }

  .hero-media {
    margin: 0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
  }
  .hero-media img {
    display: block;
    width: 100%;
    height: auto;
  }
</style>
