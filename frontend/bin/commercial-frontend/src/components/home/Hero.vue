<script setup lang="ts">
  import { RouterLink } from 'vue-router'
  import { useI18n } from 'vue-i18n'
  const { t } = useI18n()
  import { Button, Badge } from 'orchyra-storybook/stories'
</script>

<template>
  <section class="hero" aria-labelledby="hero-title">
    <div class="container">
      <div class="copy">
        <p class="eyebrow">
          {{ t('home.hero.eyebrow', 'Open-source AI servers you can trust') }}
        </p>

        <h1 id="hero-title" class="title">
          {{ t('home.hero.h1', 'Transparent, predictable AI hosting.') }}
        </h1>

        <p class="sub">
          {{
            t(
              'home.hero.sub1',
              'Built on open-source foundations and designed for clarity. No lock-in, no hidden fees — hosting that grows with you.',
            )
          }}
        </p>

        <p class="sub">
          {{
            t(
              'home.hero.sub2',
              'Start small on shared servers with prepaid plans, then scale to dedicated GPU servers as your needs grow. Every step is cost-predictable and transparent.',
            )
          }}
        </p>

        <div class="ctas" role="group" aria-label="primary actions">
          <Button as="router-link" to="/service-menu" variant="primary">
            {{ t('home.hero.ctaMenu', 'Explore service plans') }}
          </Button>
          <Button as="router-link" to="/how-it-works" variant="ghost">
            {{ t('home.hero.ctaSecondary', 'How it works') }}
          </Button>
          <div class="inline-links" aria-label="quick links">
            <RouterLink to="/public-tap">
              {{ t('home.hero.quickPublic', 'Try shared servers') }}
            </RouterLink>
            <span aria-hidden="true">·</span>
            <RouterLink to="/private-tap">
              {{ t('home.hero.quickPrivate', 'Request dedicated demo') }}
            </RouterLink>
          </div>
        </div>
      </div>

      <figure class="hero-media">
        <img
          src="/assets/hero_pipes.png"
          width="980"
          height="420"
          alt="clean vector isometric data-center rack with GPU blades and a shield-check audit log icon; subtle cyan/teal accents; left-to-right flow arrows; minimal, professional; grid background; no literal pipes; wide banner crop"
          loading="eager"
          decoding="async"
          fetchpriority="high"
          sizes="(min-width: 920px) 520px, 100vw"
        />
      </figure>
    </div>
  </section>
</template>

<style scoped>
  :root {
    --hero-bg:
      radial-gradient(
        1000px 480px at 18% -8%,
        color-mix(in srgb, var(--acc-cyan) 22%, var(--surface-alt)) 0%,
        transparent 60%
      ),
      radial-gradient(
        680px 320px at 115% 45%,
        color-mix(in srgb, var(--acc-teal) 18%, var(--surface-alt)) 0%,
        transparent 55%
      );
    --ink: var(--text);
    --brand: var(--acc-cyan);
  }
  .hero {
    background: var(--hero-bg);
    padding: clamp(2rem, 4vw, 3.25rem) 1rem;
  }

  .container {
    max-width: 1120px;
    margin: 0 auto;
    display: grid;
    align-items: center;
    gap: 1.75rem;
    grid-template-columns: 1fr; /* mobile */
  }
  @media (min-width: 920px) {
    .container {
      grid-template-columns: minmax(420px, 1fr) minmax(360px, 520px);
      gap: 2rem;
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
    font-size: clamp(1.9rem, 4.2vw, 3.1rem);
    margin: 0 0 0.55rem;
  }
  .sub {
    color: var(--muted);
    font-size: clamp(1rem, 1.28vw, 1.1rem);
    margin: 0 0 1rem;
    max-width: 56ch;
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

  /* badges: switch to grid so they wrap cleanly without overflow */
  .badges {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, max-content));
    gap: 0.5rem 0.75rem;
    list-style: none;
    padding: 0;
    margin: 0.25rem 0 0 0;
    align-items: start;
  }
  .badges li {
    min-width: 0;
  }

  /* Very small devices: stack CTAs to avoid overflow */
  @media (max-width: 359px) {
    .ctas {
      flex-direction: column;
      align-items: stretch;
    }
    .ctas > :is(a, button) {
      width: 100%;
      justify-content: center;
    }
  }

  .hero-media {
    margin: 0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
    width: 100%;
    max-width: 520px;
    aspect-ratio: 16/9;
    justify-self: center;
  }
  .hero-media img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
</style>
