<script setup lang="ts">
  defineProps<{ navAriaLabel?: string }>()
</script>

<template>
  <header class="nav" role="banner">
    <nav class="nav-inner" :aria-label="navAriaLabel || 'Primary navigation'">
      <div class="slot-brand"><slot name="brand" /></div>
      <div class="slot-toggle"><slot name="toggle" /></div>
      <div class="slot-links"><slot name="links" /></div>
      <div class="right"><slot name="right" /></div>
    </nav>
    <slot name="drawer" />
  </header>
</template>

<style scoped>
  .nav {
    --nav-h: 56px; /* mobile/default height; larger than icon-only sm button (32px) */
    position: sticky;
    top: 0;
    z-index: 50;
    backdrop-filter: saturate(140%) blur(6px);
    background: var(--surface-alt);
    border-bottom: 1px solid var(--surface-muted);
  }
  .nav-inner {
    max-width: 1440px;
    margin: 0 auto;
    padding: 0.6rem 1rem;
    display: grid;
    grid-template-columns: auto 1fr auto; /* mobile-first: toggle | spacer | right */
    grid-template-areas: 'toggle . right';
    gap: 0.75rem 1rem;
    align-items: center;
    column-gap: 1.25rem;
    padding-block: 0;
    height: var(--nav-h); /* enforce fixed height on all breakpoints */
  }

  /* Grid areas for children */
  .slot-brand { grid-area: brand; display: none; }
  .slot-toggle { grid-area: toggle; }
  .slot-links { grid-area: links; display: none; }

  .right {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem; /* keep default tight spacing; switcher adds its own margin */
    justify-self: end;
    grid-area: right;
  }

  /* hide CTA button on small to keep bar tight */
  @media (max-width: 919.98px) {
    .right :deep(.ui-btn:not(.login-btn)),
    .right :deep(.lang-switcher),
    .right :deep(.theme-toggle) {
      display: none;
    }
  }

  /* Desktop fixed height + center alignment */
  @media (min-width: 920px) {
    .nav {
      --nav-h: 64px; /* give desktop a bit more breathing room */
    }
    .nav-inner {
      height: var(--nav-h);
      align-items: center;
      grid-template-columns: auto auto 1fr auto; /* brand | toggle | links | right */
      grid-template-areas: 'brand toggle links right';
    }
    /* Show brand and links on desktop */
    .slot-brand { display: block; }
    .slot-links { display: block; }
    /* Hide any menu toggle inside the shell on desktop */
    .nav-inner :deep(.menu-toggle) {
      display: none;
    }
  }
</style>
