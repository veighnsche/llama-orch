<script setup lang="ts">
  defineProps<{ navAriaLabel?: string }>()
</script>

<template>
  <header class="nav" role="banner">
    <nav class="nav-inner" :aria-label="navAriaLabel || 'Primary navigation'">
      <slot name="brand" />
      <slot name="toggle" />
      <slot name="links" />
      <div class="right">
        <slot name="right" />
      </div>
    </nav>
    <slot name="drawer" />
  </header>
</template>

<style scoped>
  :root {
    --nav-h: 44px; /* mobile default */
  }

  .nav {
    position: sticky;
    top: 0;
    z-index: 50;
    backdrop-filter: saturate(140%) blur(6px);
    background: var(--surface-alt);
    border-bottom: 1px solid var(--surface-muted);
  }
  .nav-inner {
    max-width: 1120px;
    margin: 0 auto;
    padding: 0.6rem 1rem;
    display: grid;
    grid-template-columns: auto auto 1fr auto; /* brand | toggle | links | right */
    gap: 0.75rem 1rem;
    align-items: center;
    column-gap: 1.25rem;
    padding-block: 0;
  }

  .right {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    justify-self: end;
  }

  /* hide CTA button on small to keep bar tight */
  @media (max-width: 919.98px) {
    .right :deep(.ui-btn) {
      display: none;
    }
  }

  /* Desktop fixed height + center alignment */
  @media (min-width: 920px) {
    :root {
      --nav-h: 56px;
    }
    .nav-inner {
      height: var(--nav-h);
      align-items: center;
    }
    /* Hide any menu toggle inside the shell on desktop */
    .nav-inner :deep(.menu-toggle) {
      display: none;
    }
  }
</style>
