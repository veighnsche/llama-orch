<script setup lang="ts">
  const props = withDefaults(
    defineProps<{
      open: boolean
      labels?: { open?: string; close?: string; srOnly?: string }
    }>(),
    {
      labels: () => ({ open: 'Open menu', close: 'Close menu', srOnly: 'Menu' }),
    },
  )

  const emit = defineEmits<{ toggle: [] }>()
</script>

<template>
  <button
    class="menu-toggle"
    :aria-expanded="props.open ? 'true' : 'false'"
    :aria-label="props.open ? props.labels.close : props.labels.open"
    @click="emit('toggle')"
  >
    <span class="sr-only">{{ props.labels.srOnly }}</span>
    <svg v-if="!props.open" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
    </svg>
    <svg v-else viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6 6l12 12M18 6l-12 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
    </svg>
  </button>
</template>

<style scoped>
  .menu-toggle {
    appearance: none;
    border: 1px solid var(--surface-muted);
    background: var(--surface-alt);
    border-radius: var(--radius-lg);
    padding: 0.4rem 0.5rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    color: var(--text);
  }
  .menu-toggle:focus-visible {
    outline: 4px solid var(--ring);
    outline-offset: 2px;
  }
  .menu-toggle svg {
    width: 22px;
    height: 22px;
  }

  /* desktop: hide hamburger */
  @media (min-width: 920px) {
    .menu-toggle {
      display: none;
    }
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
</style>
