<script setup lang="ts">
  const props = withDefaults(
    defineProps<{
      open: boolean
      labels?: { open?: string; close?: string; srOnly?: string }
      ariaControls?: string
      size?: 'sm' | 'md' | 'lg'
    }>(),
    {
      labels: () => ({ open: 'Open menu', close: 'Close menu', srOnly: 'Menu' }),
      ariaControls: undefined,
      size: 'md',
    },
  )

  const emit = defineEmits<{
    toggle: []
    'update:open': [value: boolean]
  }>()

  function onClick() {
    emit('toggle')
    emit('update:open', !props.open)
  }
</script>

<template>
  <button
    class="menu-toggle"
    :class="[`menu-toggle--${props.size}`]"
    type="button"
    :aria-expanded="props.open ? 'true' : 'false'"
    :aria-label="props.open ? props.labels.close : props.labels.open"
    :aria-controls="props.ariaControls"
    @click="onClick"
  >
    <span class="sr-only">{{ props.labels.srOnly }}</span>
    <slot name="icon-closed" v-if="!props.open">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
      </svg>
    </slot>
    <slot name="icon-open" v-else>
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M6 6l12 12M18 6l-12 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
      </svg>
    </slot>
  </button>
</template>

<style scoped>
  .menu-toggle {
    appearance: none;
    border: 1px solid var(--surface-muted);
    background: var(--surface-alt);
    border-radius: var(--radius-lg);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.4rem 0.5rem;
    width: 38px;
    height: 38px;
    color: var(--text);
  }
  .menu-toggle--sm { width: 32px; height: 32px; }
  .menu-toggle--md { width: 38px; height: 38px; }
  .menu-toggle--lg { width: 44px; height: 44px; }
  .menu-toggle:focus-visible {
    outline: 4px solid var(--ring);
    outline-offset: 2px;
  }
  .menu-toggle svg {
    width: 22px;
    height: 22px;
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
