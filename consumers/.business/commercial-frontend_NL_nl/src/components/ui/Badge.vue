<script setup lang="ts">
const props = withDefaults(defineProps<{
  variant?: 'default' | 'green' | 'cyan' | 'slate' | 'purple'
  size?: 'sm' | 'md'
}>(), {
  variant: 'default',
  size: 'md',
})
</script>

<template>
  <span :class="['ui-badge', `ui-badge--${props.variant}`, `ui-badge--${props.size}`]">
    <span class="ui-badge__icon" aria-hidden="true">
      <slot name="icon" />
    </span>
    <span class="ui-badge__label">
      <slot />
    </span>
  </span>
</template>

<style scoped>
.ui-badge {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  padding: .35rem .55rem;
  border-radius: 999px;
  border: 1px solid var(--badge-border, #e2e8f0);
  background: var(--badge-bg, #f1f5f9);
  color: var(--badge-fg, #0f172a);
  font-size: .9rem;
  font-weight: 600;
}
.ui-badge--sm { padding: .25rem .45rem; font-size: .85rem; }

/* default variant uses neutral chip */
.ui-badge--default {
  --badge-border: #e2e8f0;
  --badge-bg: #eef2f7;
  --badge-fg: #0f172a;
}

/* Variant color tokens */
.ui-badge--green { --badge-color: var(--ok-green, #22c55e); }
.ui-badge--cyan  { --badge-color: var(--acc-cyan,  #22d3ee); }
.ui-badge--slate { --badge-color: #64748b; }
.ui-badge--purple{ --badge-color: var(--acc-purple, #7c5cff); }

/* Use color-mix to derive contrasting chip background */
.ui-badge--green,
.ui-badge--cyan,
.ui-badge--slate,
.ui-badge--purple {
  --badge-border: color-mix(in srgb, var(--badge-color) 42%, #ffffff);
  --badge-bg:     color-mix(in srgb, var(--badge-color) 18%, #ffffff);
  --badge-fg:     #0f172a;
}
.ui-badge__icon { display: inline-grid; place-items: center; color: var(--badge-color); }
.ui-badge__icon :where(svg) { width: 18px; height: 18px; }

@media (prefers-color-scheme: dark) {
  .ui-badge--default {
    --badge-border: #0b1b2e;
    --badge-bg: #0f172a;
    --badge-fg: #e5f1ff;
  }
  .ui-badge--green,
  .ui-badge--cyan,
  .ui-badge--slate,
  .ui-badge--purple {
    --badge-border: color-mix(in srgb, var(--badge-color) 36%, #0b1220);
    --badge-bg:     color-mix(in srgb, var(--badge-color) 20%, #0b1220);
    --badge-fg:     #e5f1ff;
  }
}
</style>
