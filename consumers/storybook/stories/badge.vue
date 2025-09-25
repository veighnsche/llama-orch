<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    variant?: 'default' | 'green' | 'cyan' | 'slate' | 'purple'
    size?: 'sm' | 'md'
  }>(),
  {
    variant: 'default',
    size: 'md',
  },
)
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
  gap: 0.4rem;
  padding: 0.35rem 0.55rem;
  border-radius: var(--radius-pill);
  border: 1px solid var(--badge-border, var(--surface-muted));
  background: var(--badge-bg, var(--surface));
  color: var(--badge-fg, var(--text));
  font-size: 0.9rem;
  font-weight: 600;
}
.ui-badge--sm {
  padding: 0.25rem 0.45rem;
  font-size: 0.85rem;
}

/* default variant uses neutral chip */
.ui-badge--default {
  --badge-border: var(--surface-muted);
  --badge-bg: var(--surface);
  --badge-fg: var(--text);
}

/* Variant color tokens */
.ui-badge--green {
  --badge-color: var(--ok-green);
}
.ui-badge--cyan {
  --badge-color: var(--acc-cyan);
}
.ui-badge--slate {
  --badge-color: var(--muted);
}
.ui-badge--purple {
  --badge-color: var(--acc-purple);
}

/* Use color-mix to derive contrasting chip background */
.ui-badge--green,
.ui-badge--cyan,
.ui-badge--slate,
.ui-badge--purple {
  --badge-border: color-mix(in srgb, var(--badge-color) 42%, var(--surface-alt));
  --badge-bg: color-mix(in srgb, var(--badge-color) 18%, var(--surface-alt));
  --badge-fg: var(--text);
}
.ui-badge__icon {
  display: inline-grid;
  place-items: center;
  color: var(--badge-color);
}
.ui-badge__icon :where(svg) {
  width: 18px;
  height: 18px;
}

@media (prefers-color-scheme: dark) {
  .ui-badge--default {
    --badge-border: var(--surface-muted);
    --badge-bg: var(--surface);
    --badge-fg: var(--text);
  }
  .ui-badge--green,
  .ui-badge--cyan,
  .ui-badge--slate,
  .ui-badge--purple {
    --badge-border: color-mix(in srgb, var(--badge-color) 36%, var(--surface));
    --badge-bg: color-mix(in srgb, var(--badge-color) 20%, var(--surface));
    --badge-fg: var(--text);
  }
}
</style>
