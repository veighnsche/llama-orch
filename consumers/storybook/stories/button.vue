<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink, type RouteLocationRaw } from 'vue-router'
import { useAttrs } from 'vue'

const props = withDefaults(defineProps<{
  as?: 'button' | 'a' | 'router-link'
  to?: RouteLocationRaw
  href?: string
  type?: 'button' | 'submit' | 'reset'
  variant?: 'primary' | 'ghost' | 'link'
  size?: 'sm' | 'md' | 'lg'
  iconOnly?: boolean
  disabled?: boolean
  block?: boolean
}>(), {
  as: undefined,
  type: 'button',
  variant: 'primary',
  size: 'md',
  iconOnly: false,
  disabled: false,
  block: false,
})

const attrsFromUser = useAttrs()

const isRouter = computed(() => props.as === 'router-link' || !!props.to)
const isAnchor = computed(() => props.as === 'a' || (!!props.href && !isRouter.value))

const tag = computed(() => (isRouter.value ? RouterLink : isAnchor.value ? 'a' : 'button'))

// Merge user attrs with our controlled attrs. Our controlled attrs take precedence.
const attrs = computed(() => {
  const a: Record<string, unknown> = { ...attrsFromUser }

  if (isRouter.value) {
    a.to = props.to
    // prevent navigating when "disabled"
    if (props.disabled) {
      a['aria-disabled'] = 'true'
      a.tabindex = -1
    }
  } else if (isAnchor.value) {
    a.href = props.href
    // Safe external targets
    if (a.target === '_blank') {
      const existingRel = (a.rel as string | undefined) || ''
      // Ensure noopener noreferrer present
      const relSet = new Set(existingRel.split(' ').filter(Boolean))
      relSet.add('noopener'); relSet.add('noreferrer')
      a.rel = Array.from(relSet).join(' ')
    }
    if (props.disabled) {
      // anchor elements don't support disabled
      a['aria-disabled'] = 'true'
      a.tabindex = -1
    }
  } else {
    a.type = props.type
    a.disabled = props.disabled
  }

  return a
})

const classes = computed(() => {
  const variantClass = props.variant === 'primary'
    ? 'ui-btn--primary'
    : props.variant === 'ghost'
    ? 'ui-btn--ghost'
    : 'ui-btn--link'

  return [
    'ui-btn',
    variantClass,
    `ui-btn--${props.size}`,
    props.iconOnly ? 'ui-btn--icon' : null,
    props.block ? 'ui-btn--block' : null,
    props.disabled ? 'is-disabled' : null,
  ]
})

// Block clicks / navigation when "disabled" on non-button tags.
function onClick(e: MouseEvent) {
  if (!props.disabled) return
  // For router-link / anchor, stop navigation if disabled
  if (isRouter.value || isAnchor.value) {
    e.preventDefault()
    e.stopImmediatePropagation?.()
    e.stopPropagation()
  }
}
</script>

<template>
  <component :is="tag" v-bind="attrs" :class="classes" @click="onClick">
    <slot />
  </component>
</template>

<style scoped>
.ui-btn {
  -webkit-tap-highlight-color: transparent;
  appearance: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: .4rem;
  border-radius: var(--radius-md);
  border: 1px solid var(--surface-muted);
  background: var(--surface-alt);
  color: var(--text);
  font-weight: 700;
  text-decoration: none;
  cursor: pointer;
  transition: transform .06s ease, box-shadow .06s ease, border-color .2s ease, background .2s ease, filter .2s ease;
}

.ui-btn:hover { transform: translateY(-1px); }
.ui-btn:focus-visible { outline: 2px solid var(--ring); outline-offset: 2px; }

/* Disabled (works for both <button disabled> and links using .is-disabled) */
.ui-btn:disabled,
.ui-btn.is-disabled {
  opacity: .6;
  cursor: not-allowed;
  transform: none;
  pointer-events: none; /* avoid accidental hover/active for non-button */
}

/* variants */
.ui-btn--primary {
  background: var(--acc-cyan);
  border-color: var(--acc-cyan);
  color: var(--surface-alt);
}
.ui-btn--primary:hover { filter: brightness(1.05); }

.ui-btn--ghost {
  background: var(--surface-alt);
  border-color: var(--surface-muted);
  color: var(--text);
}

.ui-btn--link {
  background: transparent;
  border-color: transparent;
  color: var(--acc-purple);
}
.ui-btn--link:hover {
  text-decoration: underline;
  transform: none;
}

/* layout helpers */
.ui-btn--block { width: 100%; display: inline-flex; }

/* sizes */
.ui-btn--sm { padding: .35rem .6rem; font-size: .9rem; }
.ui-btn--md { padding: .55rem .8rem; font-size: 1rem; }
.ui-btn--lg { padding: .7rem 1rem; font-size: 1.05rem; }

/* icon-only adjustments */
.ui-btn--icon { gap: 0; padding: 0; width: 38px; height: 38px; }
.ui-btn--icon.ui-btn--sm { width: 32px; height: 32px; }
.ui-btn--icon.ui-btn--md { width: 38px; height: 38px; }
.ui-btn--icon.ui-btn--lg { width: 44px; height: 44px; }

@media (prefers-color-scheme: dark) {
  .ui-btn--ghost {
    background: var(--surface);
    border-color: var(--surface-muted);
    color: var(--text);
  }
  .ui-btn--link { color: var(--acc-cyan); }
}
</style>
