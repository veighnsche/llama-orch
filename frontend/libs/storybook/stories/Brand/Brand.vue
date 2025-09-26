<script setup lang="ts">
  import { computed } from 'vue'
  import { RouterLink, type RouteLocationRaw } from 'vue-router'

  type Brand = {
    label: string
    to?: RouteLocationRaw
    href?: string
    ariaLabel?: string
    showGlyph?: boolean
  }

  const props = defineProps<{ brand: Brand }>()

  const isRouter = computed(() => !!props.brand.to)
  const isAnchor = computed(() => !!props.brand.href && !props.brand.to)
  const tag = computed(() => (isRouter.value ? RouterLink : isAnchor.value ? 'a' : RouterLink))
</script>

<template>
  <component
    :is="tag"
    class="brand"
    :to="brand.to || '/'"
    :href="brand.href"
    :aria-label="brand.ariaLabel || brand.label"
  >
    <span v-if="brand.showGlyph" class="brand-glyph" aria-hidden="true">
      <svg viewBox="0 0 24 24" focusable="false" aria-hidden="true">
        <circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="2" />
        <path
          d="M12 6v12M6 12h12M8.7 8.7l6.6 6.6M15.3 8.7l-6.6 6.6"
          stroke="currentColor"
          stroke-width="1.6"
          stroke-linecap="round"
        />
        <path d="M2 12h3M19 12h3" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
        <circle cx="12" cy="12" r="2.2" fill="currentColor" />
      </svg>
    </span>
    <span class="brand-word">{{ brand.label }}</span>
  </component>
</template>

<style scoped>
  .brand {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0 1rem;
    border-radius: var(--radius-lg);
    text-decoration: none;
    color: var(--text);
    font-weight: 800;
    letter-spacing: 0.2px;
  }
  .brand:hover {
    text-decoration: none;
  }
  .brand-glyph {
    width: 28px;
    height: 28px;
    display: grid;
    place-items: center;
    border-radius: var(--radius-md);
    background: var(--surface);
    border: 1px solid var(--surface-muted);
    color: var(--acc-cyan);
  }
  .brand-glyph svg {
    width: 18px;
    height: 18px;
  }
  .brand-word {
    font-size: 1.06rem;
  }
</style>
