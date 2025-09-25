<script setup lang="ts">
import { inject, onMounted, onBeforeUnmount } from 'vue'
import { RouterLink, type RouteLocationRaw } from 'vue-router'
import type { DrawerContext } from './DrawerRoot.vue'

const props = withDefaults(
  defineProps<{
    items?: Array<{ label: string; to?: RouteLocationRaw; href?: string; ariaLabel?: string }>
  }>(),
  { items: () => [] },
)

const ctx = inject<DrawerContext>('drawer-ctx')

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') {
    e.preventDefault()
    ctx?.setOpen(false)
  }
}

onMounted(() => {
  document.addEventListener('keydown', onKeydown)
})

onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKeydown)
})
</script>

<template>
  <transition name="fade">
    <div
      v-if="ctx?.open"
      class="drawer"
      :id="ctx?.id"
      :class="{ 'drawer--desktop-hidden': ctx?.options.hideOnDesktop }"
      @click.self="ctx?.setOpen(false)"
    >
      <div
        class="drawer-panel"
        :role="ctx?.options.ariaRole"
        :aria-modal="String(ctx?.options.ariaModal)"
        :aria-labelledby="ctx?.options.labelledBy"
        :aria-label="ctx?.options.label"
        tabindex="-1"
      >
        <template v-if="props.items && props.items.length">
          <ul class="drawer-links">
            <li v-for="(item, idx) in props.items" :key="idx">
              <component :is="item.to ? RouterLink : 'a'" :to="item.to" :href="item.href" @click="ctx?.setOpen(false)">
                {{ item.label }}
              </component>
            </li>
          </ul>
        </template>
        <slot />
        <div class="drawer-ops">
          <slot name="ops" />
        </div>
      </div>
    </div>
  </transition>
</template>

<style scoped>
.drawer {
  position: fixed;
  inset: 0;
  background: color-mix(in srgb, var(--text) 42%, transparent);
  z-index: 1000;
  display: block;
}
.drawer-panel {
  margin: 0.5rem;
  border-radius: var(--radius-xl);
  background: var(--surface-alt);
  border: 1px solid var(--surface-muted);
  box-shadow: var(--shadow-lg);
  padding: 0.75rem;
}
.drawer-links {
  list-style: none;
  margin: 0;
  padding: 0.25rem 0 0.5rem 0;
  display: grid;
  gap: 0.25rem;
}
.drawer-links a,
.drawer-links :global(a.router-link-active),
.drawer-links :global(a.router-link-exact-active) {
  display: block;
  padding: 0.6rem 0.6rem;
  border-radius: var(--radius-lg);
  text-decoration: none;
  color: var(--text);
  font-weight: 600;
}
.drawer-links a:hover,
.drawer-links :global(a.router-link-active):hover,
.drawer-links :global(a.router-link-exact-active):hover {
  background: var(--surface);
}
.drawer-ops {
  margin-top: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

@media (min-width: 920px) {
  .drawer--desktop-hidden { display: none !important; }
}

.fade-enter-active,
.fade-leave-active { transition: opacity 0.12s ease; }
.fade-enter-from,
.fade-leave-to { opacity: 0; }
</style>
