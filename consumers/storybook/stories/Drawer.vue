<script setup lang="ts">
  import { RouterLink, type RouteLocationRaw } from 'vue-router'
  type NavItem = {
    label: string
    to?: RouteLocationRaw
    href?: string
    ariaLabel?: string
  }
  const props = defineProps<{ open: boolean; items: NavItem[] }>()
  const emit = defineEmits<{ close: [] }>()
</script>

<template>
  <transition name="fade">
    <div v-if="props.open" class="drawer" @click.self="emit('close')">
      <div class="drawer-panel">
        <ul class="drawer-links">
          <li v-for="(item, idx) in props.items" :key="idx">
            <component :is="item.to ? RouterLink : 'a'" :to="item.to" :href="item.href" @click="emit('close')">
              {{ item.label }}
            </component>
          </li>
        </ul>
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

  /* show only on small; desktop drawer always hidden by shell layout */
  @media (min-width: 920px) {
    .drawer { display: none !important; }
  }

  .fade-enter-active,
  .fade-leave-active { transition: opacity 0.12s ease; }
  .fade-enter-from,
  .fade-leave-to { opacity: 0; }
</style>
