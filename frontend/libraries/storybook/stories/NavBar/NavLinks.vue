<script setup lang="ts">
  import { RouterLink, type RouteLocationRaw } from 'vue-router'

  type NavItem = {
    label: string
    to?: RouteLocationRaw
    href?: string
    ariaLabel?: string
  }

  defineProps<{ items: NavItem[] }>()
</script>

<template>
  <ul class="links" role="menubar">
    <li v-for="(item, idx) in items" :key="idx" role="none">
      <component
        :is="item.to ? RouterLink : 'a'"
        role="menuitem"
        :to="item.to"
        :href="item.href"
        :aria-label="item.ariaLabel || item.label"
      >
        {{ item.label }}
      </component>
    </li>
  </ul>
</template>

<style scoped>
  .links {
    display: none; /* mobile-first: hidden */
    align-items: center;
    gap: 1rem;
    list-style: none;
    margin: 0;
    padding: 0;
    min-width: 0; /* allow shrinking inside grid track */
  }
  .links a,
  .links :global(a.router-link-active),
  .links :global(a.router-link-exact-active) {
    display: inline-flex;
    align-items: center;
    height: var(--nav-h);
    padding: 0 0.5rem;
    border-radius: var(--radius-lg);
    color: var(--muted);
    font-weight: 600;
    text-decoration: none;
  }
  .links a:hover,
  .links :global(a.router-link-active):hover,
  .links :global(a.router-link-exact-active):hover {
    background: var(--surface);
  }

  /* desktop: show links */
  @media (min-width: 920px) {
    .links {
      display: inline-flex;
    }
  }
</style>
