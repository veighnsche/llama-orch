<template>
  <header class="nav">
    <nav class="nav-inner" aria-label="Primary">
      <RouterLink class="brand" to="/">Orchyra</RouterLink>
      <ul class="links">
        <li><RouterLink to="/public-tap">{{ $t('nav.publicTap') }}</RouterLink></li>
        <li><RouterLink to="/private-tap">{{ $t('nav.privateTap') }}</RouterLink></li>
        <li><RouterLink to="/pricing">{{ $t('nav.pricing') }}</RouterLink></li>
        <li><RouterLink to="/proof">{{ $t('nav.proof') }}</RouterLink></li>
        <li><RouterLink to="/faqs">{{ $t('nav.faqs') }}</RouterLink></li>
        <li><RouterLink to="/about">{{ $t('nav.about') }}</RouterLink></li>
        <li><RouterLink to="/contact">{{ $t('nav.contact') }}</RouterLink></li>
      </ul>
      <LanguageSwitcher />
    </nav>
  </header>
  
</template>

<script setup lang="ts">
import { RouterLink } from 'vue-router'
import LanguageSwitcher from '@/components/LanguageSwitcher.vue'
</script>

<style scoped>
.nav {
  border-bottom: 1px solid var(--color-border, #e5e7eb);
}
.nav-inner {
  display: flex;
  align-items: center;
  gap: 1rem;
}
.brand {
  font-weight: 700;
  text-decoration: none;
}
.links {
  list-style: none;
  display: flex;
  gap: 1rem;
  margin: 0;
  padding: 0;
}
.links a {
  text-decoration: none;
}
</style>
