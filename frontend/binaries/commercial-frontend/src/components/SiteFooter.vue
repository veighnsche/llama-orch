<template>
  <footer class="footer">
    <div class="footer-inner">
      <p class="brand">
        {{ $t('footer.brandLine') }}
      </p>
      <nav class="links" :aria-label="$t('a11y.footerNav')">
        <ul>
          <li>
            <RouterLink to="/about">
              {{ $t('footer.about') }}
            </RouterLink>
          </li>
          <li>
            <RouterLink to="/proof">
              {{ $t('footer.proof') }}
            </RouterLink>
          </li>
          <li>
            <RouterLink to="/pricing">
              {{ $t('footer.pricing') }}
            </RouterLink>
          </li>
          <li>
            <RouterLink to="/contact">
              {{ $t('footer.contactLegal') }}
            </RouterLink>
          </li>
        </ul>
      </nav>
      <small class="microcopy">{{ $t('footer.microcopy') }}</small>
    </div>
  </footer>
</template>

<script setup lang="ts">
  import { RouterLink } from 'vue-router'
</script>

<style scoped>
  .footer {
    border-top: 1px solid var(--surface-muted);
    padding: 1rem 0;
  }
  .footer-inner {
    display: grid;
    gap: 0.5rem;
  }
  .brand {
    font-weight: 600;
  }
  .links ul {
    display: flex;
    gap: 1rem;
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .links a {
    text-decoration: none;
  }
  .microcopy {
    color: var(--muted);
  }
</style>
