<script setup lang="ts">
  import { useI18n } from 'vue-i18n'

  const { t } = useI18n()
</script>

<template>
  <section class="public card section">
    <h3>{{ t('home.public.title') }}</h3>
    <p>{{ t('home.public.p1') }}</p>
    <ul>
      <li>{{ t('home.public.b1') }}</li>
      <li>{{ t('home.public.b2') }}</li>
      <li>{{ t('home.public.b3') }}</li>
    </ul>
  </section>
</template>

<style scoped>
  section {
    margin: 1.25rem 0;
  }
  ul {
    padding-left: 1.1rem;
  }
</style>
