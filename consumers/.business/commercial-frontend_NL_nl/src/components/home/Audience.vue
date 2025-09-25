<script setup lang="ts">
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
</script>

<template>
  <section class="audience card section">
    <h3>{{ t('home.audience.title') }}</h3>
    <ul>
      <li>{{ t('home.audience.b1') }}</li>
      <li>{{ t('home.audience.b2') }}</li>
      <li>{{ t('home.audience.b3') }}</li>
    </ul>
  </section>
</template>

<style scoped>
section { margin: 1.25rem 0; }
ul { padding-left: 1.1rem; }
</style>
