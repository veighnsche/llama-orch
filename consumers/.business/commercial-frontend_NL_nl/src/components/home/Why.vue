<script setup lang="ts">
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
</script>

<template>
  <section class="why">
    <h3>{{ t('home.why.title') }}</h3>
    <ul>
      <li>{{ t('home.why.b1') }}</li>
      <li>{{ t('home.why.b2') }}</li>
      <li>{{ t('home.why.b3') }}</li>
    </ul>
  </section>
</template>

<style scoped>
section { margin: 1.25rem 0; }
ul { padding-left: 1.1rem; }
</style>
