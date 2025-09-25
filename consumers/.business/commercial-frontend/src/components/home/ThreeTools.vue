<script setup lang="ts">
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
</script>

<template>
  <section class="three card section">
    <h3>{{ t('home.three.title') }}</h3>
    <ul>
      <li>{{ t('home.three.i1') }}</li>
      <li>{{ t('home.three.i2') }}</li>
      <li>{{ t('home.three.i3') }}</li>
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
