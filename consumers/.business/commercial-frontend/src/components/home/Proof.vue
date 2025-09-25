<script setup lang="ts">
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
</script>

<template>
  <section class="proof card section">
    <h3>{{ t('home.proof.title') }}</h3>
    <ul>
      <li>{{ t('home.proof.b1') }}</li>
      <li>{{ t('home.proof.b2') }}</li>
      <li>{{ t('home.proof.b3') }}</li>
      <li>{{ t('home.proof.b4') }}</li>
    </ul>
  </section>
</template>

<style scoped>
section { margin: 1.25rem 0; }
ul { padding-left: 1.1rem; }
</style>
