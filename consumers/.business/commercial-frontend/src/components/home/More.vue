<script setup lang="ts">
import { RouterLink } from 'vue-router'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
</script>

<template>
  <section class="more">
    <RouterLink to="/faqs">
      {{ t('home.more.faqs') }}
    </RouterLink>
    ·
    <RouterLink to="/about">
      {{ t('home.more.about') }}
    </RouterLink>
  </section>
</template>

<style scoped>
section {
  margin: 1.25rem 0;
}
</style>
