<!-- TEAM-FE-010: Nuxt app root with page routing -->
<!-- TEAM-FE-CONSOLIDATE: Added Navigation component -->
<script setup lang="ts">
import { Navigation } from '~/stories'
</script>

<template>
  <div>
    <Navigation />
    <NuxtPage />
  </div>
</template>
