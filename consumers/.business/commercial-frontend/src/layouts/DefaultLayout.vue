<template>
  <div class="layout">
    <NavBar />
    <main class="content bp-grid">
      <RouterView />
    </main>
    <SiteFooter />
  </div>
</template>

<script setup lang="ts">
  import { RouterView } from 'vue-router'
  import NavBar from '@/components/NavBar.vue'
  import SiteFooter from '@/components/SiteFooter.vue'
</script>

<style scoped>
  .layout {
    display: flex;
    min-height: 100vh;
    flex-direction: column;
  }
  .content {
    flex: 1 1 auto;
    padding: 1rem 0;
  }
</style>
