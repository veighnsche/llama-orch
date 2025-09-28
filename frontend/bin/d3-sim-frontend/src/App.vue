<template>
  <main class="container">
    <h1>D3 Simulation UI</h1>
    <RunPanel />
    <section class="grid">
      <InputsEditor />
      <ProgressLogs />
    </section>
    <OutputsViewer />
  </main>
</template>

<script setup lang="ts">
import InputsEditor from './components/InputsEditor.vue'
import RunPanel from './components/RunPanel.vue'
import ProgressLogs from './components/ProgressLogs.vue'
import OutputsViewer from './components/OutputsViewer.vue'
</script>

<style>
.container { padding: 1rem; font-family: system-ui, sans-serif; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
</style>
