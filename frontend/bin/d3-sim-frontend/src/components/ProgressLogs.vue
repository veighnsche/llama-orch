<template>
  <div>
    <h2>Progress Logs</h2>
    <pre>{{ logs }}</pre>
  </div>
</template>
<script setup lang="ts">
import { ref } from 'vue'
const logs = ref('JSONL output will appear here...')
</script>
