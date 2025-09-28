<template>
  <div>
    <h2>Run Panel</h2>
    <button @click="run">Run (scaffold)</button>
  </div>
</template>
<script setup lang="ts">
function run(){
  // TODO: spawn CLI via backend/electron wrapper
  alert('Run clicked (scaffold)')
}
</script>
