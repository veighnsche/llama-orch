<script setup lang="ts">
import { reactive, ref } from 'vue'
import MenuToggle from './MenuToggle.vue'

const labels = reactive({ open: 'Open menu', close: 'Close menu', srOnly: 'Menu' })
const open = ref(false)
</script>

<template>
  <Story title="UI/MenuToggle" :layout="{ type: 'single', iframe: false }">
    <Variant title="Playground">
      <MenuToggle :open="open" :labels="labels" @toggle="open = !open" />
    </Variant>

    <template #controls>
      <HstCheckbox v-model="open" title="open" />
      <HstText v-model="labels.open" title="labels.open" />
      <HstText v-model="labels.close" title="labels.close" />
      <HstText v-model="labels.srOnly" title="labels.srOnly" />
    </template>
  </Story>
</template>
