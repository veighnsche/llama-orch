<script setup lang="ts">
import { reactive, ref } from 'vue'
import Drawer from './Drawer.vue'
import Button from './Button.vue'

const state = reactive({
  items: [
    { label: 'Home', to: '/' },
    { label: 'About', to: '/about' },
    { label: 'Contact', to: '/contact' },
  ] as Array<{ label: string; to?: string; href?: string }>,
})

const open = ref(false)
</script>

<template>
  <Story title="UI/Drawer" :layout="{ type: 'single', width: 360 }">
    <Variant title="Playground">
      <div style="padding: 8px">
        <Button variant="ghost" @click="open = true">Open drawer</Button>
      </div>
      <Drawer :open="open" :items="state.items" @close="open = false">
        <template #ops>
          <Button as="router-link" :to="'/contact'" variant="primary">Contact us</Button>
        </template>
      </Drawer>
    </Variant>

    <template #controls>
      <HstCheckbox v-model="open" title="open" />
      <HstJson v-model="state.items" title="items (array of {label,to|href})" />
      <p>Tip: Keep preview width below 920px to see the drawer (it hides on desktop).</p>
    </template>
  </Story>
</template>
