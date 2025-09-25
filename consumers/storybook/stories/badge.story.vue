<script setup lang="ts">
import { reactive } from 'vue'
import Badge from './badge.vue'

const state = reactive({
  label: 'Badge',
  variant: 'default' as 'default' | 'green' | 'cyan' | 'slate' | 'purple',
  size: 'md' as 'sm' | 'md',
  showIcon: true,
})
</script>

<template>
  <Story title="UI/Badge" :layout="{ type: 'grid', width: 320 }">
    <!-- Playground with controls -->
    <Variant title="Playground">
      <div style="padding: 8px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
        <Badge :variant="state.variant" :size="state.size">
          <template v-if="state.showIcon" #icon>
            <!-- Simple icon using currentColor so it adapts to --badge-color -->
            <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true" width="18" height="18">
              <circle cx="10" cy="10" r="6" />
            </svg>
          </template>
          {{ state.label }}
        </Badge>
      </div>
    </Variant>

    <!-- Variant presets: by variant -->
    <Variant title="Default"><Badge>Default</Badge></Variant>
    <Variant title="Green"><Badge variant="green">Green</Badge></Variant>
    <Variant title="Cyan"><Badge variant="cyan">Cyan</Badge></Variant>
    <Variant title="Slate"><Badge variant="slate">Slate</Badge></Variant>
    <Variant title="Purple"><Badge variant="purple">Purple</Badge></Variant>

    <!-- Sizes -->
    <Variant title="Size: sm"><Badge size="sm">Small</Badge></Variant>
    <Variant title="Size: md"><Badge size="md">Medium</Badge></Variant>

    <!-- With icon slot -->
    <Variant title="With icon">
      <Badge variant="green">
        <template #icon>
          <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true" width="18" height="18">
            <circle cx="10" cy="10" r="6" />
          </svg>
        </template>
        With icon
      </Badge>
    </Variant>

    <template #controls>
      <HstText v-model="state.label" title="label" />
      <HstSelect v-model="state.variant" title="variant" :options="['default', 'green', 'cyan', 'slate', 'purple']" />
      <HstSelect v-model="state.size" title="size" :options="['sm', 'md']" />
      <HstCheckbox v-model="state.showIcon" title="showIcon" />
    </template>
  </Story>
</template>
