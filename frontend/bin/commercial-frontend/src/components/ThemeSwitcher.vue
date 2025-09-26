<script setup lang="ts">
import { computed, ref, watchEffect } from 'vue'
import type { ThemeMode } from '@/composables/useTheme'
import { useTheme } from '@/composables/useTheme'

const { get, set, effective, onSystemChange } = useTheme()

const mode = ref<ThemeMode>(get())

watchEffect(() => {
  // Keep local state in sync if something else changes it
  mode.value = get()
})

function cycle() {
  mode.value = mode.value === 'system' ? 'light' : mode.value === 'light' ? 'dark' : 'system'
  set(mode.value)
}

const label = computed(() => {
  if (mode.value === 'system') return 'Theme: System'
  if (mode.value === 'light') return 'Theme: Light'
  return 'Theme: Dark'
})

const icon = computed(() => {
  // simple emoji icons; you can swap with SVGs if desired
  return mode.value === 'system' ? '💻' : mode.value === 'light' ? '☀️' : '🌙'
})
</script>

<template>
  <button class="theme-toggle" type="button" :aria-label="label" @click="cycle">
    <span class="icon" aria-hidden="true">{{ icon }}</span>
  </button>
</template>

<style scoped>
.theme-toggle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 28px;
  width: 32px;
  padding: 0 6px;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text);
  border: 1px solid color-mix(in srgb, var(--text) 16%, transparent);
  box-shadow: var(--shadow-sm);
  cursor: pointer;
}
.theme-toggle:hover {
  background: color-mix(in srgb, var(--surface-alt) 80%, transparent);
}
.icon {
  font-size: 13px;
  line-height: 1;
}
</style>
