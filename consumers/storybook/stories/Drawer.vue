<script setup lang="ts">
import { ref, watch, provide } from 'vue'

export type DrawerContext = {
  id: string | undefined
  open: boolean
  setOpen: (v: boolean) => void
  toggle: () => void
  // Root-level options for panel
  options: {
    hideOnDesktop: boolean
    ariaRole: 'dialog' | 'menu' | 'navigation'
    ariaModal: boolean
    label?: string
    labelledBy?: string
    closeOnEsc: boolean
    lockScroll: boolean
  }
}

const props = withDefaults(
  defineProps<{
    modelValue: boolean
    id?: string
    hideOnDesktop?: boolean
    ariaRole?: 'dialog' | 'menu' | 'navigation'
    ariaModal?: boolean
    label?: string
    labelledBy?: string
    closeOnEsc?: boolean
    lockScroll?: boolean
  }>(),
  {
    hideOnDesktop: true,
    ariaRole: 'dialog',
    ariaModal: true,
    closeOnEsc: true,
    lockScroll: true,
  },
)

const emit = defineEmits<{ 'update:modelValue': [value: boolean] }>()

const openRef = ref(props.modelValue)
watch(
  () => props.modelValue,
  (v) => {
    openRef.value = v
  },
)
watch(openRef, (v) => emit('update:modelValue', v))

function setOpen(v: boolean) {
  openRef.value = v
}
function toggle() {
  openRef.value = !openRef.value
}

provide('drawer-ctx', {
  get id() {
    return props.id
  },
  get open() {
    return openRef.value
  },
  setOpen,
  toggle,
  get options() {
    return {
      hideOnDesktop: !!props.hideOnDesktop,
      ariaRole: props.ariaRole || 'dialog',
      ariaModal: !!props.ariaModal,
      label: props.label,
      labelledBy: props.labelledBy,
      closeOnEsc: !!props.closeOnEsc,
      lockScroll: !!props.lockScroll,
    }
  },
} as DrawerContext)
</script>

<template>
  <slot />
</template>
