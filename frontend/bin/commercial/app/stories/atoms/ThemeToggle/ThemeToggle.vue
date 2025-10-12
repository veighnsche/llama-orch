<!-- Created by: TEAM-FE-010 -->
<!-- TEAM-FE-011: Fixed theme detection, removed hover effect, matched nav link colors -->
<script setup lang="ts">
import { useDark, useToggle } from '@vueuse/core'
import { Moon, Sun } from 'lucide-vue-next'
import { Button } from '../../index'

const isDark = useDark({
  selector: 'html',
  attribute: 'class',
  valueDark: 'dark',
  valueLight: '',
})
const toggleDark = useToggle(isDark)

interface Props {
  size?: 'default' | 'sm' | 'lg' | 'icon' | 'icon-sm' | 'icon-lg'
}

const props = withDefaults(defineProps<Props>(), {
  size: 'icon',
})
</script>

<template>
  <Button
    :size="props.size"
    variant="ghost"
    @click="toggleDark()"
    aria-label="Toggle theme"
    class="relative overflow-hidden text-muted-foreground hover:text-foreground hover:bg-transparent dark:hover:bg-transparent transition-colors"
  >
    <Sun
      v-if="!isDark"
      :size="20"
      class="transition-all duration-200"
    />
    <Moon
      v-else
      :size="20"
      class="transition-all duration-200"
    />
  </Button>
</template>
