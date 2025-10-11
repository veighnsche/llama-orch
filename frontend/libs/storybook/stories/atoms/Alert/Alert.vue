<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-001: Implemented Alert component ported from React reference -->
<script setup lang="ts">
import { computed } from 'vue'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../../lib/utils'

// Define variants using CVA - ported from React reference
const alertVariants = cva(
  'relative w-full rounded-lg border px-4 py-3 text-sm grid has-[>svg]:grid-cols-[calc(var(--spacing)*4)_1fr] grid-cols-[0_1fr] has-[>svg]:gap-x-3 gap-y-0.5 items-start [&>svg]:size-4 [&>svg]:translate-y-0.5 [&>svg]:text-current',
  {
    variants: {
      variant: {
        default: 'bg-card text-card-foreground',
        destructive:
          'text-destructive bg-card [&>svg]:text-current *:data-[slot=alert-description]:text-destructive/90',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
)

// Define props interface
interface Props {
  variant?: VariantProps<typeof alertVariants>['variant']
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
})

// Compute classes
const classes = computed(() =>
  cn(alertVariants({ variant: props.variant }), props.class)
)
</script>

<template>
  <div
    role="alert"
    :class="classes"
    data-slot="alert"
  >
    <slot />
  </div>
</template>
