<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-002: Implemented PricingCard molecule -->
<script setup lang="ts">
import { computed } from 'vue'
import Card from '../../atoms/Card/Card.vue'
import CardHeader from '../../atoms/Card/CardHeader.vue'
import CardTitle from '../../atoms/Card/CardTitle.vue'
import CardContent from '../../atoms/Card/CardContent.vue'
import CardFooter from '../../atoms/Card/CardFooter.vue'
import Button from '../../atoms/Button/Button.vue'
import { Check } from 'lucide-vue-next'
import { cn } from '../../../lib/utils'

interface Props {
  title: string
  price: string
  priceSubtext: string
  description: string
  features: string[]
  buttonText: string
  buttonVariant?: 'default' | 'outline' | 'secondary' | 'destructive' | 'ghost' | 'link'
  highlighted?: boolean
  badge?: string
  teamSize?: string
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  buttonVariant: 'default',
  highlighted: false,
})

const cardClasses = computed(() =>
  cn(
    'relative p-8 space-y-6',
    props.highlighted
      ? 'bg-primary/10 border-2 border-primary'
      : 'bg-card border-2 border-border',
    props.class
  )
)

const buttonClasses = computed(() => {
  if (props.highlighted) {
    return 'w-full bg-primary hover:bg-primary/90 text-primary-foreground'
  }
  return 'w-full bg-transparent'
})

const firstFeatureClass = computed(() => {
  return props.highlighted ? 'text-foreground font-medium' : 'text-muted-foreground'
})
</script>

<template>
  <Card :class="cardClasses">
    <!-- Badge for "Most Popular" -->
    <div
      v-if="badge"
      class="absolute -top-4 left-1/2 -translate-x-1/2"
    >
      <span class="bg-primary text-primary-foreground px-4 py-1 rounded-full text-sm font-medium">
        {{ badge }}
      </span>
    </div>

    <!-- Header with title and price -->
    <div>
      <CardHeader class="p-0">
        <CardTitle class="text-2xl font-bold text-foreground">
          {{ title }}
        </CardTitle>
      </CardHeader>
      <div class="mt-4">
        <span class="text-4xl font-bold text-foreground">{{ price }}</span>
        <span class="text-muted-foreground ml-2">{{ priceSubtext }}</span>
      </div>
      <p
        v-if="teamSize"
        class="text-sm text-muted-foreground mt-1"
      >
        {{ teamSize }}
      </p>
    </div>

    <!-- Features list -->
    <CardContent class="p-0">
      <ul class="space-y-3">
        <li
          v-for="(feature, index) in features"
          :key="feature"
          class="flex items-start gap-2"
        >
          <Check class="h-5 w-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
          <span :class="index === 0 ? firstFeatureClass : 'text-muted-foreground'">
            {{ feature }}
          </span>
        </li>
      </ul>
    </CardContent>

    <!-- CTA Button -->
    <CardFooter class="p-0">
      <Button
        :variant="buttonVariant"
        :class="buttonClasses"
      >
        {{ buttonText }}
      </Button>
    </CardFooter>

    <!-- Description -->
    <p class="text-sm text-muted-foreground text-center">
      {{ description }}
    </p>
  </Card>
</template>
