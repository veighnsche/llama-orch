<script setup lang="ts">
  import { reactive, computed } from 'vue'
  import Carousel from './Carousel.vue'

  const state = reactive({
    loop: true,
    autoplayMs: 0,
    pauseOnHover: true,
    pauseOnFocus: true,
    pauseOnInteraction: true,
    showArrows: true,
    showDots: true,
    snapAlign: 'center' as 'center' | 'start',
    height: 220,
    modelValue: 0,
    ariaLabel: 'Carousel',
    ariaLive: 'polite' as 'polite' | 'off',
    idPrefix: 'demo-carousel',
    labels: {
      prev: 'Previous slide',
      next: 'Next slide',
      dot: 'Go to slide {n}',
    },
    slides: [
      { title: 'Slide 1', color: 'linear-gradient(135deg, #64dfdf, #72efdd)' },
      { title: 'Slide 2', color: 'linear-gradient(135deg, #80ffdb, #48bfe3)' },
      { title: 'Slide 3', color: 'linear-gradient(135deg, #7400b8, #80ffdb)' },
    ] as Array<{ title: string; color: string }>,
  })

  const propsForCarousel = computed(() => ({
    slides: state.slides,
    modelValue: state.modelValue,
    loop: state.loop,
    autoplayMs: state.autoplayMs,
    pauseOnHover: state.pauseOnHover,
    pauseOnFocus: state.pauseOnFocus,
    pauseOnInteraction: state.pauseOnInteraction,
    showArrows: state.showArrows,
    showDots: state.showDots,
    snapAlign: state.snapAlign,
    height: state.height,
    ariaLabel: state.ariaLabel,
    ariaLive: state.ariaLive,
    labels: state.labels,
    idPrefix: state.idPrefix,
  }))
</script>

<template>
  <Story title="UI/Carousel" :layout="{ type: 'single', width: 720 }">
    <Variant title="Playground">
      <Carousel v-bind="propsForCarousel" @update:modelValue="state.modelValue = $event">
        <template #slide="{ item, index }">
          <div
            :style="{
              background: item.color,
              minHeight: state.height + 'px',
              display: 'grid',
              placeItems: 'center',
              color: 'var(--surface-alt)',
              fontWeight: 800,
              fontSize: '2rem',
            }"
          >
            {{ item.title }} ({{ index + 1 }})
          </div>
        </template>
      </Carousel>
    </Variant>

    <Variant title="Autoplay (3s)" :source="false">
      <Carousel :slides="state.slides" :height="state.height" :autoplayMs="3000" />
    </Variant>

    <Variant title="Dots only, no arrows" :source="false">
      <Carousel
        :slides="state.slides"
        :height="state.height"
        :showArrows="false"
        :showDots="true"
      />
    </Variant>

    <template #controls>
      <HstNumber v-model="state.height" title="height (px)" :min="120" :max="600" />
      <HstCheckbox v-model="state.loop" title="loop" />
      <HstNumber v-model="state.autoplayMs" title="autoplayMs (0 = off)" :step="500" :min="0" />
      <HstCheckbox v-model="state.pauseOnHover" title="pauseOnHover" />
      <HstCheckbox v-model="state.showArrows" title="showArrows" />
      <HstCheckbox v-model="state.showDots" title="showDots" />
    </template>
  </Story>
</template>
