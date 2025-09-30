<script setup lang="ts">
import { reactive, computed } from 'vue'
import MediaCard from './MediaCard.vue'
import Carousel from '../Carousel/Carousel.vue'
import Button from '../Button/Button.vue'

const state = reactive({
  imageSrc:
    'https://images.unsplash.com/photo-1518779578993-ec3579fee39f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80',
  imageAlt: 'Abstract GPU server rack',
  title: 'Private GPU Hosting',
  subtitle: 'Dedicated clusters in the Netherlands',
  side: 'left' as 'left' | 'right',
  overlayStrength: 0.55,
  overlayColor: 'rgba(0,0,0,1)',
  enforceAspect: true,
  aspectRatio: '16 / 9',
  alignY: 'center' as 'top' | 'center' | 'bottom',
})

const slides = reactive([
  {
    id: 's1',
    imageSrc:
      'https://images.unsplash.com/photo-1518779578993-ec3579fee39f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80',
    imageAlt: 'GPU rack on the left',
    title: 'Latency under 20ms',
    subtitle: 'Amsterdam region, privacy-first',
    side: 'left' as 'left' | 'right',
  },
  {
    id: 's2',
    imageSrc:
      'https://images.unsplash.com/photo-1518770660439-4636190af475?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80',
    imageAlt: 'Cables and switches',
    title: 'Predictable pricing',
    side: 'right' as 'left' | 'right',
  },
  {
    id: 's3',
    imageSrc:
      'https://images.unsplash.com/photo-1527443154391-507e9dc6c5cc?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80',
    imageAlt: 'Data center corridor',
    title: 'Managed operations',
    subtitle: 'Monitoring and alerting included',
    side: 'left' as 'left' | 'right',
  },
])

const cardProps = computed(() => ({
  imageSrc: state.imageSrc,
  imageAlt: state.imageAlt,
  title: state.title,
  subtitle: state.subtitle,
  side: state.side,
  overlayStrength: state.overlayStrength,
  overlayColor: state.overlayColor,
  enforceAspect: state.enforceAspect,
  aspectRatio: state.aspectRatio,
  alignY: state.alignY,
}))
</script>

<template>
  <Story title="UI/MediaCard" :layout="{ type: 'single', width: 920 }">
    <Variant title="Playground">
      <MediaCard v-bind="cardProps">
        <Button variant="primary"> Learn more </Button>
        <Button variant="ghost"> Contact </Button>
      </MediaCard>
    </Variant>

    <Variant title="Left (16:9)">
      <MediaCard
        image-src="https://images.unsplash.com/photo-1518779578993-ec3579fee39f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80"
        image-alt="GPU rack on the left"
        title="Low latency"
        subtitle="AMS region"
        side="left"
      >
        <Button variant="primary"> Get started </Button>
      </MediaCard>
    </Variant>

    <Variant title="Right (16:9)">
      <MediaCard
        image-src="https://images.unsplash.com/photo-1518770660439-4636190af475?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80"
        image-alt="Cables and switches"
        title="Predictable pricing"
        subtitle="Clear SLAs and quotas"
        side="right"
      >
        <Button variant="primary"> Pricing </Button>
      </MediaCard>
    </Variant>

    <Variant title="Carousel demo" :source="false">
      <div style="position: relative; width: 100%; aspect-ratio: 16 / 9; max-width: 960px;">
        <div style="position: absolute; inset: 0; display: flex;">
          <Carousel :slides="slides" style="width: 100%; height: 100%" height="100%" :showDots="true" :showArrows="true" :pauseOnHover="true" :autoplayMs="4000" :itemKey="(s) => s.id">
            <template #slide="{ item }">
              <MediaCard :image-src="item.imageSrc" :image-alt="item.imageAlt" :title="item.title" :subtitle="item.subtitle" :side="item.side" :enforce-aspect="false">
                <Button variant="primary"> Learn more </Button>
              </MediaCard>
            </template>
          </Carousel>
        </div>
      </div>
    </Variant>

    <template #controls>
      <HstText v-model="state.imageSrc" title="imageSrc" />
      <HstText v-model="state.imageAlt" title="imageAlt" />
      <HstText v-model="state.title" title="title" />
      <HstText v-model="state.subtitle" title="subtitle" />
      <HstSelect v-model="state.side" title="side" :options="['left', 'right']" />
      <HstSelect v-model="state.alignY" title="alignY" :options="['top', 'center', 'bottom']" />
      <HstNumber v-model="state.overlayStrength" title="overlayStrength (0..1)" :min="0" :max="1" :step="0.05" />
      <HstText v-model="state.overlayColor" title="overlayColor (CSS color)" />
      <HstCheckbox v-model="state.enforceAspect" title="enforceAspect" />
      <HstText v-model="state.aspectRatio" title="aspectRatio (e.g. '16 / 9')" />
    </template>
  </Story>
</template>
