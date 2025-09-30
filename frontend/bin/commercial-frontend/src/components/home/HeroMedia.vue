<script setup lang="ts">
import { computed } from 'vue'
// Import from the workspace dependency, not via relative paths
import { Carousel, MediaCard, Button } from 'orchyra-storybook/stories'

type Slide = {
  id: string
  imageSrc: string
  imageAlt: string
  title: string
  subtitle?: string
  side: 'left' | 'right'
}

const slides: Slide[] = [
  {
    id: 'hero-1',
    imageSrc:
      'https://images.unsplash.com/photo-1518779578993-ec3579fee39f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80',
    imageAlt: 'GPU rack on the left',
    title: 'Latency under 20ms',
    subtitle: 'Amsterdam region, privacy-first',
    side: 'left',
  },
  {
    id: 'hero-2',
    imageSrc:
      'https://images.unsplash.com/photo-1518770660439-4636190af475?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80',
    imageAlt: 'Cables and switches',
    title: 'Predictable pricing',
    subtitle: 'Clear SLAs and quotas',
    side: 'right',
  },
  {
    id: 'hero-3',
    imageSrc:
      'https://images.unsplash.com/photo-1527443154391-507e9dc6c5cc?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80',
    imageAlt: 'Data center corridor',
    title: 'Managed operations',
    subtitle: 'Monitoring and alerting included',
    side: 'left',
  },
]

const rootAria = computed(() => ({
  role: 'region',
  ariaLabel: 'Homepage highlights',
}))
</script>

<template>
  <section class="hero-media" v-bind="rootAria">
    <div class="hero-media__frame">
      <div class="hero-media__inner">
        <Carousel
          :slides="slides"
          height="100%"
          :showDots="true"
          :showArrows="true"
          :autoplayMs="5000"
          :pauseOnHover="true"
          :itemKey="(s: Slide) => s.id"
        >
          <template #slide="{ item }">
            <MediaCard
              :image-src="item.imageSrc"
              :image-alt="item.imageAlt"
              :title="item.title"
              :subtitle="item.subtitle"
              :side="item.side"
              :enforce-aspect="false"
            >
              <Button as="router-link" to="/service-menu" variant="primary"> Learn more </Button>
            </MediaCard>
          </template>
        </Carousel>
      </div>
    </div>
  </section>
</template>

<style scoped>
/* Center the carousel; width handled by frame (mobile-first: 100%) */
.hero-media {
  margin: 0;
  display: grid;
  justify-items: center;
}

.hero-media__frame {
  position: relative;
  width: 100%;                /* mobile: no cap */
  margin-inline: auto;        /* center within section */
  aspect-ratio: 16 / 9;       /* enforce 16:9 viewport */
  max-width: 420px;           /* cap on larger screens */
}

.hero-media__inner {
  position: absolute;
  inset: 0;
  display: flex;
}

/* ensure the carousel fits and inherits rounded corners */
:deep(.carousel) {
  width: 100%;
  height: 100%;
}
:deep(.carousel__viewport) {
  border-radius: var(--radius-md);
  width: 100%;
  height: 100%;
}
</style>

