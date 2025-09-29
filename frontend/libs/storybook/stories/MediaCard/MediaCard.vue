<script setup lang="ts">
  import { computed } from 'vue'

  const props = withDefaults(
    defineProps<{
      imageSrc: string
      imageAlt?: string
      title?: string
      subtitle?: string
      side?: 'left' | 'right'
      overlayStrength?: number // 0..1
      overlayColor?: string
      enforceAspect?: boolean
      aspectRatio?: string
      alignY?: 'top' | 'center' | 'bottom'
      ariaLabel?: string
    }>(),
    {
      imageAlt: '',
      title: '',
      subtitle: '',
      side: 'left',
      overlayStrength: 0.55,
      overlayColor: 'rgba(0, 0, 0, 1)',
      enforceAspect: true,
      aspectRatio: '16 / 9',
      alignY: 'center',
      ariaLabel: undefined,
    },
  )

  const dir = computed(() => (props.side === 'left' ? 'to right' : 'to left'))
  const justify = computed(() => (props.side === 'left' ? 'flex-start' : 'flex-end'))
  const alignY = computed(() =>
    props.alignY === 'top' ? 'flex-start' : props.alignY === 'bottom' ? 'flex-end' : 'center',
  )

  const overlay = computed(() => {
    const s = Math.max(0, Math.min(props.overlayStrength ?? 0.55, 1))
    const base = props.overlayColor?.trim() || 'rgba(0,0,0,1)'
    const toOverlayRgba = (color: string, alpha: number): string => {
      const m = color.match(/rgba?\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d.]+))?\)/i)
      if (m) {
        const r = Number(m[1]) || 0
        const g = Number(m[2]) || 0
        const b = Number(m[3]) || 0
        return `rgba(${r}, ${g}, ${b}, ${alpha})`
      }
      return `rgba(0, 0, 0, ${alpha})`
    }
    const col = toOverlayRgba(base, s)
    return `linear-gradient(${dir.value}, ${col} 0%, transparent 60%)`
  })

  const bgStyle = computed(() => ({
    backgroundImage: `${overlay.value}, url("${props.imageSrc}")`,
  }))
</script>

<template>
  <section
    class="media-card"
    :class="{
      'is-left': side === 'left',
      'is-right': side === 'right',
      'is-ratio': enforceAspect,
    }"
    role="group"
    :aria-label="ariaLabel || title || imageAlt || 'Media card'"
  >
    <div class="media-card__background" :style="bgStyle" aria-hidden="true" />

    <div class="media-card__content" :style="{ justifyContent: justify, alignItems: alignY }">
      <div
        class="media-card__inner"
        :class="{ 'on-left': side === 'left', 'on-right': side === 'right' }"
      >
        <header class="media-card__header">
          <h3 v-if="title" class="media-card__title">{{ title }}</h3>
          <p v-if="subtitle" class="media-card__subtitle">{{ subtitle }}</p>
        </header>
        <div class="media-card__actions">
          <slot />
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
  .media-card {
    position: relative;
    width: 100%;
    border-radius: var(--radius-md);
    overflow: hidden;
    background: var(--surface, #111);
    color: var(--surface-alt, #fff);
  }
  .media-card.is-ratio {
    aspect-ratio: v-bind(aspectRatio);
  }
  /* When not enforcing aspect ratio, fill parent (e.g., Carousel viewport) */
  .media-card:not(.is-ratio) {
    height: 100%;
  }
  .media-card:not(.is-ratio) .media-card__content {
    min-height: 100%;
  }

  .media-card__background {
    position: absolute;
    inset: 0;
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
    filter: saturate(0.98);
  }

  .media-card__content {
    position: relative;
    z-index: 1;
    display: flex;
    width: 100%;
    height: 100%;
  }
  .media-card.is-ratio .media-card__content { min-height: 160px; }

  .media-card__inner {
    display: grid;
    gap: 0.5rem;
    padding: clamp(14px, 4vw, 28px);
    max-width: 60ch;
  }
  .media-card__inner.on-left { margin-left: clamp(8px, 2vw, 16px); }
  .media-card__inner.on-right {
    margin-right: clamp(8px, 2vw, 16px);
    text-align: right;         /* text nodes */
    justify-items: end;        /* grid children alignment */
  }

  .media-card__title {
    margin: 0;
    font-size: clamp(1.05rem, 1.2rem + 0.6vw, 1.6rem);
    line-height: 1.2;
    font-weight: 800;
  }
  .media-card__subtitle {
    margin: 0;
    opacity: 0.95;
  }

  .media-card__actions {
    display: inline-flex;
    gap: 8px;
    margin-top: 6px;
  }
  .media-card__inner.on-right .media-card__actions {
    justify-self: end;
  }

  /* Small screens: allow content to breathe */
  @media (max-width: 560px) {
    .media-card__inner { max-width: 100%; }
  }
</style>
