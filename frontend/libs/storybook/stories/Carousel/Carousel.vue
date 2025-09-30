<script setup lang="ts" generic="T = any">
import { ref, computed, watch, onMounted, onBeforeUnmount, nextTick } from 'vue'
import Button from '../Button/Button.vue'

// Public labels contract for i18n/customization
type CarouselLabels = {
  prev: string
  next: string
  dot: string // Use `{n}` placeholder for 1-based index
}

// Slots typing for DX
// - slide: per-slide content
// - empty: optional UI when there are no slides
defineSlots<{
  slide: (props: { item: T; index: number }) => any
  empty: () => any
}>()

const props = withDefaults(
  defineProps<{
    // Content
    slides: T[]
    // Index control (v-model)
    modelValue?: number
    // Behavior
    loop?: boolean
    autoplayMs?: number
    pauseOnHover?: boolean
    pauseOnFocus?: boolean
    pauseOnInteraction?: boolean
    // UI
    showArrows?: boolean
    showDots?: boolean
    snapAlign?: 'start' | 'center'
    height?: string | number
    // A11y
    ariaLabel?: string
    ariaLive?: 'polite' | 'off'
    labels?: CarouselLabels
    idPrefix?: string
    itemKey?: (item: T, index: number) => string | number
    respectReducedMotion?: boolean
  }>(),
  {
    modelValue: 0,
    loop: true,
    autoplayMs: 0,
    pauseOnHover: true,
    pauseOnFocus: true,
    pauseOnInteraction: true,
    showArrows: true,
    showDots: true,
    snapAlign: 'center',
    height: 'auto',
    ariaLabel: 'Carousel',
    ariaLive: 'polite',
    labels: () => ({ prev: 'Previous slide', next: 'Next slide', dot: 'Go to slide {n}' }),
    idPrefix: 'carousel',
    respectReducedMotion: true,
  },
)

const emit = defineEmits<{
  (e: 'update:modelValue', value: number): void
  (e: 'change', value: number): void
}>()

const viewportEl = ref<HTMLDivElement | null>(null)
const trackEl = ref<HTMLDivElement | null>(null)
const internalIndex = ref(Math.max(0, Math.min(props.modelValue ?? 0, Math.max(0, (props.slides?.length ?? 1) - 1))))

const slideCount = computed(() => Math.max(0, props.slides?.length ?? 0))
const canGoPrev = computed(() => props.loop || internalIndex.value > 0)
const canGoNext = computed(() => props.loop || internalIndex.value < slideCount.value - 1)

let rafId: number | null = null
let resizeObserver: ResizeObserver | null = null
let autoplayTimer: number | null = null
let isHovering = false
let isFocused = false
let isInteracting = false
let interactionTimer: number | null = null
let prefersReduced = false
let mql: MediaQueryList | null = null

// Stable handlers for add/removeEventListener
const onPointerDown = () => setInteractionPause()
const onWheel = () => setInteractionPause()
function updateReducedMotion() {
  prefersReduced = !!mql?.matches
  // restart autoplay to reflect preference
  startAutoplay()
}

function width(): number {
  return viewportEl.value?.clientWidth ?? 0
}

// Default render helpers when no slot is provided
function toLabel(item: T): string {
  if (item == null) return ''
  const t = typeof item
  if (t === 'string' || t === 'number' || t === 'boolean') return String(item)
  if (t === 'object') {
    const anyItem = item as any
    const val = anyItem?.title ?? anyItem?.label ?? anyItem?.name
    if (val != null) return String(val)
    try {
      return JSON.stringify(item)
    } catch {
      return Object.prototype.toString.call(item)
    }
  }
  return String(item)
}

function defaultStyle(item: T): Record<string, string> | undefined {
  if (item && typeof item === 'object') {
    const anyItem = item as any
    const bg = anyItem?.color ?? anyItem?.background
    if (bg) return { background: String(bg) }
  }
  return undefined
}

function setInteractionPause() {
  if (!props.pauseOnInteraction) return
  isInteracting = true
  if (interactionTimer) clearTimeout(interactionTimer)
  const delay = Math.max(800, props.autoplayMs ? Math.min(props.autoplayMs * 1.2, 5000) : 1200)
  interactionTimer = window.setTimeout(() => {
    isInteracting = false
  }, delay) as unknown as number
}

function onKeydown(e: KeyboardEvent) {
  const key = e.key
  if (key === 'ArrowLeft') {
    e.preventDefault()
    setInteractionPause()
    prev()
  } else if (key === 'ArrowRight') {
    e.preventDefault()
    setInteractionPause()
    next()
  } else if (key === 'Home') {
    e.preventDefault()
    setInteractionPause()
    scrollToIndex(0)
  } else if (key === 'End') {
    e.preventDefault()
    setInteractionPause()
    scrollToIndex(slideCount.value - 1)
  }
}

function onFocusIn() {
  isFocused = true
}
function onFocusOut() {
  isFocused = false
}

function scrollToIndex(idx: number, behavior?: ScrollBehavior) {
  if (!trackEl.value) return
  const maxIdx = Math.max(0, slideCount.value - 1)
  let next = Math.min(Math.max(idx, 0), maxIdx)
  // wrap if loop
  if (props.loop) {
    if (idx < 0) next = maxIdx
    if (idx > maxIdx) next = 0
  }

  const left = next * width()
  const bh: ScrollBehavior = behavior ?? (prefersReduced ? 'auto' : 'smooth')
  try {
    trackEl.value.scrollTo({ left, behavior: bh })
  } catch {
    // Safari sometimes throws on smooth; fallback
    trackEl.value.scrollLeft = left
  }
  if (internalIndex.value !== next) {
    internalIndex.value = next
    emit('update:modelValue', next)
    emit('change', next)
  }
}

function onScroll() {
  if (!trackEl.value) return
  if (rafId) cancelAnimationFrame(rafId)
  rafId = requestAnimationFrame(() => {
    const w = width()
    if (!w) return
    const idx = Math.round(trackEl.value!.scrollLeft / w)
    if (idx !== internalIndex.value) {
      internalIndex.value = idx
      emit('update:modelValue', idx)
    }
  })
}

function prev() {
  if (!canGoPrev.value) return
  scrollToIndex(internalIndex.value - 1)
}
function next() {
  if (!canGoNext.value) return
  scrollToIndex(internalIndex.value + 1)
}

function startAutoplay() {
  stopAutoplay()
  if (!props.autoplayMs || props.autoplayMs <= 0) return
  if (props.respectReducedMotion && prefersReduced) return
  autoplayTimer = window.setInterval(() => {
    if ((props.pauseOnHover && isHovering) || (props.pauseOnFocus && isFocused) || (props.pauseOnInteraction && isInteracting)) return
    if (!slideCount.value) return
    if (props.loop || internalIndex.value < slideCount.value - 1) {
      scrollToIndex(internalIndex.value + 1)
    } else {
      // rewind if not looping
      scrollToIndex(0)
    }
  }, props.autoplayMs) as unknown as number
}
function stopAutoplay() {
  if (autoplayTimer) {
    clearInterval(autoplayTimer)
    autoplayTimer = null
  }
}

watch(
  () => props.modelValue,
  (v) => {
    if (typeof v === 'number' && v !== internalIndex.value) {
      // clamp to range
      const maxIdx = Math.max(0, slideCount.value - 1)
      const next = Math.min(Math.max(v, 0), maxIdx)
      internalIndex.value = next
      nextTick(() => scrollToIndex(next))
    }
  },
)

watch(
  () => props.autoplayMs,
  () => startAutoplay(),
)

// Re-clamp index when slides change
watch(
  () => slideCount.value,
  () => {
    const maxIdx = Math.max(0, slideCount.value - 1)
    if (internalIndex.value > maxIdx) {
      internalIndex.value = maxIdx
      nextTick(() => scrollToIndex(maxIdx, 'auto'))
    } else {
      // realign after structural change
      nextTick(() => scrollToIndex(internalIndex.value, 'auto'))
    }
  },
)

onMounted(() => {
  // initial position
  nextTick(() => scrollToIndex(internalIndex.value, 'auto'))

  trackEl.value?.addEventListener('scroll', onScroll, { passive: true })
  trackEl.value?.addEventListener('pointerdown', onPointerDown, { passive: true })
  trackEl.value?.addEventListener('wheel', onWheel, { passive: true })
  viewportEl.value?.addEventListener('keydown', onKeydown)
  viewportEl.value?.addEventListener('focusin', onFocusIn)
  viewportEl.value?.addEventListener('focusout', onFocusOut)

  // observe resize to keep alignment
  resizeObserver = new ResizeObserver(() => {
    // maintain current index position after resize
    scrollToIndex(internalIndex.value, 'auto')
  })
  if (viewportEl.value) resizeObserver.observe(viewportEl.value)

  // reduced motion detection
  if (typeof window !== 'undefined' && 'matchMedia' in window) {
    mql = window.matchMedia('(prefers-reduced-motion: reduce)')
    updateReducedMotion()
    mql.addEventListener?.('change', updateReducedMotion)
  }

  startAutoplay()
})

onBeforeUnmount(() => {
  trackEl.value?.removeEventListener('scroll', onScroll)
  trackEl.value?.removeEventListener('pointerdown', onPointerDown)
  trackEl.value?.removeEventListener('wheel', onWheel)
  viewportEl.value?.removeEventListener('keydown', onKeydown)
  viewportEl.value?.removeEventListener('focusin', onFocusIn)
  viewportEl.value?.removeEventListener('focusout', onFocusOut)
  if (resizeObserver && viewportEl.value) resizeObserver.unobserve(viewportEl.value)
  stopAutoplay()
  if (rafId) cancelAnimationFrame(rafId)
  if (interactionTimer) clearTimeout(interactionTimer)
  if (mql) mql.removeEventListener?.('change', updateReducedMotion)
})

const onMouseEnter = () => {
  isHovering = true
}
const onMouseLeave = () => {
  isHovering = false
}

// Public methods
defineExpose({
  next,
  prev,
  goTo: scrollToIndex,
  play: startAutoplay,
  pause: stopAutoplay,
})
</script>

<template>
  <div
    class="carousel"
    :class="{ 'carousel--snap-start': snapAlign === 'start' }"
    role="region"
    :aria-label="ariaLabel"
    aria-roledescription="carousel"
    :aria-live="ariaLive"
  >
    <div
      ref="viewportEl"
      class="carousel__viewport"
      :style="{ height: typeof height === 'number' ? height + 'px' : height }"
      @mouseenter="onMouseEnter"
      @mouseleave="onMouseLeave"
      tabindex="0"
    >
      <div v-if="slideCount > 0" ref="trackEl" class="carousel__track">
        <div
          v-for="(item, i) in slides"
          :key="itemKey ? itemKey(item, i) : i"
          class="carousel__slide"
          role="group"
          aria-roledescription="slide"
          :aria-label="`Slide ${i + 1} of ${slideCount}`"
          :id="`${idPrefix}-slide-${i}`"
        >
          <slot name="slide" :item="item" :index="i">
            <!-- default renderer if no slot provided -->
            <div class="carousel__default" :style="defaultStyle(item)">
              {{ toLabel(item) }}
            </div>
          </slot>
        </div>
      </div>
      <div v-else class="carousel__empty">
        <slot name="empty">No slides</slot>
      </div>

      <!-- arrows -->
      <div v-if="showArrows && slideCount > 0" class="carousel__arrows" aria-hidden="false">
        <div class="carousel__arrow carousel__arrow--prev">
          <Button
            variant="ghost"
            icon-only
            @click="() => { setInteractionPause(); prev() }"
            :disabled="!canGoPrev"
            :aria-label="(labels?.prev) ?? 'Previous slide'"
          >
            ‹
          </Button>
        </div>
        <div class="carousel__arrow carousel__arrow--next">
          <Button
            variant="ghost"
            icon-only
            @click="() => { setInteractionPause(); next() }"
            :disabled="!canGoNext"
            :aria-label="(labels?.next) ?? 'Next slide'"
          >
            ›
          </Button>
        </div>
      </div>

      <!-- dots -->
      <div v-if="showDots && slideCount > 0" class="carousel__dots" role="tablist" aria-label="Carousel Pagination">
        <button
          v-for="i in slideCount"
          :key="i - 1"
          class="carousel__dot"
          :class="{ 'is-active': internalIndex === i - 1 }"
          role="tab"
          :aria-selected="internalIndex === i - 1"
          :aria-controls="`${idPrefix}-slide-${i - 1}`"
          :aria-label="(labels?.dot || 'Go to slide {n}').replace('{n}', String(i))"
          :aria-current="internalIndex === i - 1 ? 'page' : undefined"
          @click="() => { setInteractionPause(); scrollToIndex(i - 1) }"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.carousel {
  position: relative;
}
.carousel__viewport {
  position: relative;
  width: 100%;
  overflow: hidden;
  border-radius: var(--radius-md);
  background: var(--surface);
  display: flex;
  height: 100%;
  min-height: 0;
}
.carousel__track {
  display: flex;
  overflow-x: auto;
  overflow-y: hidden;
  height: 100%;
  min-height: 0;
  flex: 1 1 auto;
  scroll-snap-type: x mandatory;
  -webkit-overflow-scrolling: touch;
  scroll-behavior: smooth;
  /* allow native touch scroll and contain horizontal overscroll */
  overscroll-behavior-x: contain;
  scrollbar-width: none; /* Firefox */
}
.carousel__track::-webkit-scrollbar { /* WebKit */
  display: none;
}
.carousel__slide {
  flex: 0 0 100%;
  display: flex;
  scroll-snap-align: center;
  height: 100%;
  min-height: 0;
}
.carousel__slide > * { flex: 1 1 auto; height: 100%; }
.carousel__slide :deep(.media-card) { height: 100%; }
.carousel__slide :deep(.media-card__content) { height: 100%; }
.carousel--snap-start .carousel__slide { scroll-snap-align: start; }

.carousel__empty {
  display: grid;
  place-items: center;
  min-height: 160px;
  color: var(--text-muted, var(--text));
}

/* default slide content */
.carousel__default {
  display: grid;
  place-items: center;
  min-height: 180px;
  color: var(--text);
}

/* arrows */
.carousel__arrows {
  position: absolute;
  inset: 0;
  z-index: 2;
  opacity: 0;
  visibility: hidden;
  transition: opacity 160ms ease;
}
.carousel__arrow {
  pointer-events: auto; /* buttons are interactive */
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  z-index: 2;
  box-shadow: 0 1px 4px rgba(0,0,0,0.12);
}
.carousel__arrow--prev { left: 8px; }
.carousel__arrow--next { right: 8px; }

/* Reveal arrows when interacting */
.carousel:hover .carousel__arrows,
.carousel:focus-within .carousel__arrows {
  opacity: 1;
  visibility: visible;
}

/* dots */
.carousel__dots {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  bottom: 8px;
  display: flex;
  gap: 6px;
  padding: 4px 6px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--surface), transparent 40%);
  z-index: 2;
}
.carousel__dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--surface-muted);
  border: none;
  padding: 0;
  cursor: pointer;
}
.carousel__dot.is-active {
  background: var(--acc-cyan);
}
</style>
