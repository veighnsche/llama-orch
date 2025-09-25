<script setup lang="ts">
  import { onMounted, onBeforeUnmount, ref, watch, nextTick } from 'vue'
  import { RouterLink, type RouteLocationRaw } from 'vue-router'
  type NavItem = {
    label: string
    to?: RouteLocationRaw
    href?: string
    ariaLabel?: string
  }
  const props = withDefaults(
    defineProps<{
      id?: string
      open: boolean
      items?: NavItem[]
      hideOnDesktop?: boolean
      /* Accessibility */
      ariaRole?: 'dialog' | 'menu' | 'navigation'
      ariaModal?: boolean
      label?: string
      labelledBy?: string
      /* Behavior */
      closeOnEsc?: boolean
      lockScroll?: boolean
    }>(),
    {
      hideOnDesktop: true,
      ariaRole: 'dialog',
      ariaModal: true,
      closeOnEsc: true,
      lockScroll: true,
      items: () => [],
    },
  )
  const emit = defineEmits<{ close: []; 'update:open': [value: boolean] }>()

  const panelEl = ref<HTMLElement | null>(null)
  let prevFocused: Element | null = null

  function doClose() {
    emit('close')
    emit('update:open', false)
  }

  function onKeydown(e: KeyboardEvent) {
    if (props.closeOnEsc && e.key === 'Escape') {
      e.preventDefault()
      doClose()
    }
  }

  function lockScrollOnBody(lock: boolean) {
    if (!props.lockScroll) return
    const body = document?.body
    if (!body) return
    if (lock) {
      body.style.overflow = 'hidden'
    } else {
      body.style.overflow = ''
    }
  }

  async function focusFirstFocusable() {
    await nextTick()
    const root = panelEl.value
    if (!root) return
    const focusable = root.querySelector<HTMLElement>(
      'a,button,textarea,input,select,[tabindex]:not([tabindex="-1"])',
    )
    ;(focusable || root).focus()
  }

  watch(
    () => props.open,
    async (isOpen) => {
      if (isOpen) {
        prevFocused = document.activeElement
        document.addEventListener('keydown', onKeydown)
        lockScrollOnBody(true)
        await focusFirstFocusable()
      } else {
        document.removeEventListener('keydown', onKeydown)
        lockScrollOnBody(false)
        // restore focus
        ;(prevFocused as HTMLElement | null)?.focus?.()
      }
    },
    { immediate: true },
  )

  onMounted(() => {
    if (props.open) {
      document.addEventListener('keydown', onKeydown)
      lockScrollOnBody(true)
    }
  })
  onBeforeUnmount(() => {
    document.removeEventListener('keydown', onKeydown)
    lockScrollOnBody(false)
  })
</script>

<template>
  <transition name="fade">
    <div
      v-if="props.open"
      class="drawer"
      :id="props.id"
      :class="{ 'drawer--desktop-hidden': props.hideOnDesktop }"
      @click.self="doClose()"
    >
      <div
        class="drawer-panel"
        :role="props.ariaRole"
        :aria-modal="String(props.ariaModal)"
        :aria-labelledby="props.labelledBy"
        :aria-label="props.label"
        tabindex="-1"
        ref="panelEl"
      >
        <template v-if="props.items && props.items.length">
          <ul class="drawer-links">
            <li v-for="(item, idx) in props.items" :key="idx">
              <component :is="item.to ? RouterLink : 'a'" :to="item.to" :href="item.href" @click="doClose()">
                {{ item.label }}
              </component>
            </li>
          </ul>
        </template>
        <slot />
        <div class="drawer-ops">
          <slot name="ops" />
        </div>
      </div>
    </div>
  </transition>
</template>

<style scoped>
  .drawer {
    position: fixed;
    inset: 0;
    background: color-mix(in srgb, var(--text) 42%, transparent);
    z-index: 1000;
    display: block;
  }
  .drawer-panel {
    margin: 0.5rem;
    border-radius: var(--radius-xl);
    background: var(--surface-alt);
    border: 1px solid var(--surface-muted);
    box-shadow: var(--shadow-lg);
    padding: 0.75rem;
  }
  .drawer-links {
    list-style: none;
    margin: 0;
    padding: 0.25rem 0 0.5rem 0;
    display: grid;
    gap: 0.25rem;
  }
  .drawer-links a,
  .drawer-links :global(a.router-link-active),
  .drawer-links :global(a.router-link-exact-active) {
    display: block;
    padding: 0.6rem 0.6rem;
    border-radius: var(--radius-lg);
    text-decoration: none;
    color: var(--text);
    font-weight: 600;
  }
  .drawer-links a:hover,
  .drawer-links :global(a.router-link-active):hover,
  .drawer-links :global(a.router-link-exact-active):hover {
    background: var(--surface);
  }
  .drawer-ops {
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  /* Optionally hide on desktop when class present */
  @media (min-width: 920px) {
    .drawer--desktop-hidden { display: none !important; }
  }

  .fade-enter-active,
  .fade-leave-active { transition: opacity 0.12s ease; }
  .fade-enter-from,
  .fade-leave-to { opacity: 0; }
</style>
