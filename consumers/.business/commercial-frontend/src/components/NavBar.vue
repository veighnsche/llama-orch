<template>
  <header class="nav" role="banner">
    <nav class="nav-inner" :aria-label="$t('a11y.navPrimary')">
      <!-- Brand -->
      <RouterLink class="brand" to="/" aria-label="Orchyra — home">
        <span class="brand-glyph">
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="2"/>
            <path d="M12 6v12M6 12h12M8.7 8.7l6.6 6.6M15.3 8.7l-6.6 6.6"
                  stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
            <path d="M2 12h3M19 12h3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            <circle cx="12" cy="12" r="2.2" fill="currentColor"/>
          </svg>
        </span>
        <span class="brand-word">Orchyra</span>
      </RouterLink>

      <!-- Mobile toggle -->
      <button
        class="menu-toggle"
        :aria-expanded="open ? 'true' : 'false'"
        :aria-label="open ? $t('a11y.closeMenu') : $t('a11y.openMenu')"
        @click="open = !open"
      >
        <span class="sr-only">Menu</span>
        <svg v-if="!open" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
        <svg v-else viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6l-12 12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
      </button>

      <!-- Desktop links -->
      <ul class="links" role="menubar">
        <li role="none"><RouterLink role="menuitem" to="/public-tap">{{ $t('nav.publicTap') }}</RouterLink></li>
        <li role="none"><RouterLink role="menuitem" to="/private-tap">{{ $t('nav.privateTap') }}</RouterLink></li>
        <li role="none"><RouterLink role="menuitem" to="/pricing">{{ $t('nav.pricing') }}</RouterLink></li>
        <li role="none"><RouterLink role="menuitem" to="/proof">{{ $t('nav.proof') }}</RouterLink></li>
        <li role="none"><RouterLink role="menuitem" to="/faqs">{{ $t('nav.faqs') }}</RouterLink></li>
        <li role="none"><RouterLink role="menuitem" to="/about">{{ $t('nav.about') }}</RouterLink></li>
        <li role="none"><RouterLink role="menuitem" to="/contact">{{ $t('nav.contact') }}</RouterLink></li>
      </ul>

      <!-- Right side: language + CTA -->
      <div class="right">
        <LanguageSwitcher />
        <Button as="router-link" to="/service-menu" variant="primary">
          {{ $t('nav.serviceMenu', 'Service menu') }}
        </Button>
      </div>
    </nav>

    <!-- Mobile drawer -->
    <transition name="fade">
      <div v-if="open" class="drawer" @click.self="open=false">
        <div class="drawer-panel">
          <ul class="drawer-links">
            <li><RouterLink @click="open=false" to="/public-tap">{{ $t('nav.publicTap') }}</RouterLink></li>
            <li><RouterLink @click="open=false" to="/private-tap">{{ $t('nav.privateTap') }}</RouterLink></li>
            <li><RouterLink @click="open=false" to="/pricing">{{ $t('nav.pricing') }}</RouterLink></li>
            <li><RouterLink @click="open=false" to="/proof">{{ $t('nav.proof') }}</RouterLink></li>
            <li><RouterLink @click="open=false" to="/faqs">{{ $t('nav.faqs') }}</RouterLink></li>
            <li><RouterLink @click="open=false" to="/about">{{ $t('nav.about') }}</RouterLink></li>
            <li><RouterLink @click="open=false" to="/contact">{{ $t('nav.contact') }}</RouterLink></li>
          </ul>
          <div class="drawer-ops">
            <LanguageSwitcher />
            <Button class="wide" as="router-link" @click="open=false" to="/service-menu" variant="primary" size="sm">
              {{ $t('nav.serviceMenu', 'Service menu') }}
            </Button>
          </div>
        </div>
      </div>
    </transition>
  </header>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useRoute, RouterLink } from 'vue-router'
import LanguageSwitcher from '@/components/LanguageSwitcher.vue'
import Button from 'orchyra-storybook/stories/button/index.vue'

const open = ref(false)
const route = useRoute()
// close the drawer on route change
watch(() => route.fullPath, () => { open.value = false })
</script>

<style scoped>
/* unify top-bar rhythm */
:root { --nav-h: 44px; }

/* ---- layout ---- */
.nav {
  position: sticky;
  top: 0;
  z-index: 50;
  backdrop-filter: saturate(140%) blur(6px);
  background: var(--surface-alt);
  border-bottom: 1px solid var(--surface-muted);
}
.nav-inner {
  max-width: 1120px;
  margin: 0 auto;
  padding: .6rem 1rem;
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: .75rem 1rem;
  align-items: center;
  /* a touch more horizontal air between brand and links */
  column-gap: 1.25rem;
  padding-block: 0; /* height controlled by items */
}

/* ---- brand ---- */
.brand {
  display: inline-flex;
  align-items: center;
  gap: .55rem;
  padding: 0 .5rem;              /* horizontal padding = better hitbox */
  height: var(--nav-h);          /* matches link height */
  border-radius: var(--radius-lg);
  text-decoration: none;
  color: var(--text);            /* from tokens */
  font-weight: 800;
  letter-spacing: .2px;
}
.brand:hover { text-decoration: none; }

.brand-mark {
  width: 1.25rem;            /* ~20px; scales with font if you prefer: 1em */
  height: 1.25rem;
  color: var(--acc-cyan);    /* accent for the mark */
  flex: 0 0 auto;
  translate: 0 .5px;         /* tiny baseline nudge for optical alignment */
}

/* icon gets its own padded box so the glyph has breathing room */
.brand-glyph {
  width: 28px;                   /* outer box */
  height: 28px;
  display: grid;
  place-items: center;
  border-radius: var(--radius-md);
  background: var(--surface);
  border: 1px solid var(--surface-muted);
  color: var(--acc-cyan);        /* cyan accent drives the svg via currentColor */
}
.brand-glyph svg { width: 18px; height: 18px; } /* inner padding = visual comfort */

.brand-word { font-size: 1.06rem; }

/* dark mode fine-tune if you keep a dark bar */
/* Dark mode relies on token overrides in tokens.css; no extra rules needed here. */

/* ---- desktop links ---- */
.links {
  display: none;
  align-items: center;
  gap: 1rem;
  list-style: none;
  margin: 0;
  padding: 0;
}
/* links: match height with brand so the left side aligns visually */
.links a {
  display: inline-flex;
  align-items: center;
  height: var(--nav-h);
  padding: 0 .5rem;
  border-radius: var(--radius-lg);
  color: var(--muted);
  font-weight: 600;
  text-decoration: none;
}
.links a:hover { background: var(--surface); }

/* ---- right cluster ---- */
.right {
  display: inline-flex;
  align-items: center;
  gap: .6rem;
}
/* drawer button stretch */
.drawer-ops :deep(.ui-btn.wide) { flex: 1; text-align: center; }

/* ---- mobile toggle ---- */
.menu-toggle {
  appearance: none;
  border: 1px solid var(--surface-muted);
  background: var(--surface-alt);
  border-radius: var(--radius-lg);
  padding: .4rem .5rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 38px;
  height: 38px;
  color: var(--text);
}
.menu-toggle:focus-visible { outline: 4px solid var(--ring); outline-offset: 2px; }
.menu-toggle svg { width: 22px; height: 22px; }

/* ---- drawer (mobile) ---- */
.drawer {
  position: fixed;
  inset: 0;
  background: color-mix(in srgb, var(--text) 42%, transparent);
  display: none;
}
.drawer-panel {
  margin: .5rem;
  border-radius: var(--radius-xl);
  background: var(--surface-alt);
  border: 1px solid var(--surface-muted);
  box-shadow: var(--shadow-lg);
  padding: .75rem;
}
.drawer-links {
  list-style: none;
  margin: 0;
  padding: .25rem 0 .5rem 0;
  display: grid;
  gap: .25rem;
}
.drawer-links a {
  display: block;
  padding: .6rem .6rem;
  border-radius: var(--radius-lg);
  text-decoration: none;
  color: var(--text);
  font-weight: 600;
}
.drawer-links a:hover { background: var(--surface); }
.drawer-links :global(.router-link-active) { background: var(--surface); }
.drawer-ops {
  margin-top: .5rem;
  display: flex;
  align-items: center;
  gap: .5rem;
}
/* handled above via :deep(.ui-btn.wide) */

/* ---- transitions ---- */
.fade-enter-active, .fade-leave-active { transition: opacity .12s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

/* ---- responsive ---- */
@media (min-width: 920px) {
  .menu-toggle { display: none; }
  .links { display: inline-flex; }
  .drawer { display: none !important; }
}
@media (max-width: 919.98px) {
  .drawer { display: block; }
  .right :deep(.ui-btn) { display: none; } /* keep topbar tight on small screens; CTA moves into drawer */
}
/* Dark mode token overrides cover the drawer and panel as well. */
.sr-only {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden;
  clip: rect(0,0,0,0); white-space: nowrap; border: 0;
}
</style>
