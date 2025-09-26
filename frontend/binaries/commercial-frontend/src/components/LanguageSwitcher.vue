<template>
  <div class="lang-switcher" role="group" :aria-label="$t('a11y.language')">
    <button
      type="button"
      class="lang-btn"
      :class="{ active: locale === 'nl' }"
      :aria-pressed="locale === 'nl'"
      @click="setLang('nl')"
    >
      NL
    </button>
    <span class="sep" aria-hidden="true">|</span>
    <button
      type="button"
      class="lang-btn"
      :class="{ active: locale === 'en' }"
      :aria-pressed="locale === 'en'"
      @click="setLang('en')"
    >
      EN
    </button>
  </div>
</template>

<script setup lang="ts">
  import { useI18n } from 'vue-i18n'
  import { watch } from 'vue'

  const { locale } = useI18n()

  function applyHtmlLang(l: string) {
    document.documentElement.setAttribute('lang', l)
  }

  function setLang(next: 'nl' | 'en') {
    locale.value = next
    localStorage.setItem('orchyra_locale', next)
    applyHtmlLang(next)
  }

  // restore persisted locale
  const persisted = localStorage.getItem('orchyra_locale')
  if (persisted) {
    locale.value = persisted
    applyHtmlLang(persisted)
  } else {
    applyHtmlLang(locale.value)
  }

  watch(locale, (l) => applyHtmlLang(l))
</script>

<style scoped>
  .lang-switcher {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    margin-inline-end: 1.5rem; /* create space before the CTA button */
  }
  .lang-btn {
    appearance: none;
    background: none;
    border: none;
    padding: 0;
    margin: 0;
    font: inherit;
    color: inherit;
    cursor: pointer;
    line-height: 1;
  }
  .lang-btn.active {
    text-decoration: underline;
    font-weight: 600;
  }
  .sep {
    padding: 0 0.25rem;
  }
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
</style>
