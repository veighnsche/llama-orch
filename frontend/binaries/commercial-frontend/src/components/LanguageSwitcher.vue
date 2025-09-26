<template>
  <div class="lang-switcher">
    <label class="sr-only" for="lang">{{ $t('a11y.language') }}</label>
    <select id="lang" :value="locale" :aria-label="$t('a11y.language')" @change="onChange($event)">
      <option value="nl">NL</option>
      <option value="en">EN</option>
    </select>
  </div>
</template>

<script setup lang="ts">
  import { useI18n } from 'vue-i18n'
  import { watch } from 'vue'

  const { locale } = useI18n()

  function applyHtmlLang(l: string) {
    document.documentElement.setAttribute('lang', l)
  }

  function onChange(e: Event) {
    const target = e.target as HTMLSelectElement
    const next = target.value
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
  .lang-switcher select {
    padding: 0.25rem 0.5rem;
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
