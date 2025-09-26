import { createI18n } from 'vue-i18n'
import en from './en.json'
import nl from './nl.json'

const messages = {
  en,
  nl,
}

const stored = typeof localStorage !== 'undefined' ? localStorage.getItem('orchyra_locale') : null
const defaultLocale = stored || 'nl'
export const i18n = createI18n({
  legacy: false,
  globalInjection: true,
  locale: defaultLocale,
  fallbackLocale: 'en',
  messages,
})

export default i18n
