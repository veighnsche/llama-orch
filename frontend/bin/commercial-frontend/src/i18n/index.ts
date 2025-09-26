import { createI18n } from 'vue-i18n'
import en from './en_2'
import nl from './nl'

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
