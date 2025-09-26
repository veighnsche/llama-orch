import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import i18n from './i18n'
import { initThemeFromStorage } from './composables/useTheme'

// Ensure theme classes/attributes are applied on hydration
initThemeFromStorage()

const app = createApp(App)

app.use(router)
app.use(i18n)

app.mount('#app')
