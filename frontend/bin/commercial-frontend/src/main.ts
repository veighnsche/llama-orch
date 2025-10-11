// Created by: TEAM-FE-000
// TEAM-FE-009: Removed CSS import - now handled via @import in main.css per Tailwind v4 docs
import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

const app = createApp(App)

app.use(router)

app.mount('#app')
