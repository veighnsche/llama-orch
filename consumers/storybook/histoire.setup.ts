import { defineSetupVue3 } from '@histoire/plugin-vue'
import { createRouter, createMemoryHistory } from 'vue-router'
import './styles/tokens.css'

export const setupVue3 = defineSetupVue3(({ app }) => {
    const router = createRouter({
        history: createMemoryHistory(),
        routes: [
            { path: '/', component: { template: '<div />' } },
            { path: '/about', component: { template: '<div>About</div>' } },
            { path: '/contact', component: { template: '<div>Contact</div>' } },
        ],
    })
    app.use(router)
})
