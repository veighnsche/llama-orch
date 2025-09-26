import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import PublicTapView from '@/views/PublicTapView.vue'
import PrivateTapView from '@/views/PrivateTapView.vue'
import ToolkitView from '@/views/ToolkitView.vue'
import PricingView from '@/views/PricingView.vue'
import ProofView from '@/views/ProofView.vue'
import FAQsView from '@/views/FAQsView.vue'
import AboutView from '@/views/AboutView.vue'
import ContactLegalView from '@/views/ContactLegalView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    },
    {
      path: '/public-tap',
      name: 'public-tap',
      component: PublicTapView,
    },
    {
      path: '/private-tap',
      name: 'private-tap',
      component: PrivateTapView,
    },
    {
      path: '/toolkit',
      name: 'toolkit',
      component: ToolkitView,
    },
    {
      path: '/pricing',
      name: 'pricing',
      component: PricingView,
    },
    {
      path: '/proof',
      name: 'proof',
      component: ProofView,
    },
    {
      path: '/faqs',
      name: 'faqs',
      component: FAQsView,
    },
    {
      path: '/about',
      name: 'about',
      component: AboutView,
    },
    {
      path: '/contact',
      name: 'contact',
      component: ContactLegalView,
    },
  ],
})

export default router
