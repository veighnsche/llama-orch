// Created by: TEAM-FE-000
// TEAM-FE-002: Added pricing route
// TEAM-FE-009: Added all page routes
import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    },
    {
      path: '/developers',
      name: 'developers',
      component: () => import('../views/DevelopersView.vue'),
    },
    {
      path: '/enterprise',
      name: 'enterprise',
      component: () => import('../views/EnterpriseView.vue'),
    },
    {
      path: '/gpu-providers',
      name: 'gpu-providers',
      component: () => import('../views/ProvidersView.vue'),
    },
    {
      path: '/features',
      name: 'features',
      component: () => import('../views/FeaturesView.vue'),
    },
    {
      path: '/use-cases',
      name: 'use-cases',
      component: () => import('../views/UseCasesView.vue'),
    },
    {
      path: '/pricing',
      name: 'pricing',
      component: () => import('../views/PricingView.vue'),
    },
  ],
})

export default router
