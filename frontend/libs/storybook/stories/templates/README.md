# View Templates for Commercial Frontend

**Created by:** TEAM-FE-009  
**Date:** 2025-10-11

## What's Here

This directory contains **pre-built Vue view templates** that assemble components from the storybook into complete pages. All the hard work is done - you just need to wire them up!

**Location:** `/frontend/libs/storybook/stories/templates/`  
**Stories:** Available in storybook under `templates/` category

## Available Views

- `HomeView.vue` - Main landing page
- `DevelopersView.vue` - Developer-focused page
- `EnterpriseView.vue` - Enterprise solutions page
- `FeaturesView.vue` - Features showcase
- `PricingView.vue` - Pricing plans
- `ProvidersView.vue` - AI providers page
- `UseCasesView.vue` - Use cases page

## How Each View Works

Each view imports pre-built components from `rbee-storybook/stories` and assembles them into a page layout. Example:

```vue
<script setup lang="ts">
import {
  HeroSection,
  WhatIsRbee,
  EmailCapture,
  Footer,
} from 'rbee-storybook/stories'
</script>

<template>
  <main class="min-h-screen pt-16">
    <HeroSection />
    <WhatIsRbee />
    <EmailCapture />
    <Footer />
  </main>
</template>
```

**That's it!** No styling needed - components are pre-styled.

## Assignment for Junior Developer

### Task: Rebuild Commercial Frontend

**Difficulty:** ⭐ Easy  
**Time Estimate:** 30 minutes  
**Prerequisites:** Basic Vue.js knowledge

### Step 1: Create New Vue Project

```bash
cd frontend/bin
pnpm create vite@latest commercial-frontend -- --template vue-ts
cd commercial-frontend
```

### Step 2: Install Dependencies

```bash
pnpm add rbee-storybook@workspace:* vue-router
pnpm add -D tailwindcss @tailwindcss/postcss postcss
```

### Step 3: Configure Tailwind CSS

Create `postcss.config.js`:
```javascript
export default {
  plugins: {
    '@tailwindcss/postcss': {},
  },
}
```

Create `tailwind.config.js`:
```javascript
export default {
  content: [
    './index.html',
    './src/**/*.{vue,js,ts,jsx,tsx}',
    '../../libs/storybook/stories/**/*.{vue,js,ts,jsx,tsx}',
  ],
}
```

Update `src/assets/main.css`:
```css
@import 'tailwindcss';

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    --primary: 38 92% 50%;
    --primary-foreground: 0 0% 100%;
    --radius: 0.5rem;
  }
  
  body {
    background-color: hsl(var(--background));
    color: hsl(var(--foreground));
  }
}
```

### Step 4: Copy View Templates

```bash
# Copy all view templates to your src/views directory
cp ../../libs/storybook/stories/templates/*.vue src/views/
# Exclude .story.vue files
rm src/views/*.story.vue
```

### Step 5: Set Up Router

Create `src/router/index.ts`:
```typescript
import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', name: 'home', component: HomeView },
    { path: '/developers', name: 'developers', component: () => import('../views/DevelopersView.vue') },
    { path: '/enterprise', name: 'enterprise', component: () => import('../views/EnterpriseView.vue') },
    { path: '/features', name: 'features', component: () => import('../views/FeaturesView.vue') },
    { path: '/pricing', name: 'pricing', component: () => import('../views/PricingView.vue') },
    { path: '/providers', name: 'providers', component: () => import('../views/ProvidersView.vue') },
    { path: '/use-cases', name: 'use-cases', component: () => import('../views/UseCasesView.vue') },
  ],
})

export default router
```

Update `src/main.ts`:
```typescript
import './assets/main.css'
import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

const app = createApp(App)
app.use(router)
app.mount('#app')
```

Update `src/App.vue`:
```vue
<script setup lang="ts">
import { RouterView } from 'vue-router'
</script>

<template>
  <RouterView />
</template>
```

### Step 6: Run Dev Server

```bash
pnpm run dev
```

Open http://localhost:5173/ - **Done!** 🎉

## Verification Checklist

- [ ] All 7 pages load without errors
- [ ] Tailwind CSS styles are applied (orange buttons, proper spacing)
- [ ] Navigation works between pages
- [ ] Components from storybook render correctly
- [ ] No console errors in browser DevTools

## Troubleshooting

### "Cannot find module 'rbee-storybook/stories'"

**Solution:** Run `pnpm install` in the frontend root to set up workspace links.

### "Styles not loading"

**Solution:** Make sure you have:
1. `postcss.config.js` with `@tailwindcss/postcss`
2. `tailwind.config.js` with correct content paths
3. `@import 'tailwindcss'` in `main.css`

### "Component not found"

**Solution:** Check that the component is exported from `rbee-storybook/stories/index.ts`

## What You DON'T Need to Do

❌ **Don't** create any new components - they're all in the storybook  
❌ **Don't** write any CSS - Tailwind handles everything  
❌ **Don't** modify the view templates - they're ready to use  
❌ **Don't** add complex state management - views are stateless

## What You DO Need to Do

✅ **Do** follow the setup steps exactly  
✅ **Do** copy the view templates as-is  
✅ **Do** set up Vue Router with the provided routes  
✅ **Do** test all pages work correctly

## Success Criteria

Your assignment is complete when:
1. All 7 pages render correctly
2. Tailwind CSS styles are working
3. No errors in console
4. You can navigate between pages

**Expected time:** 30 minutes  
**If stuck:** Check the troubleshooting section above

Good luck! 🚀
