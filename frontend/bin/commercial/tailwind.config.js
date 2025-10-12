// TEAM-FE-011: Tailwind config to scan storybook package for classes
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './app/**/*.{vue,js,ts,jsx,tsx}',
    './components/**/*.{vue,js,ts,jsx,tsx}',
    './layouts/**/*.{vue,js,ts,jsx,tsx}',
    './pages/**/*.{vue,js,ts,jsx,tsx}',
    './plugins/**/*.{js,ts}',
    // Scan the storybook package for Tailwind classes
    '../../libs/storybook/stories/**/*.{vue,js,ts,jsx,tsx}',
  ],
}
