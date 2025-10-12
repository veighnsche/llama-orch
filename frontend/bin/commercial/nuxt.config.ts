// https://nuxt.com/docs/api/configuration/nuxt-config
import { defineNuxtConfig } from "nuxt/config";
import tailwindcss from "@tailwindcss/vite";

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║                                                                          ║
// ║  ⛔ FORBIDDEN: NO CROSS-PROJECT IMPORTS IN VITE/TAILWIND CONFIG ⛔      ║
// ║                                                                          ║
// ║  NEVER use relative paths like "../../libs/storybook/..."               ║
// ║  ALWAYS use workspace package names like "rbee-storybook/..."           ║
// ║                                                                          ║
// ║  If an export is missing, fix it in the package's package.json exports  ║
// ║  Cross-project imports have NEVER worked and NEVER will work.           ║
// ║                                                                          ║
// ╚══════════════════════════════════════════════════════════════════════════╝

export default defineNuxtConfig({
  compatibilityDate: "2025-07-15",
  devtools: { enabled: true },

  nitro: {
    preset: "cloudflare_module",

    cloudflare: {
      deployConfig: true,
    },
  },

  modules: ["nitro-cloudflare-dev"],

  css: [
    "~/assets/css/main.css",
    "rbee-storybook/styles/tokens-base.css",
  ],

  vite: {
    plugins: [
      tailwindcss(),
      // ⛔ DO NOT configure content paths here
      // Tailwind v4 scans imported components automatically
    ],
  },
});
