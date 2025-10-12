// https://nuxt.com/docs/api/configuration/nuxt-config
import { defineNuxtConfig } from "nuxt/config";
import tailwindcss from "@tailwindcss/vite";

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║                                                                          ║
// ║  All UI components are now local to this project in ~/app/stories       ║
// ║  Design tokens and styles are in ~/app/styles                           ║
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
  ],

  vite: {
    plugins: [
      tailwindcss(),
    ],
  },
});
