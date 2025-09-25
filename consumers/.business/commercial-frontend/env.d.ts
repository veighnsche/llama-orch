/// <reference types="vite/client" />

declare module '*.vue' {
    import type { DefineComponent } from 'vue'
    // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/ban-types
    const component: DefineComponent<{}, {}, any>
    export default component
}

interface ImportMetaEnv {
    readonly VITE_SITE_URL?: string
    readonly VITE_GITHUB_URL?: string
    readonly VITE_CONTACT_EMAIL?: string
    readonly VITE_LINKEDIN_URL?: string
}

interface ImportMeta {
    readonly env: ImportMetaEnv
}
