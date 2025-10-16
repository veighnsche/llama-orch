/**
 * Global type declarations for the rbee-ui package
 */

/// <reference types="vite/client" />

// Image module declarations - Vite resolves these as URL strings
declare module '*.png' {
  const value: string
  export default value
}

declare module '*.jpg' {
  const value: string
  export default value
}

declare module '*.jpeg' {
  const value: string
  export default value
}

declare module '*.webp' {
  const value: string
  export default value
}

declare module '*.svg' {
  const value: string
  export default value
}
