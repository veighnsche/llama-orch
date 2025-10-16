/**
 * Global type declarations for the rbee-ui package
 */

import { StaticImageData } from 'next/image'

// Image module declarations
declare module '*.png' {
  const value: StaticImageData
  export default value
}

declare module '*.jpg' {
  const value: StaticImageData
  export default value
}

declare module '*.jpeg' {
  const value: StaticImageData
  export default value
}

declare module '*.webp' {
  const value: StaticImageData
  export default value
}

declare module '*.svg' {
  const value: StaticImageData
  export default value
}
