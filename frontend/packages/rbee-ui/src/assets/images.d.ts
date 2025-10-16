/**
 * Type declarations for image imports
 * Using Next.js StaticImageData for raster images
 */

import { StaticImageData } from 'next/image'

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
