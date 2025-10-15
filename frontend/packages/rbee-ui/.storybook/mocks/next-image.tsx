import type * as React from 'react'

// Mock Next.js Image component for Storybook
export default function Image({
  src,
  alt,
  width,
  height,
  priority,
  className,
  style,
  ...props
}: React.ImgHTMLAttributes<HTMLImageElement> & {
  src: string
  alt: string
  width?: number
  height?: number
  priority?: boolean
}) {
  return <img src={src} alt={alt} width={width} height={height} className={className} style={style} {...props} />
}
