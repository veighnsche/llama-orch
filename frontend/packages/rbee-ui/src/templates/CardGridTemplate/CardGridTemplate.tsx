import type { ReactNode } from 'react'

export interface CardGridTemplateProps {
  cards: ReactNode[]
  columns?: 1 | 2 | 3 | 4
  gap?: 'sm' | 'md' | 'lg'
}

const gapClasses = {
  sm: 'gap-4',
  md: 'gap-6',
  lg: 'gap-8',
}

const columnClasses = {
  1: 'grid-cols-1',
  2: 'grid-cols-1 md:grid-cols-2',
  3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
  4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4',
}

export function CardGridTemplate({ cards, columns = 2, gap = 'md' }: CardGridTemplateProps) {
  return <div className={`grid ${columnClasses[columns]} ${gapClasses[gap]}`}>{cards}</div>
}
