export interface RunProgress {
  ts: string
  level: string
  event: string
  message?: string
}

export const runStore = {
  items: [] as RunProgress[],
}
