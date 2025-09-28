export interface AppState {
  pipelines: string[]
  seed: number | null
}

export const appState: AppState = {
  pipelines: ['public','private'],
  seed: 424242,
}
