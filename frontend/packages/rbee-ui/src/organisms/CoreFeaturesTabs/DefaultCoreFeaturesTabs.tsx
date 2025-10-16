import { CoreFeaturesTabs } from './CoreFeaturesTabs'
import { defaultTabConfigs } from './tabConfigs'

export function DefaultCoreFeaturesTabs() {
  return (
    <CoreFeaturesTabs
      title="Core capabilities"
      description="Swap in the API, scale across your hardware, route with code, and watch jobs stream in real time."
      tabs={defaultTabConfigs}
      defaultTab="api"
    />
  )
}
