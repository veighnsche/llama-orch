// Created by: TEAM-FE-000 (Scaffolding)
// TEAM-FE-001: Implemented Alert story with all variants

import Alert from './Alert.vue'
import AlertTitle from './AlertTitle.vue'
import AlertDescription from './AlertDescription.vue'

export default {
  title: 'atoms/Alert',
  component: Alert,
}

export const Default = () => ({
  components: { Alert, AlertTitle, AlertDescription },
  template: `
    <Alert>
      <AlertTitle>Default Alert</AlertTitle>
      <AlertDescription>This is a default alert message.</AlertDescription>
    </Alert>
  `,
})

export const Destructive = () => ({
  components: { Alert, AlertTitle, AlertDescription },
  template: `
    <Alert variant="destructive">
      <AlertTitle>Error</AlertTitle>
      <AlertDescription>Something went wrong. Please try again.</AlertDescription>
    </Alert>
  `,
})

export const WithIcon = () => ({
  components: { Alert, AlertTitle, AlertDescription },
  template: `
    <Alert>
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="16" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12.01" y2="8"></line>
      </svg>
      <AlertTitle>Heads up!</AlertTitle>
      <AlertDescription>You can add icons to alerts for better visual communication.</AlertDescription>
    </Alert>
  `,
})

export const DestructiveWithIcon = () => ({
  components: { Alert, AlertTitle, AlertDescription },
  template: `
    <Alert variant="destructive">
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
      </svg>
      <AlertTitle>Error</AlertTitle>
      <AlertDescription>Your session has expired. Please log in again.</AlertDescription>
    </Alert>
  `,
})

export const AllVariants = () => ({
  components: { Alert, AlertTitle, AlertDescription },
  template: `
    <div style="display: flex; flex-direction: column; gap: 16px;">
      <Alert>
        <AlertTitle>Default Alert</AlertTitle>
        <AlertDescription>This is a default informational alert.</AlertDescription>
      </Alert>

      <Alert variant="destructive">
        <AlertTitle>Destructive Alert</AlertTitle>
        <AlertDescription>This is an error or destructive alert.</AlertDescription>
      </Alert>

      <Alert>
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="16" x2="12" y2="12"></line>
          <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
        <AlertTitle>With Icon</AlertTitle>
        <AlertDescription>Alert with an icon for better visual communication.</AlertDescription>
      </Alert>

      <Alert variant="destructive">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="15" y1="9" x2="9" y2="15"></line>
          <line x1="9" y1="9" x2="15" y2="15"></line>
        </svg>
        <AlertTitle>Destructive with Icon</AlertTitle>
        <AlertDescription>Error alert with an icon.</AlertDescription>
      </Alert>
    </div>
  `,
})
