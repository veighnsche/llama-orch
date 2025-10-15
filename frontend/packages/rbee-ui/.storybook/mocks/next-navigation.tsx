// Mock Next.js navigation hooks for Storybook

export function useRouter() {
  return {
    push: (url: string) => console.log('Navigate to:', url),
    replace: (url: string) => console.log('Replace with:', url),
    back: () => console.log('Go back'),
    forward: () => console.log('Go forward'),
    refresh: () => console.log('Refresh'),
    prefetch: () => {},
    pathname: '/',
    query: {},
    asPath: '/',
  }
}

export function usePathname() {
  return '/'
}

export function useSearchParams() {
  return new URLSearchParams()
}
