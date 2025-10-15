import { redirect } from 'next/navigation'

export default function Home() {
  // Redirect to docs page
  redirect('/docs')
}
