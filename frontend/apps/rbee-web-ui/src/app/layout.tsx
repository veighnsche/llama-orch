import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import '@rbee/ui/styles/globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'rbee Web UI',
  description: 'Dashboard for managing rbee infrastructure',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>{children}</body>
    </html>
  );
}
