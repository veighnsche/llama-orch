import type { Metadata } from "next";
// TEAM-288: Import order matches commercial app - app CSS first, then UI CSS
import "./globals.css";
import "@rbee/ui/styles.css";
import { ThemeProvider } from "next-themes";
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@rbee/ui/atoms";
import { AppSidebar } from "@/src/components/AppSidebar";

export const metadata: Metadata = {
  title: "rbee Web UI",
  description: "Dashboard for managing rbee infrastructure",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <SidebarProvider>
            <AppSidebar />
            <SidebarInset>
              <header className="flex h-16 shrink-0 items-center gap-2 border-b border-border px-4">
                <SidebarTrigger />
              </header>
              <div className="flex flex-1 flex-col gap-4 p-4">{children}</div>
            </SidebarInset>
          </SidebarProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
