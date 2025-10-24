// TEAM-291: Application sidebar with navigation
// Inspired by generic_ai_market's collapsible drawer pattern

"use client";

import {
  HomeIcon,
  TerminalIcon,
  SettingsIcon,
  HelpCircleIcon,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
} from "@rbee/ui/atoms";

export function AppSidebar() {
  const pathname = usePathname();

  const mainNavigation = [
    {
      title: "Dashboard",
      href: "/dashboard",
      icon: HomeIcon,
      tooltip: "View dashboard",
    },
    {
      title: "Bee Keeper",
      href: "/keeper",
      icon: TerminalIcon,
      tooltip: "CLI operations",
    },
  ];

  const secondaryNavigation = [
    {
      title: "Settings",
      href: "/settings",
      icon: SettingsIcon,
      tooltip: "Configuration",
    },
    {
      title: "Help",
      href: "/help",
      icon: HelpCircleIcon,
      tooltip: "Documentation",
    },
  ];

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="border-b border-sidebar-border p-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🐝</span>
          <span className="font-semibold">rbee</span>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Main</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {mainNavigation.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton
                    asChild
                    isActive={pathname === item.href}
                    tooltip={item.tooltip}
                  >
                    <Link href={item.href}>
                      <item.icon />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarSeparator />
        <SidebarGroup>
          <SidebarGroupLabel>System</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {secondaryNavigation.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton
                    asChild
                    isActive={pathname === item.href}
                    tooltip={item.tooltip}
                  >
                    <Link href={item.href}>
                      <item.icon />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="border-t border-sidebar-border p-2">
        <div className="text-xs text-muted-foreground px-2 py-1">
          <span className="font-mono">v0.1.0</span>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
