// TEAM-292: Application sidebar with navigation
// Ported from web-ui.old - React Router instead of Next.js

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
} from '@rbee/ui/atoms'
import { ThemeToggle } from '@rbee/ui/molecules'
import { HelpCircleIcon, HomeIcon, SettingsIcon, TerminalIcon } from 'lucide-react'
import { Link, useLocation } from 'react-router-dom'

export function AppSidebar() {
  const location = useLocation()

  const mainNavigation = [
    {
      title: 'Dashboard',
      href: '/dashboard',
      icon: HomeIcon,
      tooltip: 'View dashboard',
    },
    {
      title: 'Bee Keeper',
      href: '/keeper',
      icon: TerminalIcon,
      tooltip: 'CLI operations',
    },
  ]

  const secondaryNavigation = [
    {
      title: 'Settings',
      href: '/settings',
      icon: SettingsIcon,
      tooltip: 'Configuration',
    },
    {
      title: 'Help',
      href: '/help',
      icon: HelpCircleIcon,
      tooltip: 'Documentation',
    },
  ]

  return (
    <Sidebar collapsible="none" className="border-r border-border h-screen">
      <SidebarHeader className="p-4">
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
                  <SidebarMenuButton asChild isActive={location.pathname === item.href} tooltip={item.tooltip}>
                    <Link to={item.href}>
                      <item.icon />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup>
          <SidebarGroupLabel>System</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {secondaryNavigation.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton asChild isActive={location.pathname === item.href} tooltip={item.tooltip}>
                    <Link to={item.href}>
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
      <SidebarFooter className="mt-auto p-4">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground font-mono">v0.1.0</span>
          <ThemeToggle />
        </div>
      </SidebarFooter>
    </Sidebar>
  )
}
