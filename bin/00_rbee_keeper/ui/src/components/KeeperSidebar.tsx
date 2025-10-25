// TEAM-295: Navigation sidebar for Bee Keeper app
// Based on AppSidebar from queen-rbee UI

import { HomeIcon, SettingsIcon, HelpCircleIcon } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
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
} from "@rbee/ui/atoms";
import { BrandLogo, ThemeToggle } from "@rbee/ui/molecules";

export function KeeperSidebar() {
  const location = useLocation();

  const mainNavigation = [
    {
      title: "Services",
      href: "/",
      icon: HomeIcon,
      tooltip: "Manage services",
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
    <Sidebar collapsible="none" className="border-r border-border h-screen">
      <SidebarHeader className="p-4">
        <Link to="/">
          <BrandLogo size="md" />
        </Link>
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
                    isActive={location.pathname === item.href}
                    tooltip={item.tooltip}
                  >
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
                  <SidebarMenuButton
                    asChild
                    isActive={location.pathname === item.href}
                    tooltip={item.tooltip}
                  >
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
          <span className="text-xs text-muted-foreground font-mono">
            v0.1.0
          </span>
          <ThemeToggle />
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
