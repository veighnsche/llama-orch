// TEAM-295: Navigation sidebar for Bee Keeper app
// Based on AppSidebar from queen-rbee UI
// TEAM-339: Simplified to work with react-resizable-panels (removed fixed-width Sidebar wrapper)
// TEAM-340: Added Queen navigation item with iframe page
// TEAM-342: Added Hives section with dynamic navigation to installed hives

import { ThemeToggle } from "@rbee/ui/molecules";
import {
  CrownIcon,
  HelpCircleIcon,
  HomeIcon,
  ServerIcon,
  SettingsIcon,
} from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { useSshHivesStore } from "@/store/hiveStore";

export function KeeperSidebar() {
  const location = useLocation();
  const { hives, installedHives } = useSshHivesStore();

  // TEAM-342: Filter hives to only show installed ones
  const installedHivesList = hives.filter((hive) =>
    installedHives.includes(hive.host),
  );

  const mainNavigation = [
    {
      title: "Services",
      href: "/",
      icon: HomeIcon,
      tooltip: "Manage services",
    },
    {
      title: "Queen",
      href: "/queen",
      icon: CrownIcon,
      tooltip: "Queen web interface",
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
    <div className="h-full w-full flex flex-col border-r border-border bg-background">
      {/* Main navigation */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-6">
          {/* Main section */}
          <div className="space-y-2">
            <h3 className="text-xs font-semibold font-serif text-muted-foreground uppercase tracking-wider px-2">
              Main
            </h3>
            <nav className="space-y-1">
              {mainNavigation.map((item) => {
                const isActive = location.pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    to={item.href}
                    title={item.tooltip}
                    className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      isActive
                        ? "bg-primary text-primary-foreground"
                        : "text-foreground hover:bg-muted"
                    }`}
                  >
                    <item.icon className="w-4 h-4 flex-shrink-0" />
                    <span className="truncate">{item.title}</span>
                  </Link>
                );
              })}
            </nav>
          </div>

          {/* TEAM-342: Hives section */}
          <div className="space-y-2">
            <h3 className="text-xs font-semibold font-serif text-muted-foreground uppercase tracking-wider px-2">
              Hives
            </h3>
            {installedHivesList.length === 0 ? (
              <p className="text-xs text-muted-foreground px-2 py-2">
                No hives installed yet
              </p>
            ) : (
              <nav className="space-y-1">
                {installedHivesList.map((hive) => {
                  const href = `/hive/${hive.host}`;
                  const isActive = location.pathname === href;
                  return (
                    <Link
                      key={hive.host}
                      to={href}
                      title={`${hive.user}@${hive.hostname}:${hive.port}`}
                      className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                        isActive
                          ? "bg-primary text-primary-foreground"
                          : "text-foreground hover:bg-muted"
                      }`}
                    >
                      <ServerIcon className="w-4 h-4 flex-shrink-0" />
                      <span className="truncate">{hive.host}</span>
                    </Link>
                  );
                })}
              </nav>
            )}
          </div>
        </div>
      </div>

      {/* System section */}
      <div className="border-t border-border p-4">
        <div className="space-y-2">
          <h3 className="text-xs font-semibold font-serif text-muted-foreground uppercase tracking-wider px-2">
            System
          </h3>
          <nav className="space-y-1">
            {secondaryNavigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.href}
                  to={item.href}
                  title={item.tooltip}
                  className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "text-foreground hover:bg-muted"
                  }`}
                >
                  <item.icon className="w-4 h-4 flex-shrink-0" />
                  <span className="truncate">{item.title}</span>
                </Link>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-border p-4">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground font-mono">
            v0.1.0
          </span>
          <ThemeToggle />
        </div>
      </div>
    </div>
  );
}
