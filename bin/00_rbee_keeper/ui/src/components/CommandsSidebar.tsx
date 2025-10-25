// TEAM-292: Commands sidebar for Bee Keeper page
// Ported from web-ui.old
// TEAM-294: Added ThemeToggle at bottom and collapsible "See more" sections

import { useState } from "react";
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
import { ThemeToggle } from "./ThemeToggle";
import { ChevronDown, ChevronRight } from "lucide-react";

interface CommandItemProps {
  commandId: string;
  label: string;
}

interface CommandsSidebarProps {
  onCommandClick: (command: string) => void;
  activeCommand?: string;
  disabled?: boolean;
}

export function CommandsSidebar({
  onCommandClick,
  activeCommand,
  disabled = false,
}: CommandsSidebarProps) {
  const [queenExpanded, setQueenExpanded] = useState(false);
  const [hiveExpanded, setHiveExpanded] = useState(false);

  function CommandItem({ commandId, label }: CommandItemProps) {
    return (
      <SidebarMenuItem>
        <SidebarMenuButton
          onClick={() => onCommandClick(commandId)}
          isActive={activeCommand === commandId}
          disabled={disabled}
        >
          {label}
        </SidebarMenuButton>
      </SidebarMenuItem>
    );
  }

  function SeeMoreButton({
    expanded,
    onClick,
  }: {
    expanded: boolean;
    onClick: () => void;
  }) {
    return (
      <SidebarMenuItem>
        <SidebarMenuButton
          onClick={onClick}
          className="text-xs text-muted-foreground hover:text-foreground"
        >
          {expanded ? (
            <>
              <ChevronDown className="h-3 w-3 mr-1" />
              See less
            </>
          ) : (
            <>
              <ChevronRight className="h-3 w-3 mr-1" />
              See more
            </>
          )}
        </SidebarMenuButton>
      </SidebarMenuItem>
    );
  }

  return (
    <Sidebar collapsible="none" className="border-r border-border">
      <SidebarHeader className="p-4">
        <div>
          <h2 className="font-semibold">Commands</h2>
          <p className="text-xs text-muted-foreground mt-1">CLI operations</p>
        </div>
      </SidebarHeader>
      <SidebarContent>
        {/* Queen Operations */}
        <SidebarGroup>
          <SidebarGroupLabel>Queen</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <CommandItem commandId="queen-start" label="Start" />
              <CommandItem commandId="queen-stop" label="Stop" />
              <SeeMoreButton
                expanded={queenExpanded}
                onClick={() => setQueenExpanded(!queenExpanded)}
              />
              {queenExpanded && (
                <>
                  <CommandItem commandId="queen-status" label="Status" />
                  <CommandItem commandId="queen-info" label="Info" />
                  <CommandItem commandId="queen-rebuild" label="Rebuild" />
                </>
              )}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Hive Operations (localhost) */}
        <SidebarGroup>
          <SidebarGroupLabel>Hive (localhost)</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <CommandItem commandId="hive-start" label="Start" />
              <CommandItem commandId="hive-stop" label="Stop" />
              <SeeMoreButton
                expanded={hiveExpanded}
                onClick={() => setHiveExpanded(!hiveExpanded)}
              />
              {hiveExpanded && (
                <>
                  <CommandItem commandId="hive-status" label="Status" />
                  <CommandItem commandId="hive-list" label="List Hives" />
                </>
              )}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="p-4 border-t border-border">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Theme</span>
          <ThemeToggle />
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
