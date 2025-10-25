// TEAM-291: Commands sidebar for Bee Keeper page

"use client";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@rbee/ui/atoms";

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
              <CommandItem commandId="queen-start" label="Start Queen" />
              <CommandItem commandId="queen-stop" label="Stop Queen" />
              <CommandItem commandId="queen-restart" label="Restart Queen" />
              <CommandItem commandId="queen-status" label="Queen Status" />
              <CommandItem commandId="queen-build" label="Build Queen" />
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Hive Operations */}
        <SidebarGroup>
          <SidebarGroupLabel>Hive</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <CommandItem commandId="hive-start" label="Start Hive" />
              <CommandItem commandId="hive-stop" label="Stop Hive" />
              <CommandItem commandId="hive-restart" label="Restart Hive" />
              <CommandItem commandId="hive-status" label="Hive Status" />
              <CommandItem commandId="hive-build" label="Build Hive" />
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Git Operations (SSH) */}
        <SidebarGroup>
          <SidebarGroupLabel>Git (SSH)</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <CommandItem commandId="git-clone" label="Clone Repository" />
              <CommandItem commandId="git-pull" label="Pull Updates" />
              <CommandItem commandId="git-status" label="Check Status" />
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
