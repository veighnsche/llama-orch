// TEAM-292: Commands sidebar for Bee Keeper page
// Ported from web-ui.old
// TEAM-294: Added ThemeToggle at bottom

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
import { ThemeToggle } from "@rbee/ui/molecules";

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
              <CommandItem commandId="queen-status" label="Status" />
              <CommandItem commandId="queen-info" label="Info" />
              <CommandItem commandId="queen-rebuild" label="Rebuild" />
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Hive Operations (localhost) */}
        <SidebarGroup>
          <SidebarGroupLabel>Hive (localhost)</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <CommandItem commandId="hive-start" label="Start Hive" />
              <CommandItem commandId="hive-stop" label="Stop Hive" />
              <CommandItem commandId="hive-status" label="Status" />
              <CommandItem commandId="hive-list" label="List Hives" />
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
