// TEAM-294: rbee-keeper GUI - Main application component
// TEAM-292: Replaced placeholder with KeeperPage component
import KeeperPage from "./pages/KeeperPage";
import { SidebarProvider } from "@rbee/ui/atoms";

function App() {
  return (
    <SidebarProvider>
      <div className="h-screen w-screen overflow-hidden">
        <KeeperPage />
      </div>
    </SidebarProvider>
  );
}

export default App;
