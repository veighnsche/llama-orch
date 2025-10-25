// TEAM-294: rbee-keeper GUI - Main application component
import { useState } from 'react';
import { useCommand } from './hooks';
import * as api from './api';

type Tab = 'queen' | 'hives' | 'workers' | 'models' | 'inference';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('queen');
  const [output, setOutput] = useState<string>('');

  // Queen commands
  const queenStart = useCommand(api.queenStart);
  const queenStop = useCommand(api.queenStop);
  const queenStatus = useCommand(api.queenStatus);
  const queenInfo = useCommand(api.queenInfo);

  // Hive commands
  const hiveList = useCommand(api.hiveList);
  const hiveStatus = useCommand(() => api.hiveStatus('localhost'));

  // Worker commands
  const workerList = useCommand(() => api.workerProcessList('localhost'));

  // Model commands
  const modelList = useCommand(() => api.modelList('localhost'));

  const handleCommand = async (commandFn: () => Promise<unknown>) => {
    const result = await commandFn();
    if (result && typeof result === 'object' && 'message' in result) {
      const response = result as { success: boolean; message: string; data?: string };
      setOutput(`${response.success ? '✅' : '❌'} ${response.message}${response.data ? '\n\n' + response.data : ''}`);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-7xl mx-auto">
      <header className="px-8 py-6 border-b border-border text-center">
        <h1 className="text-3xl font-bold mb-2">🐝 rbee Keeper</h1>
        <p className="text-muted-foreground">Manage your rbee infrastructure</p>
      </header>

      <nav className="flex gap-2 px-8 py-4 border-b border-border bg-muted/30">
        <button
          className={`px-4 py-2 rounded-md border transition-all ${
            activeTab === 'queen'
              ? 'bg-primary text-primary-foreground border-primary'
              : 'bg-transparent border-border hover:bg-muted hover:border-primary'
          }`}
          onClick={() => setActiveTab('queen')}
        >
          Queen
        </button>
        <button
          className={`px-4 py-2 rounded-md border transition-all ${
            activeTab === 'hives'
              ? 'bg-primary text-primary-foreground border-primary'
              : 'bg-transparent border-border hover:bg-muted hover:border-primary'
          }`}
          onClick={() => setActiveTab('hives')}
        >
          Hives
        </button>
        <button
          className={`px-4 py-2 rounded-md border transition-all ${
            activeTab === 'workers'
              ? 'bg-primary text-primary-foreground border-primary'
              : 'bg-transparent border-border hover:bg-muted hover:border-primary'
          }`}
          onClick={() => setActiveTab('workers')}
        >
          Workers
        </button>
        <button
          className={`px-4 py-2 rounded-md border transition-all ${
            activeTab === 'models'
              ? 'bg-primary text-primary-foreground border-primary'
              : 'bg-transparent border-border hover:bg-muted hover:border-primary'
          }`}
          onClick={() => setActiveTab('models')}
        >
          Models
        </button>
        <button
          className={`px-4 py-2 rounded-md border transition-all ${
            activeTab === 'inference'
              ? 'bg-primary text-primary-foreground border-primary'
              : 'bg-transparent border-border hover:bg-muted hover:border-primary'
          }`}
          onClick={() => setActiveTab('inference')}
        >
          Inference
        </button>
      </nav>

      <main className="flex-1 px-8 py-6 overflow-y-auto">
        {activeTab === 'queen' && (
          <div className="max-w-3xl">
            <h2 className="text-2xl font-semibold mb-6">Queen Management</h2>
            <div className="flex flex-wrap gap-3 mb-4">
              <button
                onClick={() => handleCommand(queenStart.execute)}
                disabled={queenStart.loading}
                className="px-6 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {queenStart.loading ? 'Starting...' : 'Start Queen'}
              </button>
              <button
                onClick={() => handleCommand(queenStop.execute)}
                disabled={queenStop.loading}
                className="px-6 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {queenStop.loading ? 'Stopping...' : 'Stop Queen'}
              </button>
              <button
                onClick={() => handleCommand(queenStatus.execute)}
                disabled={queenStatus.loading}
                className="px-6 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {queenStatus.loading ? 'Checking...' : 'Check Status'}
              </button>
              <button
                onClick={() => handleCommand(queenInfo.execute)}
                disabled={queenInfo.loading}
                className="px-6 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {queenInfo.loading ? 'Loading...' : 'Get Info'}
              </button>
            </div>
            {queenStart.error && <div className="p-4 mt-4 bg-destructive/10 border border-destructive rounded-md text-destructive">{queenStart.error}</div>}
            {queenStop.error && <div className="p-4 mt-4 bg-destructive/10 border border-destructive rounded-md text-destructive">{queenStop.error}</div>}
            {queenStatus.error && <div className="p-4 mt-4 bg-destructive/10 border border-destructive rounded-md text-destructive">{queenStatus.error}</div>}
            {queenInfo.error && <div className="p-4 mt-4 bg-destructive/10 border border-destructive rounded-md text-destructive">{queenInfo.error}</div>}
          </div>
        )}

        {activeTab === 'hives' && (
          <div className="max-w-3xl">
            <h2 className="text-2xl font-semibold mb-6">Hive Management</h2>
            <div className="flex flex-wrap gap-3 mb-4">
              <button
                onClick={() => handleCommand(hiveList.execute)}
                disabled={hiveList.loading}
                className="px-6 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {hiveList.loading ? 'Loading...' : 'List Hives'}
              </button>
              <button
                onClick={() => handleCommand(hiveStatus.execute)}
                disabled={hiveStatus.loading}
                className="px-6 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {hiveStatus.loading ? 'Checking...' : 'Check Localhost Status'}
              </button>
            </div>
            {hiveList.error && <div className="p-4 mt-4 bg-destructive/10 border border-destructive rounded-md text-destructive">{hiveList.error}</div>}
            {hiveStatus.error && <div className="p-4 mt-4 bg-destructive/10 border border-destructive rounded-md text-destructive">{hiveStatus.error}</div>}
          </div>
        )}

        {activeTab === 'workers' && (
          <div className="max-w-3xl">
            <h2 className="text-2xl font-semibold mb-6">Worker Management</h2>
            <div className="flex flex-wrap gap-3 mb-4">
              <button
                onClick={() => handleCommand(workerList.execute)}
                disabled={workerList.loading}
                className="px-6 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {workerList.loading ? 'Loading...' : 'List Workers'}
              </button>
            </div>
            {workerList.error && <div className="p-4 mt-4 bg-destructive/10 border border-destructive rounded-md text-destructive">{workerList.error}</div>}
          </div>
        )}

        {activeTab === 'models' && (
          <div className="max-w-3xl">
            <h2 className="text-2xl font-semibold mb-6">Model Management</h2>
            <div className="flex flex-wrap gap-3 mb-4">
              <button
                onClick={() => handleCommand(modelList.execute)}
                disabled={modelList.loading}
                className="px-6 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {modelList.loading ? 'Loading...' : 'List Models'}
              </button>
            </div>
            {modelList.error && <div className="p-4 mt-4 bg-destructive/10 border border-destructive rounded-md text-destructive">{modelList.error}</div>}
          </div>
        )}

        {activeTab === 'inference' && (
          <div className="max-w-3xl">
            <h2 className="text-2xl font-semibold mb-6">Inference</h2>
            <p className="text-muted-foreground">Inference UI coming soon...</p>
          </div>
        )}

        {output && (
          <div className="mt-8 p-4 bg-muted border border-border rounded-md">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-3">Output:</h3>
            <pre className="p-4 bg-background rounded-md overflow-x-auto whitespace-pre-wrap break-words font-mono text-sm leading-relaxed">{output}</pre>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
