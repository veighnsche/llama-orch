import { useState } from 'react'
import { invoke } from '@tauri-apps/api/tauri'

function Queen() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [withLocalHive, setWithLocalHive] = useState(false)
  const [binaryPath, setBinaryPath] = useState('')

  const executeCommand = async (command, args = {}) => {
    setLoading(true)
    try {
      const response = await invoke(command, args)
      const data = JSON.parse(response)
      setResult(data)
    } catch (error) {
      setResult({ success: false, message: error })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h1>Queen Management</h1>

      <div className="card">
        <h2>Queen Daemon Control</h2>
        <div className="button-group">
          <button onClick={() => executeCommand('queen_start')} disabled={loading}>
            Start Queen
          </button>
          <button onClick={() => executeCommand('queen_stop')} disabled={loading}>
            Stop Queen
          </button>
          <button onClick={() => executeCommand('queen_status')} disabled={loading}>
            Check Status
          </button>
          <button onClick={() => executeCommand('queen_info')} disabled={loading}>
            Build Info
          </button>
        </div>
      </div>

      <div className="card">
        <h2>Rebuild Queen</h2>
        <div className="form-group">
          <label>
            <input
              type="checkbox"
              checked={withLocalHive}
              onChange={(e) => setWithLocalHive(e.target.checked)}
            />
            {' '}Include local hive (50-100x faster for localhost operations)
          </label>
        </div>
        <button 
          onClick={() => executeCommand('queen_rebuild', { withLocalHive })} 
          disabled={loading}
        >
          Rebuild Queen
        </button>
      </div>

      <div className="card">
        <h2>Installation</h2>
        <div className="form-group">
          <label>Binary Path (optional, auto-detect from target/)</label>
          <input
            type="text"
            value={binaryPath}
            onChange={(e) => setBinaryPath(e.target.value)}
            placeholder="/path/to/queen-rbee"
          />
        </div>
        <div className="button-group">
          <button 
            onClick={() => executeCommand('queen_install', { 
              binary: binaryPath || null 
            })} 
            disabled={loading}
          >
            Install Queen
          </button>
          <button onClick={() => executeCommand('queen_uninstall')} disabled={loading}>
            Uninstall Queen
          </button>
        </div>
      </div>

      {result && (
        <div className="card">
          <h2 className={result.success ? 'success' : 'error'}>
            {result.success ? '✓ Success' : '✗ Error'}
          </h2>
          <p>{result.message}</p>
          {result.data && (
            <pre>{result.data}</pre>
          )}
        </div>
      )}
    </div>
  )
}

export default Queen
