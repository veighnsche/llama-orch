import { useState } from 'react'
import { invoke } from '@tauri-apps/api/tauri'

function Hives() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  
  // Form state
  const [host, setHost] = useState('localhost')
  const [binary, setBinary] = useState('')
  const [installDir, setInstallDir] = useState('')
  const [port, setPort] = useState(9000)
  const [alias, setAlias] = useState('localhost')

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
      <h1>Hive Management</h1>

      <div className="card">
        <h2>Quick Actions</h2>
        <div className="button-group">
          <button onClick={() => executeCommand('hive_list')} disabled={loading}>
            List All Hives
          </button>
        </div>
      </div>

      <div className="card">
        <h2>Install Hive</h2>
        <div className="form-group">
          <label>Host</label>
          <input
            type="text"
            value={host}
            onChange={(e) => setHost(e.target.value)}
            placeholder="localhost or remote.example.com"
          />
        </div>
        <div className="form-group">
          <label>Binary Path (optional)</label>
          <input
            type="text"
            value={binary}
            onChange={(e) => setBinary(e.target.value)}
            placeholder="Auto-detect from target/"
          />
        </div>
        <div className="form-group">
          <label>Installation Directory (optional)</label>
          <input
            type="text"
            value={installDir}
            onChange={(e) => setInstallDir(e.target.value)}
            placeholder="~/.local/bin or /usr/local/bin"
          />
        </div>
        <button 
          onClick={() => executeCommand('hive_install', { 
            host,
            binary: binary || null,
            installDir: installDir || null
          })} 
          disabled={loading}
        >
          Install Hive
        </button>
      </div>

      <div className="card">
        <h2>Start Hive</h2>
        <div className="form-group">
          <label>Host</label>
          <input
            type="text"
            value={host}
            onChange={(e) => setHost(e.target.value)}
            placeholder="localhost"
          />
        </div>
        <div className="form-group">
          <label>Port</label>
          <input
            type="number"
            value={port}
            onChange={(e) => setPort(parseInt(e.target.value))}
            placeholder="9000"
          />
        </div>
        <div className="form-group">
          <label>Installation Directory (optional)</label>
          <input
            type="text"
            value={installDir}
            onChange={(e) => setInstallDir(e.target.value)}
            placeholder="~/.local/bin"
          />
        </div>
        <button 
          onClick={() => executeCommand('hive_start', { 
            host,
            installDir: installDir || null,
            port
          })} 
          disabled={loading}
        >
          Start Hive
        </button>
      </div>

      <div className="card">
        <h2>Stop Hive</h2>
        <div className="form-group">
          <label>Host</label>
          <input
            type="text"
            value={host}
            onChange={(e) => setHost(e.target.value)}
            placeholder="localhost"
          />
        </div>
        <button 
          onClick={() => executeCommand('hive_stop', { host })} 
          disabled={loading}
        >
          Stop Hive
        </button>
      </div>

      <div className="card">
        <h2>Hive Info & Status</h2>
        <div className="form-group">
          <label>Hive Alias</label>
          <input
            type="text"
            value={alias}
            onChange={(e) => setAlias(e.target.value)}
            placeholder="localhost"
          />
        </div>
        <div className="button-group">
          <button 
            onClick={() => executeCommand('hive_get', { alias })} 
            disabled={loading}
          >
            Get Details
          </button>
          <button 
            onClick={() => executeCommand('hive_status', { alias })} 
            disabled={loading}
          >
            Check Status
          </button>
          <button 
            onClick={() => executeCommand('hive_refresh_capabilities', { alias })} 
            disabled={loading}
          >
            Refresh Capabilities
          </button>
        </div>
      </div>

      <div className="card">
        <h2>Uninstall Hive</h2>
        <div className="form-group">
          <label>Host</label>
          <input
            type="text"
            value={host}
            onChange={(e) => setHost(e.target.value)}
            placeholder="localhost"
          />
        </div>
        <div className="form-group">
          <label>Installation Directory (optional)</label>
          <input
            type="text"
            value={installDir}
            onChange={(e) => setInstallDir(e.target.value)}
            placeholder="~/.local/bin"
          />
        </div>
        <button 
          onClick={() => executeCommand('hive_uninstall', { 
            host,
            installDir: installDir || null
          })} 
          disabled={loading}
        >
          Uninstall Hive
        </button>
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

export default Hives
