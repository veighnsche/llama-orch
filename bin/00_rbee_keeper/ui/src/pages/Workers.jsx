import { useState } from 'react'
import { invoke } from '@tauri-apps/api/tauri'

function Workers() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  
  // Form state
  const [hiveId, setHiveId] = useState('localhost')
  const [model, setModel] = useState('')
  const [device, setDevice] = useState('cpu')
  const [pid, setPid] = useState('')

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
      <h1>Worker Management</h1>

      <div className="card">
        <h2>Spawn Worker</h2>
        <div className="form-group">
          <label>Hive ID</label>
          <input
            type="text"
            value={hiveId}
            onChange={(e) => setHiveId(e.target.value)}
            placeholder="localhost"
          />
        </div>
        <div className="form-group">
          <label>Model</label>
          <input
            type="text"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="HF:author/model-name"
          />
        </div>
        <div className="form-group">
          <label>Device</label>
          <select value={device} onChange={(e) => setDevice(e.target.value)}>
            <option value="cpu">CPU</option>
            <option value="cuda:0">CUDA:0</option>
            <option value="cuda:1">CUDA:1</option>
            <option value="metal:0">Metal:0</option>
          </select>
        </div>
        <button 
          onClick={() => executeCommand('worker_spawn', { 
            hiveId,
            model,
            device
          })} 
          disabled={loading || !model}
        >
          Spawn Worker
        </button>
      </div>

      <div className="card">
        <h2>Worker Process Management</h2>
        <div className="form-group">
          <label>Hive ID</label>
          <input
            type="text"
            value={hiveId}
            onChange={(e) => setHiveId(e.target.value)}
            placeholder="localhost"
          />
        </div>
        <button 
          onClick={() => executeCommand('worker_process_list', { hiveId })} 
          disabled={loading}
        >
          List Worker Processes
        </button>
      </div>

      <div className="card">
        <h2>Worker Process Actions</h2>
        <div className="form-group">
          <label>Hive ID</label>
          <input
            type="text"
            value={hiveId}
            onChange={(e) => setHiveId(e.target.value)}
            placeholder="localhost"
          />
        </div>
        <div className="form-group">
          <label>Process ID (PID)</label>
          <input
            type="number"
            value={pid}
            onChange={(e) => setPid(e.target.value)}
            placeholder="12345"
          />
        </div>
        <div className="button-group">
          <button 
            onClick={() => executeCommand('worker_process_get', { 
              hiveId,
              pid: parseInt(pid)
            })} 
            disabled={loading || !pid}
          >
            Get Process Details
          </button>
          <button 
            onClick={() => executeCommand('worker_process_delete', { 
              hiveId,
              pid: parseInt(pid)
            })} 
            disabled={loading || !pid}
            style={{ backgroundColor: '#dc2626' }}
          >
            Kill Process
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

export default Workers
