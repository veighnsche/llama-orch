import { useState } from 'react'
import { invoke } from '@tauri-apps/api/tauri'

function Models() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  
  // Form state
  const [hiveId, setHiveId] = useState('localhost')
  const [model, setModel] = useState('')
  const [modelId, setModelId] = useState('')

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
      <h1>Model Management</h1>

      <div className="card">
        <h2>Download Model</h2>
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
          <label>Model Identifier</label>
          <input
            type="text"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="HF:author/model-name"
          />
        </div>
        <button 
          onClick={() => executeCommand('model_download', { 
            hiveId,
            model
          })} 
          disabled={loading || !model}
        >
          Download Model
        </button>
      </div>

      <div className="card">
        <h2>List Models</h2>
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
          onClick={() => executeCommand('model_list', { hiveId })} 
          disabled={loading}
        >
          List All Models
        </button>
      </div>

      <div className="card">
        <h2>Model Actions</h2>
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
          <label>Model ID</label>
          <input
            type="text"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            placeholder="model-id"
          />
        </div>
        <div className="button-group">
          <button 
            onClick={() => executeCommand('model_get', { 
              hiveId,
              id: modelId
            })} 
            disabled={loading || !modelId}
          >
            Get Model Details
          </button>
          <button 
            onClick={() => executeCommand('model_delete', { 
              hiveId,
              id: modelId
            })} 
            disabled={loading || !modelId}
            style={{ backgroundColor: '#dc2626' }}
          >
            Delete Model
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

export default Models
