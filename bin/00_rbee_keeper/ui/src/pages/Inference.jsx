import { useState } from 'react'
import { invoke } from '@tauri-apps/api/tauri'

function Inference() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  
  // Form state
  const [hiveId, setHiveId] = useState('localhost')
  const [model, setModel] = useState('')
  const [prompt, setPrompt] = useState('')
  const [maxTokens, setMaxTokens] = useState(20)
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState('')
  const [topK, setTopK] = useState('')
  const [device, setDevice] = useState('')
  const [workerId, setWorkerId] = useState('')
  const [stream, setStream] = useState(true)

  const executeInference = async () => {
    setLoading(true)
    try {
      const response = await invoke('infer', { 
        hiveId,
        model,
        prompt,
        maxTokens: maxTokens || null,
        temperature: temperature || null,
        topP: topP ? parseFloat(topP) : null,
        topK: topK ? parseInt(topK) : null,
        device: device || null,
        workerId: workerId || null,
        stream: stream || null
      })
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
      <h1>Inference</h1>

      <div className="card">
        <h2>Run Inference</h2>
        
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
          <label>Model Identifier *</label>
          <input
            type="text"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="HF:author/model-name"
          />
        </div>

        <div className="form-group">
          <label>Prompt *</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt here..."
            rows={4}
            style={{ width: '100%', fontFamily: 'inherit', padding: '0.5em', borderRadius: '4px', border: '1px solid #444', backgroundColor: '#1a1a1a', color: 'inherit' }}
          />
        </div>

        <div className="form-group">
          <label>Max Tokens</label>
          <input
            type="number"
            value={maxTokens}
            onChange={(e) => setMaxTokens(parseInt(e.target.value))}
            placeholder="20"
          />
        </div>

        <div className="form-group">
          <label>Temperature</label>
          <input
            type="number"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            placeholder="0.7"
          />
        </div>

        <div className="form-group">
          <label>Top P (optional)</label>
          <input
            type="number"
            step="0.1"
            value={topP}
            onChange={(e) => setTopP(e.target.value)}
            placeholder="e.g., 0.9"
          />
        </div>

        <div className="form-group">
          <label>Top K (optional)</label>
          <input
            type="number"
            value={topK}
            onChange={(e) => setTopK(e.target.value)}
            placeholder="e.g., 50"
          />
        </div>

        <div className="form-group">
          <label>Device (optional)</label>
          <select value={device} onChange={(e) => setDevice(e.target.value)}>
            <option value="">Auto</option>
            <option value="cpu">CPU</option>
            <option value="cuda">CUDA</option>
            <option value="metal">Metal</option>
          </select>
        </div>

        <div className="form-group">
          <label>Worker ID (optional)</label>
          <input
            type="text"
            value={workerId}
            onChange={(e) => setWorkerId(e.target.value)}
            placeholder="Specific worker ID"
          />
        </div>

        <div className="form-group">
          <label>
            <input
              type="checkbox"
              checked={stream}
              onChange={(e) => setStream(e.target.checked)}
            />
            {' '}Stream tokens as generated
          </label>
        </div>

        <button 
          onClick={executeInference} 
          disabled={loading || !model || !prompt}
        >
          {loading ? 'Running Inference...' : 'Run Inference'}
        </button>
      </div>

      {result && (
        <div className="card">
          <h2 className={result.success ? 'success' : 'error'}>
            {result.success ? '✓ Success' : '✗ Error'}
          </h2>
          <p>{result.message}</p>
          {result.data && (
            <div>
              <h3>Response:</h3>
              <pre>{result.data}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default Inference
