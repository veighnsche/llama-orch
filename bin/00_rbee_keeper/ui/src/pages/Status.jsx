import { useState } from 'react'
import { invoke } from '@tauri-apps/api/tauri'

function Status() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const handleGetStatus = async () => {
    setLoading(true)
    try {
      const response = await invoke('get_status')
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
      <h1>System Status</h1>
      
      <div className="card">
        <p>View the current status of all hives and workers in the system.</p>
        <button onClick={handleGetStatus} disabled={loading}>
          {loading ? 'Loading...' : 'Get Status'}
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

export default Status
