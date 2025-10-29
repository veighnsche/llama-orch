// RHAI Script Management Hook
// Manages RHAI scripts with CRUD operations

'use client'

import { useEffect, useState } from 'react'
import { useRbeeSDK } from './useRbeeSDK'
import { createNarrationStreamHandler } from '../utils/narrationBridge'

export interface RhaiScript {
  id?: string
  name: string
  content: string
  created_at?: string
  updated_at?: string
}

export interface TestResult {
  success: boolean
  output?: string
  error?: string
}

export interface UseRhaiScriptsResult {
  scripts: RhaiScript[]
  currentScript: RhaiScript | null
  loading: boolean
  saving: boolean
  testing: boolean
  error: Error | null
  testResult: TestResult | null
  loadScripts: () => Promise<void>
  selectScript: (id: string) => Promise<void>
  saveScript: (script: RhaiScript) => Promise<void>
  testScript: (content: string) => Promise<void>
  deleteScript: (id: string) => Promise<void>
  createNewScript: () => void
}

const DEFAULT_SCRIPT = `// RHAI Scheduling Script
// Define custom scheduling logic here

fn schedule_worker(job) {
  // Your scheduling logic
  print("Scheduling job: " + job.id);
  
  // Return worker_id to use
  return "worker-0";
}`

/**
 * Hook for managing RHAI scripts
 * 
 * @param baseUrl - Queen API URL (default: http://localhost:7833)
 * @returns RHAI script management functions and state
 */
export function useRhaiScripts(baseUrl: string = 'http://localhost:7833'): UseRhaiScriptsResult {
  const { sdk, loading: sdkLoading } = useRbeeSDK()
  const [scripts, setScripts] = useState<RhaiScript[]>([])
  const [currentScript, setCurrentScript] = useState<RhaiScript | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [testResult, setTestResult] = useState<TestResult | null>(null)

  // Load scripts on mount
  useEffect(() => {
    if (sdk) {
      loadScripts()
    }
  }, [sdk])

  const loadScripts = async () => {
    if (!sdk) return

    setLoading(true)
    setError(null)
    try {
      const client = new sdk.RhaiClient(baseUrl)
      const result = await client.listScripts()
      const scriptList = JSON.parse(JSON.stringify(result))
      
      // TEAM-XXX: Backend returns stub object, not array yet
      // For now, set empty array until backend is implemented
      if (Array.isArray(scriptList)) {
        setScripts(scriptList)
        
        // Select first script if none selected
        if (!currentScript && scriptList.length > 0) {
          setCurrentScript(scriptList[0])
        }
      } else {
        // Backend not implemented yet, use empty array
        console.warn('[RHAI] Backend returned non-array:', scriptList)
        setScripts([])
      }
    } catch (err) {
      setError(err as Error)
      setScripts([]) // Ensure scripts is always an array
    } finally {
      setLoading(false)
    }
  }

  const selectScript = async (id: string) => {
    if (!sdk) return

    setLoading(true)
    setError(null)
    try {
      const client = new sdk.RhaiClient(baseUrl)
      const result = await client.getScript(id)
      const script = JSON.parse(JSON.stringify(result))
      
      // TEAM-XXX: Validate response structure
      if (script && typeof script === 'object' && script.name && script.content) {
        setCurrentScript(script)
      } else {
        console.warn('[RHAI] Backend returned invalid script:', script)
      }
    } catch (err) {
      setError(err as Error)
    } finally {
      setLoading(false)
    }
  }

  const saveScript = async (script: RhaiScript) => {
    if (!sdk) return

    setSaving(true)
    setError(null)
    try {
      const client = new sdk.RhaiClient(baseUrl)
      const result = await client.saveScript(script)
      const savedScript = JSON.parse(JSON.stringify(result))
      
      // TEAM-XXX: Backend returns stub, don't update scripts list yet
      if (savedScript && typeof savedScript === 'object' && savedScript.name) {
        // Update scripts list
        setScripts(prev => {
          const existing = prev.find(s => s.id === savedScript.id)
          if (existing) {
            return prev.map(s => s.id === savedScript.id ? savedScript : s)
          }
          return [...prev, savedScript]
        })
        
        setCurrentScript(savedScript)
      } else {
        console.warn('[RHAI] Backend returned invalid save result:', savedScript)
      }
    } catch (err) {
      setError(err as Error)
      throw err
    } finally {
      setSaving(false)
    }
  }

  const testScript = async (content: string) => {
    if (!sdk) {
      console.error('[RHAI Test] SDK not loaded')
      return
    }

    console.log('[RHAI Test] Starting test...')
    setTesting(true)
    setError(null)
    setTestResult(null)
    
    try {
      const client = new sdk.QueenClient(baseUrl)
      console.log('[RHAI Test] Client created, baseUrl:', baseUrl)
      
      // TEAM-350: Operation uses #[serde(tag = "operation")] format
      const operation = {
        operation: 'rhai_script_test',
        content
      }
      console.log('[RHAI Test] Operation:', operation)
      
      const narrationHandler = createNarrationStreamHandler((event) => {
        console.log('[RHAI Test] Narration event:', event)
      })
      
      let receivedDone = false
      
      console.log('[RHAI Test] Submitting and streaming...')
      await client.submitAndStream(operation, (line: string) => {
        console.log('[RHAI Test] SSE line:', line)
        
        narrationHandler(line)
        
        if (line.includes('[DONE]')) {
          receivedDone = true
          setTestResult({ success: true, output: 'Test completed successfully' })
        }
      })
      
      console.log('[RHAI Test] Stream complete, receivedDone:', receivedDone)
      
      if (!receivedDone) {
        console.warn('[RHAI Test] No [DONE] marker received')
        setTestResult({ success: true, output: 'Test completed (no DONE marker)' })
      }
    } catch (err) {
      console.error('[RHAI Test] Error caught:', err)
      console.error('[RHAI Test] Error type:', typeof err)
      console.error('[RHAI Test] Error details:', {
        message: (err as Error).message,
        stack: (err as Error).stack,
        name: (err as Error).name,
      })
      
      const errorMsg = (err as Error).message || String(err)
      setError(err as Error)
      setTestResult({ success: false, error: errorMsg })
      throw err
    } finally {
      setTesting(false)
      console.log('[RHAI Test] Finished')
    }
  }

  const deleteScript = async (id: string) => {
    if (!sdk) return

    setLoading(true)
    setError(null)
    try {
      const client = new sdk.RhaiClient(baseUrl)
      await client.deleteScript(id)
      
      // Remove from list
      setScripts(prev => prev.filter(s => s.id !== id))
      
      // Clear current if deleted
      if (currentScript?.id === id) {
        setCurrentScript(scripts.length > 1 ? scripts[0] : null)
      }
    } catch (err) {
      setError(err as Error)
      throw err
    } finally {
      setLoading(false)
    }
  }

  const createNewScript = () => {
    setCurrentScript({
      name: 'New Script',
      content: DEFAULT_SCRIPT,
    })
  }

  return {
    scripts,
    currentScript,
    loading: loading || sdkLoading,
    saving,
    testing,
    error,
    testResult,
    loadScripts,
    selectScript,
    saveScript,
    testScript,
    deleteScript,
    createNewScript,
  }
}
