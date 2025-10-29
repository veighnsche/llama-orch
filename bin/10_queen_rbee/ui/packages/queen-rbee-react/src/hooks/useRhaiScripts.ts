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
      setScripts(scriptList)
      
      // Select first script if none selected
      if (!currentScript && scriptList.length > 0) {
        setCurrentScript(scriptList[0])
      }
    } catch (err) {
      setError(err as Error)
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
      setCurrentScript(script)
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
      
      // Update scripts list
      setScripts(prev => {
        const existing = prev.find(s => s.id === savedScript.id)
        if (existing) {
          return prev.map(s => s.id === savedScript.id ? savedScript : s)
        }
        return [...prev, savedScript]
      })
      
      setCurrentScript(savedScript)
    } catch (err) {
      setError(err as Error)
      throw err
    } finally {
      setSaving(false)
    }
  }

  const testScript = async (content: string) => {
    if (!sdk) return

    setTesting(true)
    setError(null)
    setTestResult(null)
    try {
      // TEAM-XXX: Use QueenClient with SSE streaming for narration
      const client = new sdk.QueenClient(baseUrl)
      
      // Build RHAI test operation using JSON
      const operation = {
        RhaiScriptTest: { content }
      }
      
      // Submit with streaming to get narration events
      // TEAM-XXX: Use narration bridge to send events to parent (rbee-keeper)
      const narrationHandler = createNarrationStreamHandler((event) => {
        console.log('[RHAI Test] Narration event:', event)
      })
      
      await client.submitAndStream(operation, (line: string) => {
        console.log('[RHAI Test] SSE line:', line)
        
        // Send narration to parent window
        narrationHandler(line)
        
        // Parse for [DONE] marker
        if (line.includes('[DONE]')) {
          setTestResult({ success: true, output: 'Test completed successfully' })
        }
      })
      
      // If no result set yet, mark as success
      if (!testResult) {
        setTestResult({ success: true, output: 'Test completed' })
      }
    } catch (err) {
      setError(err as Error)
      setTestResult({ success: false, error: (err as Error).message })
      throw err
    } finally {
      setTesting(false)
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
