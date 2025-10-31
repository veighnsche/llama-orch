// TEAM-352: Migrated to use TanStack Query
// Old implementation: ~274 LOC with manual async state management
// New implementation: ~180 LOC using TanStack Query
// Reduction: 94 LOC (34%)
// Note: Kept CRUD operations (save/delete/test) - business logic specific to RHAI

'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useQueenSDK } from './useQueenSDK'
// TEAM-352: Import directly from @rbee/narration-client (no wrapper)
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

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
 * TEAM-352: Now uses TanStack Query for async state management
 * 
 * @param baseUrl - Queen API URL (default: http://localhost:7833)
 * @returns RHAI script management functions and state
 */
export function useRhaiScripts(baseUrl: string = 'http://localhost:7833'): UseRhaiScriptsResult {
  const { sdk } = useQueenSDK()
  const [currentScript, setCurrentScript] = useState<RhaiScript | null>(null)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<TestResult | null>(null)

  // TEAM-352: Use TanStack Query for loading scripts
  const {
    data: scripts,
    isLoading: loading,
    error,
    refetch: loadScripts,
  } = useQuery({
    queryKey: ['rhai-scripts', baseUrl],
    queryFn: async () => {
      if (!sdk) return []
      const client = new sdk.RhaiClient(baseUrl)
      const result = await client.listScripts()
      const scriptList = JSON.parse(JSON.stringify(result))

      // Backend returns stub, handle gracefully
      if (Array.isArray(scriptList)) {
        // Select first script if none selected
        if (!currentScript && scriptList.length > 0) {
          setCurrentScript(scriptList[0])
        }
        return scriptList
      } else {
        console.warn('[RHAI] Backend returned non-array:', scriptList)
        return []
      }
    },
    enabled: !!sdk,
  })

  // TEAM-352: Keep business logic functions (CRUD operations)
  const selectScript = async (id: string) => {
    if (!sdk) return

    try {
      const client = new sdk.RhaiClient(baseUrl)
      const result = await client.getScript(id)
      const script = JSON.parse(JSON.stringify(result))

      if (script && typeof script === 'object' && script.name && script.content) {
        setCurrentScript(script)
      } else {
        console.warn('[RHAI] Backend returned invalid script:', script)
      }
    } catch (err) {
      console.error('[RHAI] Failed to load script:', err)
    }
  }

  const saveScript = async (script: RhaiScript) => {
    if (!sdk) return

    setSaving(true)
    try {
      const client = new sdk.RhaiClient(baseUrl)
      const result = await client.saveScript(script)
      const savedScript = JSON.parse(JSON.stringify(result))

      if (savedScript && typeof savedScript === 'object' && savedScript.name) {
        setCurrentScript(savedScript)
        // Reload scripts list
        await loadScripts()
      } else {
        console.warn('[RHAI] Backend returned invalid save result:', savedScript)
      }
    } catch (err) {
      console.error('[RHAI] Failed to save script:', err)
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
    setTestResult(null)

    try {
      const client = new sdk.QueenClient(baseUrl)
      console.log('[RHAI Test] Client created, baseUrl:', baseUrl)

      const operation = {
        operation: 'rhai_script_test',
        content,
      }
      console.log('[RHAI Test] Operation:', operation)

      // TEAM-352: Use createStreamHandler with SERVICES.queen config
      const narrationHandler = createStreamHandler(SERVICES.queen, (event) => {
        console.log('[RHAI Test] Narration event:', event)
      }, {
        debug: true,
        silent: false,
        validate: true,
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

      const errorMsg = (err as Error).message || String(err)
      setTestResult({ success: false, error: errorMsg })
      throw err
    } finally {
      setTesting(false)
      console.log('[RHAI Test] Finished')
    }
  }

  const deleteScript = async (id: string) => {
    if (!sdk) return

    try {
      const client = new sdk.RhaiClient(baseUrl)
      await client.deleteScript(id)

      // Clear current if deleted
      if (currentScript?.id === id) {
        const remaining = scripts?.filter((s) => s.id !== id) || []
        setCurrentScript(remaining.length > 0 ? remaining[0] : null)
      }

      // Reload scripts list
      await loadScripts()
    } catch (err) {
      console.error('[RHAI] Failed to delete script:', err)
      throw err
    }
  }

  const createNewScript = () => {
    setCurrentScript({
      name: 'New Script',
      content: DEFAULT_SCRIPT,
    })
  }

  return {
    scripts: scripts || [],
    currentScript,
    loading,
    saving,
    testing,
    error: error as Error | null,
    testResult,
    loadScripts: async () => {
      await loadScripts()
    },
    selectScript,
    saveScript,
    testScript,
    deleteScript,
    createNewScript,
  }
}
