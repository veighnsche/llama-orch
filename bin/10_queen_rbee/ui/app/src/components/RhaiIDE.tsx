// RHAI IDE Component
// Code editor for RHAI scheduling scripts with full CRUD

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@rbee/ui/atoms";
import { useRhaiScripts } from "@rbee/queen-rbee-react";
import { Trash2, Plus, Save, Play } from "lucide-react";

export function RhaiIDE() {
  const {
    scripts,
    currentScript,
    loading,
    saving,
    testing,
    error,
    testResult,
    saveScript,
    testScript,
    deleteScript,
    selectScript,
    createNewScript,
  } = useRhaiScripts();

  const [editedContent, setEditedContent] = useState(currentScript?.content || "");
  const [editedName, setEditedName] = useState(currentScript?.name || "New Script");
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Update local state when current script changes
  useState(() => {
    if (currentScript) {
      setEditedContent(currentScript.content);
      setEditedName(currentScript.name);
    }
  });

  const handleSave = async () => {
    try {
      await saveScript({
        ...currentScript,
        name: editedName,
        content: editedContent,
      });
      setSuccessMessage("✅ Script saved successfully!");
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      // Error handled by hook
    }
  };

  const handleTest = async () => {
    try {
      await testScript(editedContent);
    } catch (err) {
      // Error handled by hook
    }
  };

  const handleDelete = async () => {
    if (!currentScript?.id) return;
    if (!confirm(`Delete "${currentScript.name}"?`)) return;
    try {
      await deleteScript(currentScript.id);
    } catch (err) {
      // Error handled by hook
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>RHAI Scheduler IDE</CardTitle>
          <div className="flex gap-2">
            <button
              onClick={createNewScript}
              className="p-2 hover:bg-accent rounded transition-colors"
              title="New Script"
            >
              <Plus className="h-4 w-4" />
            </button>
            {currentScript?.id && (
              <button
                onClick={handleDelete}
                className="p-2 hover:bg-destructive/10 text-destructive rounded transition-colors"
                title="Delete Script"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Script Selector */}
          <div className="flex gap-2">
            <select
              value={currentScript?.id || ""}
              onChange={(e) => selectScript(e.target.value)}
              className="flex-1 px-3 py-2 border rounded-lg bg-background"
              disabled={loading}
            >
              {scripts.map((script) => (
                <option key={script.id} value={script.id}>
                  {script.name}
                </option>
              ))}
              {!currentScript?.id && (
                <option value="">New Script</option>
              )}
            </select>
          </div>

          {/* Script Name */}
          <input
            type="text"
            value={editedName}
            onChange={(e) => setEditedName(e.target.value)}
            className="w-full px-3 py-2 border rounded-lg bg-background"
            placeholder="Script name..."
          />

          {/* Code Editor */}
          <textarea
            value={editedContent}
            onChange={(e) => setEditedContent(e.target.value)}
            className="w-full h-64 p-4 font-mono text-sm bg-muted border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-ring"
            placeholder="// Write your RHAI scheduling script here..."
          />

          {/* Actions */}
          <div className="space-y-2">
            <div className="flex gap-2">
              <button
                className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
                disabled={saving || loading}
                onClick={handleSave}
              >
                <Save className="h-4 w-4" />
                {saving ? "Saving..." : "Save"}
              </button>
              <button
                className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-accent transition-colors disabled:opacity-50"
                disabled={testing || loading}
                onClick={handleTest}
              >
                <Play className="h-4 w-4" />
                {testing ? "Testing..." : "Test"}
              </button>
            </div>

            {/* Success Message */}
            {successMessage && (
              <div className="p-3 bg-green-500/10 text-green-500 rounded-lg">
                <p className="text-xs">{successMessage}</p>
              </div>
            )}

            {/* Test Result */}
            {testResult && (
              <div className={`p-3 rounded-lg ${testResult.success ? 'bg-green-500/10 text-green-500' : 'bg-destructive/10 text-destructive'}`}>
                <pre className="text-xs whitespace-pre-wrap">
                  {testResult.success ? `✅ Test passed!\n${testResult.output || ""}` : `❌ Test failed:\n${testResult.error || "Unknown error"}`}
                </pre>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="p-3 bg-destructive/10 text-destructive rounded-lg">
                <p className="text-xs">{error.message}</p>
              </div>
            )}
          </div>

          <p className="text-xs text-muted-foreground">
            ⚠️ Backend endpoints are stubs - will return 404 until implemented
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
