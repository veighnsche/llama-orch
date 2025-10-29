// RHAI IDE Component
// Code editor for RHAI scheduling scripts with full CRUD

import { useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Button,
  IconButton,
  ButtonGroup,
  Input,
  Textarea,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Alert,
  AlertDescription,
} from "@rbee/ui/atoms";
import { useRhaiScripts } from "@rbee/queen-rbee-react";
import {
  Trash2,
  Plus,
  Save,
  Play,
  CheckCircle2,
  XCircle,
  AlertCircle,
} from "lucide-react";

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

  const [editedContent, setEditedContent] = useState(
    currentScript?.content || "",
  );
  const [editedName, setEditedName] = useState(
    currentScript?.name || "New Script",
  );
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

  // TEAM-XXX: Handle script selection, including sentinel for new script
  const handleSelectScript = (scriptId: string) => {
    if (scriptId === "__new__") {
      createNewScript();
    } else {
      selectScript(scriptId);
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
            <IconButton onClick={createNewScript} title="New Script">
              <Plus className="h-4 w-4" />
            </IconButton>
            {currentScript?.id && (
              <IconButton
                onClick={handleDelete}
                title="Delete Script"
                className="text-destructive hover:text-destructive hover:bg-destructive/10"
              >
                <Trash2 className="h-4 w-4" />
              </IconButton>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Script Selector */}
          {/* TEAM-XXX: Fixed Radix Select crash - no empty string values allowed */}
          <Select
            value={currentScript?.id || "__new__"}
            onValueChange={handleSelectScript}
            disabled={loading}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a script" />
            </SelectTrigger>
            <SelectContent>
              {/* Only render scripts with valid IDs */}
              {scripts.filter(s => s.id && s.id.trim() !== "").map((script) => (
                <SelectItem key={script.id} value={script.id!}>
                  {script.name}
                </SelectItem>
              ))}
              {/* Sentinel value for new script instead of empty string */}
              <SelectItem value="__new__">+ New Script</SelectItem>
            </SelectContent>
          </Select>

          {/* Script Name */}
          <Input
            type="text"
            value={editedName}
            onChange={(e) => setEditedName(e.target.value)}
            placeholder="Script name..."
          />

          {/* Code Editor */}
          <Textarea
            value={editedContent}
            onChange={(e) => setEditedContent(e.target.value)}
            className="h-64 font-mono text-sm resize-none"
            placeholder="// Write your RHAI scheduling script here..."
          />

          {/* Actions */}
          <div className="space-y-2">
            <ButtonGroup>
              <Button disabled={saving || loading} onClick={handleSave}>
                <Save className="h-4 w-4" />
                {saving ? "Saving..." : "Save"}
              </Button>
              <Button
                variant="outline"
                disabled={testing || loading}
                onClick={handleTest}
              >
                <Play className="h-4 w-4" />
                {testing ? "Testing..." : "Test"}
              </Button>
            </ButtonGroup>

            {/* Success Message */}
            {successMessage && (
              <Alert variant="success">
                <CheckCircle2 className="h-4 w-4" />
                <AlertDescription>{successMessage}</AlertDescription>
              </Alert>
            )}

            {/* Test Result */}
            {testResult && (
              <Alert variant={testResult.success ? "success" : "destructive"}>
                {testResult.success ? (
                  <CheckCircle2 className="h-4 w-4" />
                ) : (
                  <XCircle className="h-4 w-4" />
                )}
                <AlertDescription>
                  <pre className="text-xs whitespace-pre-wrap font-mono">
                    {testResult.success
                      ? `✅ Test passed!\n${testResult.output || ""}`
                      : `❌ Test failed:\n${testResult.error || "Unknown error"}`}
                  </pre>
                </AlertDescription>
              </Alert>
            )}

            {/* Error */}
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error.message}</AlertDescription>
              </Alert>
            )}
          </div>

          <Alert variant="warning">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Backend endpoints are stubs - will return 404 until implemented
            </AlertDescription>
          </Alert>
        </div>
      </CardContent>
    </Card>
  );
}
