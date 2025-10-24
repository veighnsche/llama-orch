// TEAM-291: Help page

'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@rbee/ui/atoms';
import { Button } from '@rbee/ui/atoms';
import { ThemeToggle } from '@rbee/ui/molecules';
import { BookOpenIcon, GithubIcon, MessageCircleIcon, FileTextIcon } from 'lucide-react';

export default function HelpPage() {
  return (
    <div className="flex-1 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Help & Documentation</h1>
          <p className="text-muted-foreground">
            Resources to help you get started with rbee
          </p>
        </div>
        <ThemeToggle />
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpenIcon className="size-5" />
              Documentation
            </CardTitle>
            <CardDescription>Read the full documentation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-sm text-muted-foreground mb-4">
              Learn how to use rbee, from basic concepts to advanced features.
            </p>
            <Button variant="outline" className="w-full">
              View Documentation
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GithubIcon className="size-5" />
              GitHub Repository
            </CardTitle>
            <CardDescription>Source code and issues</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-sm text-muted-foreground mb-4">
              View the source code, report bugs, or contribute to the project.
            </p>
            <Button variant="outline" className="w-full">
              Open GitHub
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MessageCircleIcon className="size-5" />
              Community Support
            </CardTitle>
            <CardDescription>Get help from the community</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-sm text-muted-foreground mb-4">
              Join our community to ask questions and share knowledge.
            </p>
            <Button variant="outline" className="w-full">
              Join Community
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileTextIcon className="size-5" />
              API Reference
            </CardTitle>
            <CardDescription>Detailed API documentation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-sm text-muted-foreground mb-4">
              Complete reference for all rbee APIs and operations.
            </p>
            <Button variant="outline" className="w-full">
              View API Docs
            </Button>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Quick Start Guide</CardTitle>
          <CardDescription>Get up and running in minutes</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <h3 className="font-semibold">1. Start the Queen</h3>
            <code className="block bg-muted p-2 rounded text-sm">
              ./rbee queen start
            </code>
          </div>
          <div className="space-y-2">
            <h3 className="font-semibold">2. Start a Hive</h3>
            <code className="block bg-muted p-2 rounded text-sm">
              ./rbee hive start -a localhost
            </code>
          </div>
          <div className="space-y-2">
            <h3 className="font-semibold">3. Spawn a Worker</h3>
            <code className="block bg-muted p-2 rounded text-sm">
              ./rbee worker spawn --hive localhost --model llama-3-8b
            </code>
          </div>
          <div className="space-y-2">
            <h3 className="font-semibold">4. Run Inference</h3>
            <code className="block bg-muted p-2 rounded text-sm">
              ./rbee infer --prompt "Hello, world!"
            </code>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
