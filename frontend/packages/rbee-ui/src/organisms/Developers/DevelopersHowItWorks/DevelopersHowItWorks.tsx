import { HowItWorksSection } from '@rbee/ui/organisms'

export function DevelopersHowItWorks() {
  return (
    <HowItWorksSection
      id="quickstart"
      title="From zero to AI infrastructure in 15 minutes"
      steps={[
        {
          label: 'Install rbee',
          block: {
            kind: 'terminal',
            title: 'terminal',
            lines: (
              <>
                <div>curl -sSL https://rbee.dev/install.sh | sh</div>
                <div className="text-slate-400">rbee-keeper daemon start</div>
              </>
            ),
            copyText: 'curl -sSL https://rbee.dev/install.sh | sh\nrbee-keeper daemon start',
          },
        },
        {
          label: 'Add your machines',
          block: {
            kind: 'terminal',
            title: 'terminal',
            lines: (
              <>
                <div>rbee-keeper setup add-node --name workstation --ssh-host 192.168.1.10</div>
                <div className="text-slate-400">rbee-keeper setup add-node --name mac --ssh-host 192.168.1.20</div>
              </>
            ),
            copyText:
              'rbee-keeper setup add-node --name workstation --ssh-host 192.168.1.10\nrbee-keeper setup add-node --name mac --ssh-host 192.168.1.20',
          },
        },
        {
          label: 'Configure your IDE',
          block: {
            kind: 'terminal',
            title: 'terminal',
            lines: (
              <>
                <div>
                  <span className="text-blue-400">export</span> OPENAI_API_BASE=http://localhost:8080/v1
                </div>
                <div className="text-slate-400"># Now Zed, Cursor, or any OpenAI-compatible tool works!</div>
              </>
            ),
            copyText: 'export OPENAI_API_BASE=http://localhost:8080/v1',
          },
        },
        {
          label: 'Build AI agents',
          block: {
            kind: 'code',
            title: 'TypeScript',
            language: 'ts',
            lines: (
              <>
                <div>
                  <span className="text-purple-400">import</span> {'{'} invoke {'}'}{' '}
                  <span className="text-purple-400">from</span>{' '}
                  <span className="text-amber-400">&apos;@llama-orch/utils&apos;</span>;
                </div>
                <div className="mt-2">
                  <span className="text-blue-400">const</span> code = <span className="text-blue-400">await</span>{' '}
                  <span className="text-green-400">invoke</span>
                  {'({'}
                </div>
                <div className="pl-4">
                  prompt: <span className="text-amber-400">&apos;Generate API from schema&apos;</span>,
                </div>
                <div className="pl-4">
                  model: <span className="text-amber-400">&apos;llama-3.1-70b&apos;</span>
                </div>
                <div>{'});'}</div>
              </>
            ),
            copyText:
              "import { invoke } from '@llama-orch/utils';\n\nconst code = await invoke({\n  prompt: 'Generate API from schema',\n  model: 'llama-3.1-70b'\n});",
          },
        },
      ]}
    />
  )
}
