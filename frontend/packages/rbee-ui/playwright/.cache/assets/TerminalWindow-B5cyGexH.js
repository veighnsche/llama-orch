import { a as cn, j as jsxRuntimeExports } from './index-G2EA92RG.js'
import './index-BIkO9nGk.js'

function TerminalWindow({ title, children, variant = 'terminal', className }) {
  return /* @__PURE__ */ jsxRuntimeExports.jsxs('div', {
    className: cn('bg-card border border-border rounded-lg overflow-hidden shadow-2xl', className),
    children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs('div', {
        className: 'flex items-center gap-2 px-4 py-3 bg-muted border-b border-border',
        children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs('div', {
            className: 'flex gap-2',
            'aria-hidden': 'true',
            children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx('div', {
                className: 'h-3 w-3 rounded-full',
                style: { backgroundColor: 'var(--terminal-red)' },
              }),
              /* @__PURE__ */ jsxRuntimeExports.jsx('div', {
                className: 'h-3 w-3 rounded-full',
                style: { backgroundColor: 'var(--terminal-amber)' },
              }),
              /* @__PURE__ */ jsxRuntimeExports.jsx('div', {
                className: 'h-3 w-3 rounded-full',
                style: { backgroundColor: 'var(--terminal-green)' },
              }),
            ],
          }),
          title &&
            /* @__PURE__ */ jsxRuntimeExports.jsx('span', {
              className: 'text-muted-foreground text-sm ml-2 font-mono',
              children: title,
            }),
        ],
      }),
      /* @__PURE__ */ jsxRuntimeExports.jsx('div', { className: 'p-6 text-sm font-mono', children }),
    ],
  })
}

export { TerminalWindow }
//# sourceMappingURL=TerminalWindow-B5cyGexH.js.map
