// Local type shim to map ../esm.js to napi-generated types at package root without tracking .d.ts artifacts.
declare module '../esm.js' {
  export * from '..'

  export namespace fs {
    export function readFile(input: import('..').ReadRequestNapi): import('..').ReadResponseNapi
  }

  export namespace prompt {
    export function message(input: import('..').MessageInNapi): import('..').MessageNapi
    export function thread(input: import('..').ThreadInNapi): import('..').ThreadOutNapi
  }

  export namespace model {
    export function define(modelId: string, engineId?: string | null | undefined, poolHint?: string | null | undefined): import('..').ModelRefNapi
  }

  export namespace params {
    export function define(input: Omit<import('..').ParamsNapi, 'seed'> & { seed?: number | null | undefined }): import('..').ParamsNapi
  }

  export namespace llm {
    export function invoke(input: Omit<import('..').InvokeInNapi, 'model' | 'params'> & {
      model: Omit<import('..').ModelRefNapi, 'engineId' | 'poolHint'> & { engineId?: string | null | undefined; poolHint?: string | null | undefined };
      params: Omit<import('..').ParamsNapi, 'seed'> & { seed?: number | null | undefined };
    }): import('..').InvokeOutNapi
  }

  export namespace orch {
    export function response_extractor(result: import('..').InvokeResultNapi): string
    export function responseExtractor(result: import('..').InvokeResultNapi): string
  }

  export const probe: () => string

  const _default: {
    fs: typeof fs,
    prompt: typeof prompt,
    model: typeof model,
    params: typeof params,
    llm: typeof llm,
    orch: typeof orch,
    probe: typeof probe
  }
  export default _default
}
