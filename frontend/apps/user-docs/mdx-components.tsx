import { useMDXComponents as getDocsComponents } from 'nextra-theme-docs';

export function useMDXComponents(components: any): any {
  return {
    ...getDocsComponents(components),
    ...components,
  };
}
