import { fs } from '@llama-orch/utils';

const root = '/home/vince/Projects/llama-orch/consumers/.examples/M002-bun'

const seedFile = fs.fileReader(`${root}/.llama-orch/seed.md`)

console.log(seedFile)