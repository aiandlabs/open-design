#!/usr/bin/env node
// Install a local ComfyUI server for the `comfyui` media provider and the
// `comfyui-agent` asset loop. Pins ComfyUI to the commit in models.json and
// downloads only models whose licences permit commercial use, verifying each
// file's sha256 before it is moved into place.
//
// Models come in sets (a diffusion model with its text encoder and VAE). By
// default the script installs every default set that fits this machine's
// memory; `--sets` picks sets by id and `--list` shows them.
//
// Usage:
//   node scripts/comfyui-local/setup.ts --list
//   node scripts/comfyui-local/setup.ts --dir <install-dir>
//   node scripts/comfyui-local/setup.ts --dir <install-dir> --sets z-image-turbo,flux2-klein-4b
//   node scripts/comfyui-local/setup.ts --dir <install-dir> --models sdxl-base-1.0
//   node scripts/comfyui-local/setup.ts --dir <install-dir> --gpu cuda
//   node scripts/comfyui-local/setup.ts --dir <install-dir> --start --port 8188
//
// --gpu defaults to `mps` on Apple silicon and `cuda` on Linux. --memory-gb
// overrides the detected memory used to choose sets. Requires git and uv on PATH.

import { spawnSync, spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync, readFileSync, renameSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

interface ModelEntry {
  id: string;
  folder: string;
  file: string;
  url: string;
  sha256: string;
  bytes: number;
  license: string;
  licenseUrl: string;
  commercialUse: boolean;
}

interface ModelSet {
  id: string;
  label: string;
  models: string[];
  /** Unified (Apple silicon) or GPU memory the set needs to run without heavy swapping. */
  minMemoryGb: number;
  default?: boolean;
}

/** A custom node pinned to a commit; installed into custom_nodes/<name>. */
interface CustomNode {
  name: string;
  repo: string;
  commit: string;
  license: string;
}

interface Manifest {
  comfyui: { repo: string; commit: string };
  customNodes: CustomNode[];
  models: ModelEntry[];
  sets: ModelSet[];
}

const here = path.dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(readFileSync(path.join(here, 'models.json'), 'utf8')) as Manifest;

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

function run(cmd: string, args: string[], cwd?: string): void {
  console.log(`$ ${cmd} ${args.join(' ')}`);
  const res = spawnSync(cmd, args, { cwd, stdio: 'inherit' });
  if (res.status !== 0) throw new Error(`${cmd} exited with ${res.status ?? res.signal}`);
}

async function sha256(file: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk as Buffer);
  return hash.digest('hex');
}

function modelById(id: string): ModelEntry {
  const model = manifest.models.find((m) => m.id === id);
  if (!model) throw new Error(`unknown model id ${id}; known: ${manifest.models.map((m) => m.id).join(', ')}`);
  return model;
}

function setBytes(set: ModelSet): number {
  return set.models.reduce((sum, id) => sum + modelById(id).bytes, 0);
}

const gb = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`;
const memoryGb = Number(arg('memory-gb') ?? os.totalmem() / 2 ** 30);

if (process.argv.includes('--list')) {
  console.log(`Machine memory: ${memoryGb.toFixed(0)} GB\n`);
  for (const set of manifest.sets) {
    const fits = set.minMemoryGb <= memoryGb;
    console.log(`${set.id}${set.default ? ' (default)' : ''}: ${set.label}`);
    console.log(`  download ${gb(setBytes(set))}, needs ${set.minMemoryGb} GB memory${fits ? '' : ' — does not fit this machine'}`);
    for (const id of set.models) {
      const m = modelById(id);
      console.log(`  - ${m.folder}/${m.file} (${gb(m.bytes)}, ${m.license})`);
    }
  }
  process.exit(0);
}

const dir = arg('dir');
if (!dir) {
  console.error(
    'usage: node scripts/comfyui-local/setup.ts --dir <install-dir> [--sets id,id | --models id,id] [--gpu mps|cuda|cpu] [--memory-gb N] [--start] [--port 8188]\n' +
      '       node scripts/comfyui-local/setup.ts --list',
  );
  process.exit(2);
}

// Choose the models: explicit ids, explicit sets, or every default set that fits.
let chosenSets: ModelSet[] = [];
let wanted: string[];
const modelsArg = arg('models');
const setsArg = arg('sets');
if (modelsArg) {
  wanted = modelsArg.split(',');
} else {
  if (setsArg) {
    chosenSets = setsArg.split(',').map((id) => {
      const set = manifest.sets.find((s) => s.id === id);
      if (!set) throw new Error(`unknown set ${id}; known: ${manifest.sets.map((s) => s.id).join(', ')}`);
      if (set.minMemoryGb > memoryGb) {
        console.warn(`warning: ${id} needs ${set.minMemoryGb} GB of memory; this machine has ${memoryGb.toFixed(0)} GB, so expect heavy swapping`);
      }
      return set;
    });
  } else {
    chosenSets = manifest.sets.filter((s) => s.default && s.minMemoryGb <= memoryGb);
    const skipped = manifest.sets.filter((s) => s.default && s.minMemoryGb > memoryGb);
    for (const s of skipped) console.log(`skipping ${s.id}: needs ${s.minMemoryGb} GB of memory, this machine has ${memoryGb.toFixed(0)} GB`);
  }
  // Sets share files (FLUX.2 klein and Z-Image use the same Qwen3-4B text encoder).
  wanted = [...new Set(chosenSets.flatMap((s) => s.models))];
}
// The licence gate runs before anything is downloaded.
for (const id of wanted) {
  const model = modelById(id);
  if (!model.commercialUse) throw new Error(`${id} (${model.license}) does not permit commercial use`);
}
if (chosenSets.length) console.log(`Model sets: ${chosenSets.map((s) => s.id).join(', ')}`);

const gpu = arg('gpu') ?? (process.platform === 'darwin' ? 'mps' : 'cuda');
const root = path.resolve(dir, 'ComfyUI');
const python = path.join(root, '.venv', 'bin', 'python');

if (!existsSync(root)) run('git', ['clone', manifest.comfyui.repo, root]);
run('git', ['fetch', '--quiet', 'origin', manifest.comfyui.commit], root);
run('git', ['checkout', '--quiet', manifest.comfyui.commit], root);

if (!existsSync(python)) run('uv', ['venv', '--python', '3.12', '.venv'], root);
// The default PyPI torch wheels carry MPS on macOS and CPU on Linux; CUDA
// needs the PyTorch index.
const torchIndex = gpu === 'cuda' ? ['--index-url', 'https://download.pytorch.org/whl/cu128'] : [];
run('uv', ['pip', 'install', '--python', python, 'torch', 'torchvision', 'torchaudio', ...torchIndex], root);
run('uv', ['pip', 'install', '--python', python, '-r', 'requirements.txt'], root);

for (const node of manifest.customNodes) {
  const target = path.join(root, 'custom_nodes', node.name);
  if (!existsSync(target)) run('git', ['clone', node.repo, target]);
  run('git', ['fetch', '--quiet', 'origin', node.commit], target);
  run('git', ['checkout', '--quiet', node.commit], target);
  if (existsSync(path.join(target, 'requirements.txt'))) run('uv', ['pip', 'install', '--python', python, '-r', 'requirements.txt'], target);
  console.log(`custom node ${node.name}: ${node.commit} (${node.license})`);
}

for (const id of wanted) {
  const model = modelById(id);
  const dest = path.join(root, 'models', model.folder, model.file);
  if (existsSync(dest) && (await sha256(dest)) === model.sha256) {
    console.log(`${model.file}: present, sha256 ok`);
    continue;
  }
  const part = `${dest}.part`;
  run('curl', ['-L', '--fail', '-C', '-', '-o', part, model.url]);
  const digest = await sha256(part);
  if (digest !== model.sha256) {
    rmSync(part);
    throw new Error(`${model.file}: sha256 ${digest} does not match ${model.sha256}`);
  }
  renameSync(part, dest);
  console.log(`${model.file}: installed (${model.license})`);
}

const port = arg('port') ?? '8188';
// --offline keeps ComfyUI off the network: no partner API nodes, no remote
// templates or registry calls. Image generation stays on this machine.
const startArgs = ['main.py', '--listen', '127.0.0.1', '--port', port, '--offline', ...(gpu === 'cpu' ? ['--cpu'] : [])];
if (process.argv.includes('--start')) {
  spawn(python, startArgs, { cwd: root, stdio: 'inherit' });
} else {
  console.log(`\nStart the server with:\n  cd ${root} && .venv/bin/python ${startArgs.join(' ')}`);
  console.log(`Then set the ComfyUI base URL in Open Design to http://127.0.0.1:${port}`);
}
