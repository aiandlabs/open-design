#!/usr/bin/env node
// Install a local ComfyUI server for the `comfyui` media provider and the
// `comfyui-agent` asset loop. Pins ComfyUI to the commit in models.json and
// downloads only models whose licences permit commercial use, verifying each
// file's sha256 before it is moved into place.
//
// Usage:
//   node scripts/comfyui-local/setup.ts --dir <install-dir>
//   node scripts/comfyui-local/setup.ts --dir <install-dir> --gpu cuda
//   node scripts/comfyui-local/setup.ts --dir <install-dir> --models sdxl-base-1.0
//   node scripts/comfyui-local/setup.ts --dir <install-dir> --start --port 8188
//
// --gpu defaults to `mps` on Apple silicon and `cuda` on Linux. Requires git
// and uv on PATH.

import { spawnSync, spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync, readFileSync, renameSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

interface ModelEntry {
  id: string;
  folder: string;
  file: string;
  url: string;
  sha256: string;
  license: string;
  commercialUse: boolean;
  default?: boolean;
}

interface Manifest {
  comfyui: { repo: string; commit: string };
  models: ModelEntry[];
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

const dir = arg('dir');
if (!dir) {
  console.error('usage: node scripts/comfyui-local/setup.ts --dir <install-dir> [--gpu mps|cuda|cpu] [--models id,id] [--start] [--port 8188]');
  process.exit(2);
}
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

const wanted = arg('models')?.split(',') ?? manifest.models.filter((m) => m.default).map((m) => m.id);
for (const id of wanted) {
  const model = manifest.models.find((m) => m.id === id);
  if (!model) throw new Error(`unknown model id ${id}; known: ${manifest.models.map((m) => m.id).join(', ')}`);
  if (!model.commercialUse) throw new Error(`${id} (${model.license}) does not permit commercial use`);
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
const startArgs = ['main.py', '--listen', '127.0.0.1', '--port', port, ...(gpu === 'cpu' ? ['--cpu'] : [])];
if (process.argv.includes('--start')) {
  spawn(python, startArgs, { cwd: root, stdio: 'inherit' });
} else {
  console.log(`\nStart the server with:\n  cd ${root} && .venv/bin/python ${startArgs.join(' ')}`);
  console.log(`Then set the ComfyUI base URL in Open Design to http://127.0.0.1:${port}`);
}
