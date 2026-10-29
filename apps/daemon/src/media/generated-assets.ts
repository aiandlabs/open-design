// Images the ComfyUI provider generated in a project, read from the
// `<name>.trace.json` files renderComfyUI writes beside each image, and the
// user's review of each one, stored in that same trace file. An asset is
// accepted unless a person rejects it or its image file is deleted; accepted
// icons are the style references for the next icon in the project.

import { existsSync } from 'node:fs';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

import type {
  GeneratedAsset,
  GeneratedAssetReview,
  GeneratedAssetReviewState,
  GeneratedAssetReviewVia,
} from '@open-design/contracts';

const TRACE_SUFFIX = '.trace.json';
/** Directories that never hold generated assets (round images live under .comfy-agent/). */
const SKIP_DIRS = new Set(['.comfy-agent', '.git', 'node_modules', '.od', '.hyperframes-cache']);
const MAX_DEPTH = 6;

export class GeneratedAssetNotFoundError extends Error {
  readonly status = 404;
  readonly code = 'GENERATED_ASSET_NOT_FOUND';
}

type TraceFile = {
  kind?: unknown;
  asset?: unknown;
  set?: unknown;
  finishedAt?: unknown;
  request?: { prompt?: unknown; kind?: unknown };
  review?: { state?: unknown; updatedAt?: unknown; via?: unknown };
};

function toPosix(p: string): string {
  return p.split(path.sep).join('/');
}

async function findTraces(root: string, rel = '', depth = 0, out: string[] = []): Promise<string[]> {
  if (depth > MAX_DEPTH) return out;
  let entries;
  try {
    entries = await readdir(path.join(root, rel), { withFileTypes: true });
  } catch {
    return out;
  }
  for (const e of entries) {
    const child = rel ? path.join(rel, e.name) : e.name;
    if (e.isDirectory()) {
      if (!SKIP_DIRS.has(e.name)) await findTraces(root, child, depth + 1, out);
    } else if (e.isFile() && e.name.endsWith(TRACE_SUFFIX)) {
      out.push(child);
    }
  }
  return out;
}

/**
 * Where the image is now. Older media outputs flattened a requested
 * `assets/hero.png` to `assets_hero.png`, and agents then moved the image
 * (and sometimes the trace) into the folder they meant. Before calling an
 * image deleted, look next to the trace and at the unflattened path.
 */
function locateAsset(projectDir: string, recorded: string, traceStem: string): string {
  const ext = path.posix.extname(recorded) || '.png';
  const candidates = [recorded, `${traceStem}${ext}`];
  const base = path.posix.basename(recorded);
  if (base.includes('_')) {
    const dir = path.posix.dirname(recorded);
    const parts = base.split('_');
    for (let i = 1; i < parts.length; i += 1) {
      const nested = `${parts.slice(0, i).join('/')}/${parts.slice(i).join('_')}`;
      candidates.push(dir === '.' ? nested : `${dir}/${nested}`);
    }
  }
  return candidates.find((c) => existsSync(path.join(projectDir, c))) ?? recorded;
}

function assetFromTrace(projectDir: string, traceRel: string, trace: TraceFile): GeneratedAsset | null {
  const isAgent = typeof trace.asset === 'string' && !!trace.request;
  const isWorkflow = trace.kind === 'comfyui-workflow';
  if (!isAgent && !isWorkflow) return null;
  const stem = traceRel.slice(0, -TRACE_SUFFIX.length);
  const recorded = isAgent ? toPosix(trace.asset as string) : `${toPosix(stem)}.png`;
  const assetRel = locateAsset(projectDir, recorded, toPosix(stem));
  const exists = existsSync(path.join(projectDir, assetRel));
  const stored = trace.review;
  const review: GeneratedAssetReview = !exists
    ? { state: 'rejected', reason: 'deleted' }
    : stored && (stored.state === 'accepted' || stored.state === 'rejected')
      ? {
          state: stored.state,
          reason: 'user',
          ...(typeof stored.updatedAt === 'string' ? { updatedAt: stored.updatedAt } : {}),
          ...(stored.via === 'ui' || stored.via === 'cli' || stored.via === 'api' ? { via: stored.via } : {}),
        }
      : { state: 'accepted', reason: 'default' };
  return {
    path: assetRel,
    tracePath: toPosix(traceRel),
    model: isAgent ? 'comfyui-agent' : 'comfyui-workflow',
    ...(trace.request?.kind === 'icon' ? { kind: 'icon' as const } : {}),
    ...(typeof trace.set === 'string' && trace.set ? { set: trace.set } : {}),
    ...(typeof trace.request?.prompt === 'string' ? { prompt: trace.request.prompt } : {}),
    ...(typeof trace.finishedAt === 'string' ? { createdAt: trace.finishedAt } : {}),
    exists,
    review,
  };
}

/** Every ComfyUI-generated asset in the project, sorted by path. */
export async function listGeneratedAssets(projectDir: string): Promise<GeneratedAsset[]> {
  const out: GeneratedAsset[] = [];
  for (const rel of await findTraces(projectDir)) {
    try {
      const trace = JSON.parse(await readFile(path.join(projectDir, rel), 'utf8')) as TraceFile;
      const asset = assetFromTrace(projectDir, rel, trace);
      if (asset) out.push(asset);
    } catch {
      // Not a trace this provider wrote.
    }
  }
  return out.sort((a, b) => a.path.localeCompare(b.path));
}

/** Set a person's review of one asset; stored in its trace file. */
export async function setGeneratedAssetReview(
  projectDir: string,
  assetPath: string,
  state: GeneratedAssetReviewState,
  via: GeneratedAssetReviewVia,
): Promise<GeneratedAsset> {
  const normalized = path.posix.normalize(assetPath.replace(/\\/g, '/'));
  if (path.posix.isAbsolute(normalized) || normalized.startsWith('../') || normalized === '..') {
    throw new GeneratedAssetNotFoundError(`"${assetPath}" is not a path inside the project`);
  }
  const asset = (await listGeneratedAssets(projectDir)).find((a) => a.path === normalized);
  if (!asset) throw new GeneratedAssetNotFoundError(`no generated asset at "${normalized}" (only images made by the ComfyUI provider have a review)`);
  const traceAbs = path.join(projectDir, asset.tracePath);
  const trace = JSON.parse(await readFile(traceAbs, 'utf8')) as TraceFile & Record<string, unknown>;
  trace.review = { state, updatedAt: new Date().toISOString(), via };
  await writeFile(traceAbs, JSON.stringify(trace, null, 2));
  return assetFromTrace(projectDir, asset.tracePath, trace)!;
}

/** The chosen accepted icons, read for the asset loop's `references.set`. */
export async function loadSetReferences(
  projectDir: string,
  want: { set?: string; exclude?: string },
  max: number,
): Promise<{ id: string; label?: string; png: Buffer }[]> {
  const chosen = selectSetReferences(await listGeneratedAssets(projectDir), want, max);
  const out: { id: string; label?: string; png: Buffer }[] = [];
  for (const a of chosen) {
    try {
      out.push({ id: a.path, ...(a.prompt ? { label: a.prompt } : {}), png: await readFile(path.join(projectDir, a.path)) });
    } catch {
      // Deleted between listing and reading.
    }
  }
  return out;
}

/**
 * Accepted icons to use as style references for a new icon: icons whose file
 * still exists and that no one rejected, the requested set first, newest
 * first within each group. `exclude` is the new output's path (a regenerated
 * icon must not reference its own previous version).
 */
export function selectSetReferences(assets: GeneratedAsset[], want: { set?: string; exclude?: string }, max: number): GeneratedAsset[] {
  const usable = assets.filter((a) => a.kind === 'icon' && a.exists && a.review.state === 'accepted' && a.path !== want.exclude);
  const newest = (a: GeneratedAsset, b: GeneratedAsset) => (b.createdAt ?? '').localeCompare(a.createdAt ?? '');
  const inSet = want.set ? usable.filter((a) => a.set === want.set).sort(newest) : [];
  const rest = usable.filter((a) => !inSet.includes(a)).sort(newest);
  return [...inSet, ...rest].slice(0, Math.max(0, max));
}
