// Visual check orchestration: render → grade → save. Pure over its injected
// dependencies so the route test can stub the renderer and the ai& call.

import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

import {
  VISUAL_CHECK_DEFAULT_MODEL,
  type VisualCheckPrefs,
  type VisualCheckRequest,
  type VisualCheckResponse,
  type VisualCheckScreenshot,
  type VisualCheckViewportName,
} from '@open-design/contracts';
import type { AiandClient } from '@open-design/comfy-agent';

import { gradeVisualCheck } from './grade.js';
import type { VisualCheckRenderResult } from './render.js';

/** Project-relative folder for screenshots and result records. */
export const VISUAL_CHECK_DIR = '.visual-check';

const SCREENSHOT_PATH_RE = /^\.visual-check\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+\.png$/;

/** Is this a project-relative visual check screenshot path we would have written? */
export function isVisualCheckScreenshotPath(value: unknown): value is string {
  return typeof value === 'string' && SCREENSHOT_PATH_RE.test(value) && !value.includes('..');
}

export class VisualCheckConfigError extends Error {
  readonly code = 'VISUAL_CHECK_NOT_CONFIGURED';
}

export interface VisualCheckServiceDeps {
  render: (input: { fileName: string; viewports: VisualCheckViewportName[] }) => Promise<VisualCheckRenderResult>;
  /** Returns null when no ai& API key is configured. */
  createClient: () => Promise<Pick<AiandClient, 'structured'> | null>;
  readPrefs: () => Promise<VisualCheckPrefs | null | undefined>;
  /** The project's original prompt, used when the request has no brief or checklist. */
  readProjectPrompt: () => Promise<string | null>;
  /** Absolute project folder (managed or imported). */
  projectDir: string;
  now?: () => Date;
}

function stamp(date: Date): string {
  const pad = (n: number, w = 2) => String(n).padStart(w, '0');
  return `${date.getUTCFullYear()}${pad(date.getUTCMonth() + 1)}${pad(date.getUTCDate())}-${pad(date.getUTCHours())}${pad(date.getUTCMinutes())}${pad(date.getUTCSeconds())}-${pad(date.getUTCMilliseconds(), 3)}`;
}

function fileStem(fileName: string): string {
  const base = path.posix.basename(fileName.replace(/\\/g, '/'));
  const stem = base.replace(/\.[^.]+$/, '') || 'artifact';
  return stem.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^\.+/, '_').slice(0, 80) || 'artifact';
}

export async function runVisualCheck(
  request: VisualCheckRequest,
  deps: VisualCheckServiceDeps,
): Promise<VisualCheckResponse> {
  const started = Date.now();
  const client = await deps.createClient();
  if (!client) {
    throw new VisualCheckConfigError(
      'no ai& API key for visual check: add it under Settings → Media providers → ComfyUI, or set AIAND_API_KEY',
    );
  }
  const prefs = await deps.readPrefs().catch(() => null);
  const model = request.model ?? prefs?.model ?? VISUAL_CHECK_DEFAULT_MODEL;

  let brief = request.brief;
  let briefSource: 'brief' | 'project-prompt' | undefined = brief ? 'brief' : undefined;
  if (!request.checklist && !brief) {
    const projectPrompt = (await deps.readProjectPrompt().catch(() => null))?.trim();
    if (projectPrompt) {
      brief = projectPrompt.slice(0, 8000);
      briefSource = 'project-prompt';
    }
  }

  const viewports = request.viewports ?? ['desktop'];
  const rendered = await deps.render({ fileName: request.fileName, viewports });

  const graded = await gradeVisualCheck(client, {
    images: rendered.images,
    checklist: request.checklist,
    brief,
    briefSource,
    model,
  });

  const when = (deps.now ?? (() => new Date()))();
  const relDir = path.posix.join(VISUAL_CHECK_DIR, fileStem(request.fileName));
  const absDir = path.join(deps.projectDir, ...relDir.split('/'));
  await mkdir(absDir, { recursive: true });
  const prefix = stamp(when);
  const screenshots: VisualCheckScreenshot[] = [];
  for (const image of rendered.images) {
    const name = `${prefix}-${image.viewport}.png`;
    await writeFile(path.join(absDir, name), image.png);
    screenshots.push({
      viewport: image.viewport,
      width: image.width,
      height: image.height,
      path: `${relDir}/${name}`,
      bytes: image.png.length,
    });
  }

  const response: VisualCheckResponse = {
    verdict: graded.items.length > 0 && graded.items.every((item) => item.pass) ? 'pass' : 'fail',
    summary: graded.summary,
    items: graded.items,
    checklistSource: graded.checklistSource,
    screenshots,
    renderer: rendered.renderer,
    model: graded.model,
    apiShape: graded.apiShape,
    usage: graded.usage,
    ...(graded.estimatedCostUsd !== undefined ? { estimatedCostUsd: graded.estimatedCostUsd } : {}),
    durationMs: Date.now() - started,
  };
  // The record sits beside the screenshots, like the asset loop's traces.
  await writeFile(
    path.join(absDir, `${prefix}.json`),
    JSON.stringify({ fileName: request.fileName, request, ...response }, null, 2),
  );
  return response;
}
