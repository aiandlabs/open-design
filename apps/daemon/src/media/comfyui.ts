// ComfyUI media provider. Two models:
//
//   comfyui-workflow  runs a user-supplied API/UI-format graph once, with the
//                     prompt, size and seed slots filled; with no graph it
//                     runs the default text-to-image base graph.
//   comfyui-agent     the plan → generate → check → critique → repair loop
//                     from @open-design/comfy-agent, using the ai& API for
//                     the planner and the vision critic.
//
// Provider config (Settings → Media providers → ComfyUI, media-config.json):
//   baseUrl  ComfyUI server URL (default http://127.0.0.1:8188)
//   apiKey   ai& API key for comfyui-agent (env AIAND_API_KEY also works)
//   model    planner/critic/API shape, e.g.
//            "planner=moonshotai/kimi-k3, critic=zai-org/glm-5.3-flash, shape=openai"
//
// Bytes are returned to generateMedia like every other renderer. The trace
// (and the graph that ran) is written next to the asset as
// `<name>.trace.json`; per-round images go under `.comfy-agent/<name>/` in the
// project so the trace's image paths resolve inside the project.

import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { SETTINGS_MEDIA_PROVIDERS_PATH } from '@open-design/contracts';
import {
  AiandClient,
  ComfyAgentError,
  ComfyClient,
  LlmHttpError,
  generateAsset,
  parseModelSettings,
  resolveComfyAgentConfig,
  runWorkflow,
  toApiWorkflow,
  type ComfyAgentConfig,
} from '@open-design/comfy-agent';

export const COMFYUI_AGENT_MODEL = 'comfyui-agent';
export const COMFYUI_WORKFLOW_MODEL = 'comfyui-workflow';

export type ComfyMediaOptions = {
  /** Workflow JSON (API or UI format), already parsed. */
  workflow?: unknown;
  maxRounds?: number;
  width?: number;
  height?: number;
  transparent?: boolean;
  negative?: string;
  seed?: number;
};

type Credentials = { apiKey?: string; baseUrl?: string; model?: string };

/** Read the `comfy` options object from a media generate request body. */
export function parseComfyMediaOptions(body: unknown): ComfyMediaOptions | undefined {
  if (!body || typeof body !== 'object') return undefined;
  const raw = (body as { comfy?: unknown }).comfy;
  if (!raw || typeof raw !== 'object') return undefined;
  const r = raw as Record<string, unknown>;
  const int = (v: unknown, min: number, max: number) =>
    typeof v === 'number' && Number.isInteger(v) && v >= min && v <= max ? v : undefined;
  const out: ComfyMediaOptions = {};
  if (r.workflow && typeof r.workflow === 'object') out.workflow = r.workflow;
  const maxRounds = int(r.maxRounds, 1, 10);
  if (maxRounds !== undefined) out.maxRounds = maxRounds;
  const width = int(r.width, 16, 8192);
  const height = int(r.height, 16, 8192);
  if (width !== undefined && height !== undefined) {
    out.width = width;
    out.height = height;
  }
  if (r.transparent === true) out.transparent = true;
  if (typeof r.negative === 'string' && r.negative.trim()) out.negative = r.negative.trim();
  const seed = int(r.seed, 0, 2 ** 32 - 1);
  if (seed !== undefined) out.seed = seed;
  return Object.keys(out).length > 0 ? out : undefined;
}

/** Output size for an `--aspect` when no explicit size was given (SDXL buckets). */
export function comfySizeForAspect(aspect: string | undefined): { width: number; height: number } {
  switch (aspect) {
    case '16:9':
      return { width: 1344, height: 768 };
    case '9:16':
      return { width: 768, height: 1344 };
    case '4:3':
      return { width: 1152, height: 896 };
    case '3:4':
      return { width: 896, height: 1152 };
    default:
      return { width: 1024, height: 1024 };
  }
}

export function resolveComfyConfig(credentials: Credentials, env: NodeJS.ProcessEnv = process.env): ComfyAgentConfig {
  return resolveComfyAgentConfig(
    {
      ...(credentials.baseUrl ? { comfyUrl: credentials.baseUrl } : {}),
      ...(credentials.apiKey ? { aiandApiKey: credentials.apiKey } : {}),
      ...parseModelSettings(credentials.model),
    },
    env,
  );
}

/**
 * Is the ComfyUI provider set up in media-config? Used to gate the agent
 * prompt section: env AIAND_API_KEY alone does not opt a user into ComfyUI.
 */
export function isComfyStoredEntryConfigured(entry: Credentials | undefined): boolean {
  return Boolean(entry && (entry.baseUrl?.trim() || entry.model?.trim() || entry.apiKey?.trim()));
}

/** Turn a comfy-agent failure into the daemon's media error shape (code + status). */
export function toMediaError(err: unknown): Error {
  if (err instanceof LlmHttpError) {
    const e = new Error(`comfyui-agent ai& ${err.status || 'request'}: ${err.message}`) as Error & { code?: string; status?: number };
    e.code = err.code;
    if (err.status) e.status = err.status;
    return e;
  }
  if (err instanceof ComfyAgentError) {
    const message = err.code === 'LLM_CONFIG'
      ? `no ComfyUI agent API key — add the ai& API key under ${SETTINGS_MEDIA_PROVIDERS_PATH} → ComfyUI, or set AIAND_API_KEY.`
      : err.message;
    const e = new Error(message) as Error & { code?: string; status?: number; retryable?: boolean };
    e.code = err.code;
    e.status =
      err.code === 'COMFY_CONNECTION' ? 503
        : err.code === 'COMFY_TIMEOUT' ? 504
          : err.code === 'LLM_CONFIG' ? 400
            : 422;
    if (err.code === 'COMFY_TIMEOUT' || err.code === 'COMFY_OUTPUT_FETCH') e.retryable = true;
    return e;
  }
  return err instanceof Error ? err : new Error(String(err));
}

export async function renderComfyUI(input: {
  model: string;
  prompt: string;
  aspect: string | undefined;
  projectDir: string;
  /** Sanitised project-relative output name; the trace is written beside it. */
  outputName: string;
  credentials: Credentials;
  options?: ComfyMediaOptions | undefined;
  requestInit?: Pick<RequestInit, 'dispatcher'>;
  onProgress?: ((message: string) => void) | undefined;
}): Promise<{ bytes: Buffer; providerNote: string; suggestedExt: string }> {
  const cfg = resolveComfyConfig(input.credentials);
  const opts = input.options ?? {};
  const size = opts.width && opts.height ? { width: opts.width, height: opts.height } : comfySizeForAspect(input.aspect);
  const comfy = new ComfyClient({ baseUrl: cfg.comfyUrl });
  const stem = stripExt(input.outputName);
  const tracePath = path.join(input.projectDir, `${stem}.trace.json`);
  await mkdir(path.dirname(tracePath), { recursive: true });
  const rel = (abs: string) => path.relative(input.projectDir, abs).split(path.sep).join('/');

  try {
    if (input.model === COMFYUI_WORKFLOW_MODEL) {
      const result = await runWorkflow(comfy, {
        ...(opts.workflow !== undefined ? { workflow: opts.workflow } : {}),
        ...(input.prompt ? { prompt: input.prompt } : {}),
        ...(opts.negative ? { negative: opts.negative } : {}),
        width: size.width,
        height: size.height,
        ...(opts.seed !== undefined ? { seed: opts.seed } : {}),
        onProgress: (e) => {
          if (e.type === 'progress' && e.max > 0 && (e.value === e.max || e.value % 5 === 0)) {
            input.onProgress?.(`comfyui step ${e.value}/${e.max}`);
          }
        },
      });
      await writeFile(
        tracePath,
        JSON.stringify(
          {
            kind: 'comfyui-workflow',
            server: cfg.comfyUrl,
            promptId: result.promptId,
            durationMs: result.durationMs,
            settings: result.settings,
            unfilled: result.unfilled,
            graph: result.graph,
          },
          null,
          2,
        ),
      );
      const note = `comfyui/${COMFYUI_WORKFLOW_MODEL} · ${size.width}x${size.height} · ${(result.durationMs / 1000).toFixed(1)}s · trace ${rel(tracePath)}`;
      return {
        bytes: result.png,
        providerNote: result.unfilled.length ? `${note} · unfilled: ${result.unfilled.join(', ')}` : note,
        suggestedExt: '.png',
      };
    }

    // comfyui-agent
    const fetchWithInit: typeof fetch = (url, init) => fetch(url, { ...init, ...(input.requestInit ?? {}) } as RequestInit);
    const llm = new AiandClient({
      ...(cfg.aiandApiKey ? { apiKey: cfg.aiandApiKey } : {}),
      baseUrl: cfg.aiandBaseUrl,
      shape: cfg.apiShape,
      fetch: fetchWithInit,
    });
    const workDir = path.join(input.projectDir, '.comfy-agent', path.basename(stem));
    const baseGraph = opts.workflow !== undefined ? toApiWorkflow(opts.workflow, await comfy.getObjectInfo()) : undefined;
    const result = await generateAsset(
      {
        prompt: input.prompt,
        width: size.width,
        height: size.height,
        ...(opts.transparent ? { transparent: true } : {}),
        ...(opts.negative ? { negative: opts.negative } : {}),
        ...(opts.seed !== undefined ? { seed: opts.seed } : {}),
      },
      {
        comfy,
        llm,
        plannerModel: cfg.plannerModel,
        criticModel: cfg.criticModel,
        ...(cfg.plannerReasoningEffort ? { plannerReasoningEffort: cfg.plannerReasoningEffort } : { plannerReasoningEffort: 'low' }),
        ...(cfg.criticReasoningEffort ? { criticReasoningEffort: cfg.criticReasoningEffort } : { criticReasoningEffort: 'low' }),
        workDir,
        tracePath: rel,
        limits: { maxRounds: opts.maxRounds ?? 3 },
        ...(baseGraph ? { baseGraph } : {}),
        ...(input.onProgress ? { onProgress: input.onProgress } : {}),
      },
    );
    const trace = { ...result.trace, asset: input.outputName.replace(/\.[^.]+$/, '.png') };
    await writeFile(tracePath, JSON.stringify(trace, null, 2));
    const best = result.trace.best;
    const cost = result.trace.totals.costUsd;
    return {
      bytes: result.png,
      providerNote:
        `comfyui/${COMFYUI_AGENT_MODEL} · ${size.width}x${size.height} · ${result.trace.rounds.length} round(s) · ` +
        `${best ? `${best.score.passed}/${best.score.total} checks` : 'no score'} · ${result.trace.stopReason} · ` +
        `$${cost.toFixed(4)} · trace ${rel(tracePath)}`,
      suggestedExt: '.png',
    };
  } catch (err) {
    throw toMediaError(err);
  }
}

function stripExt(name: string): string {
  const dot = name.lastIndexOf('.');
  const slash = name.lastIndexOf('/');
  return dot > slash && dot > 0 ? name.slice(0, dot) : name;
}
