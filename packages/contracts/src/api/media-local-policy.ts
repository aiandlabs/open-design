// Local image generation policy ("Local image generation only").
//
// One app-config setting, `media.localOnly`, decides whether the daemon may
// dispatch hosted image and video providers. When it is on, the only image
// route is the local ComfyUI provider (`comfyui-agent`, `comfyui-workflow`),
// and the only video route is the local HyperFrames renderer. The daemon
// refuses every other image or video provider in `generateMedia` itself, the
// media catalogue marks them as not allowed, and both prompt sides tell the
// agent which local route to use (or that there is none).
//
// Absent means ON. This fork ships local-only by default; a user turns it off
// in Settings → Media providers or with `od media policy --local-only off`.
//
// Audio (music, speech, sfx) is out of scope: the setting does not change
// which audio providers may run.

import { SETTINGS_MEDIA_PROVIDERS_PATH } from '../settings-nav.js';

/** App-config `media` block. */
export interface MediaPolicyPrefs {
  /** Refuse hosted image and video providers. Absent resolves to `true`. */
  localOnly?: boolean;
}

/** The value `media.localOnly` resolves to when the key is absent. */
export const MEDIA_LOCAL_ONLY_DEFAULT = true;

/**
 * Normalise a stored or incoming `media` block. Unknown keys and non-boolean
 * values are dropped; an empty result is `undefined` so the default applies.
 */
export function normalizeMediaPolicyPrefs(value: unknown): MediaPolicyPrefs | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  const raw = value as Record<string, unknown>;
  return typeof raw.localOnly === 'boolean' ? { localOnly: raw.localOnly } : undefined;
}

/** Whether local-only is in effect for these prefs (absent or malformed → default on). */
export function resolveMediaLocalOnly(prefs: MediaPolicyPrefs | null | undefined): boolean {
  const normalized = normalizeMediaPolicyPrefs(prefs);
  return normalized?.localOnly ?? MEDIA_LOCAL_ONLY_DEFAULT;
}

/**
 * Media providers whose generation never leaves the machine (or the user's own
 * server), and so stay allowed under local-only.
 *
 * - `comfyui`: the user's own ComfyUI server. `@open-design/comfy-agent`
 *   refuses ComfyUI partner/API nodes (`api_node`) before it queues a graph,
 *   so a graph cannot reach a cloud model through the server. The
 *   `comfyui-agent` planner and vision critic call the ai& LLM API, which is
 *   the company's own platform; no image or video model runs there.
 * - `hyperframes`: renders agent-written HTML to MP4 with a local Chrome and
 *   ffmpeg inside the daemon process.
 * - `stub`: writes deterministic placeholder bytes locally (tests only).
 *
 * Every other image or video provider sends the prompt to a hosted API,
 * including `custom-image` (an OpenAI-compatible URL the daemon cannot prove
 * is local: a loopback URL can still be a proxy to a cloud API) and `vela`
 * (OpenDesign Cloud).
 */
export const LOCAL_MEDIA_PROVIDER_IDS = ['comfyui', 'hyperframes', 'stub'] as const;
export type LocalMediaProviderId = (typeof LOCAL_MEDIA_PROVIDER_IDS)[number];

const LOCAL_MEDIA_PROVIDER_SET: ReadonlySet<string> = new Set(LOCAL_MEDIA_PROVIDER_IDS);

/** Media surfaces the local-only policy governs. Audio is out of scope. */
export const MEDIA_LOCAL_ONLY_SURFACES = ['image', 'video'] as const;

export function isLocalMediaProvider(providerId: string | null | undefined): boolean {
  return typeof providerId === 'string' && LOCAL_MEDIA_PROVIDER_SET.has(providerId);
}

/**
 * Whether the policy allows this provider on this surface. Audio is always
 * allowed; image and video need a local provider when local-only is on.
 */
export function isMediaProviderAllowedByPolicy(
  input: { providerId: string; surface: string; localOnly: boolean },
): boolean {
  if (!input.localOnly) return true;
  if (input.surface !== 'image' && input.surface !== 'video') return true;
  return isLocalMediaProvider(input.providerId);
}

/**
 * Refusal codes. Each maps to a `nextStep` in `mediaFailureNextStep`:
 *
 * - `MEDIA_LOCAL_ONLY_SWITCH_MODEL` — ComfyUI is configured, so the agent
 *   dispatches the image once more with the local model (`switch-model`).
 * - `MEDIA_LOCAL_ONLY_SETUP_COMFYUI` — no local image model is configured;
 *   the user has to set up ComfyUI in Settings → Media providers
 *   (`open-settings`).
 * - `MEDIA_LOCAL_ONLY_VIDEO` — there is no local generative video model.
 *   HyperFrames renders HTML motion only, which is not a substitute for a
 *   generated clip (`unsupported`).
 */
export const MEDIA_LOCAL_ONLY_CODES = [
  'MEDIA_LOCAL_ONLY_SWITCH_MODEL',
  'MEDIA_LOCAL_ONLY_SETUP_COMFYUI',
  'MEDIA_LOCAL_ONLY_VIDEO',
] as const;
export type MediaLocalOnlyCode = (typeof MEDIA_LOCAL_ONLY_CODES)[number];

export interface MediaLocalOnlyDenial {
  code: MediaLocalOnlyCode;
  message: string;
  /** The local model to use instead, when one is ready. */
  localModel?: string;
}

/**
 * The refusal for a hosted image/video provider under local-only, or null when
 * the dispatch is allowed. `comfyuiModel` is the ready ComfyUI model id, or
 * null when ComfyUI is not configured.
 */
export function mediaLocalOnlyDenial(input: {
  localOnly: boolean;
  surface: string;
  providerId: string;
  model: string;
  comfyuiModel: string | null;
}): MediaLocalOnlyDenial | null {
  if (isMediaProviderAllowedByPolicy(input)) return null;
  const head = `media model "${input.model}" uses the hosted provider "${input.providerId}", and Local image generation only is on`;
  if (input.surface === 'video') {
    return {
      code: 'MEDIA_LOCAL_ONLY_VIDEO',
      message: `${head}. No local generative video model is available; the local HyperFrames renderer (--model hyperframes-html) renders HTML motion graphics only.`,
    };
  }
  if (input.comfyuiModel) {
    return {
      code: 'MEDIA_LOCAL_ONLY_SWITCH_MODEL',
      message: `${head}. Use the local ComfyUI model instead (--model ${input.comfyuiModel}).`,
      localModel: input.comfyuiModel,
    };
  }
  return {
    code: 'MEDIA_LOCAL_ONLY_SETUP_COMFYUI',
    message: `${head}, but ComfyUI is not configured. Set up ComfyUI under ${SETTINGS_MEDIA_PROVIDERS_PATH} → ComfyUI, or turn off Local image generation only there.`,
  };
}

/** Per-provider readiness in `GET /api/media/models`. */
export interface MediaProviderReadiness {
  /** Credentials, URL, or a local runtime are present. */
  configured: boolean;
  /** Generation stays on the machine or the user's own server. */
  local: boolean;
  /** The local-only policy lets the daemon dispatch this provider for image/video. */
  allowed: boolean;
}

/** Readiness fields added to each model entry in `GET /api/media/models`. */
export interface MediaModelReadiness {
  configured: boolean;
  allowed: boolean;
  /** configured && allowed. */
  ready: boolean;
}

/** `policy` block of `GET /api/media/models`. */
export interface MediaCatalogPolicy {
  localOnly: boolean;
  /** `agent` lists only allowed image/video models; `full` lists everything. */
  view: 'agent' | 'full';
}
