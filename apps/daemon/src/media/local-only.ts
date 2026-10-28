// "Local image generation only" — daemon side.
//
// The policy itself (which providers are local, the refusal codes, the
// default) lives in `@open-design/contracts` (`api/media-local-policy.ts`).
// This module resolves it for one daemon (app-config + the OD_MEDIA_LOCAL_ONLY
// process override), refuses hosted dispatches, and annotates the media
// catalogue with readiness for `GET /api/media/models` and `od media models`.

import {
  isLocalMediaProvider,
  isMediaProviderAllowedByPolicy,
  mediaLocalOnlyDenial,
  resolveMediaLocalOnly,
  type LocalComfyImageModel,
  type LocalImageGenerationPromptInput,
  type MediaCatalogPolicy,
  type MediaLocalOnlyCode,
  type MediaModelReadiness,
  type MediaProviderReadiness,
} from '@open-design/contracts';

import { readAppConfig, type AppConfigPrefs } from '../app-config.js';
import { readVelaLoginStatus } from '../integrations/vela.js';
import { COMFYUI_AGENT_MODEL, COMFYUI_WORKFLOW_MODEL } from './comfyui.js';
import { hasStoredProviderEntry, readMaskedConfig, resolveProviderConfig } from './config.js';
import {
  AUDIO_MODELS_BY_KIND,
  IMAGE_MODELS,
  MEDIA_PROVIDERS,
  VIDEO_MODELS,
  type AudioKind,
  type MediaModel,
} from './models.js';

/**
 * Process-level override, for operators and test harnesses: `0`/`false`/`off`
 * turns the policy off, `1`/`true`/`on` turns it on, whatever app-config says.
 * Unset (the normal case) defers to Settings.
 */
export const MEDIA_LOCAL_ONLY_ENV = 'OD_MEDIA_LOCAL_ONLY';

export type MediaLocalOnlySource = 'env' | 'config' | 'default';

export function mediaLocalOnlyFromEnv(env: NodeJS.ProcessEnv = process.env): boolean | null {
  const raw = env[MEDIA_LOCAL_ONLY_ENV]?.trim().toLowerCase();
  if (!raw) return null;
  if (raw === '0' || raw === 'false' || raw === 'off' || raw === 'no') return false;
  if (raw === '1' || raw === 'true' || raw === 'on' || raw === 'yes') return true;
  return null;
}

/** Resolve the policy from already-read app-config prefs. */
export function resolveMediaLocalOnlyState(
  config: Pick<AppConfigPrefs, 'media'> | null | undefined,
  env: NodeJS.ProcessEnv = process.env,
): { localOnly: boolean; source: MediaLocalOnlySource } {
  const fromEnv = mediaLocalOnlyFromEnv(env);
  if (fromEnv !== null) return { localOnly: fromEnv, source: 'env' };
  const stored = config?.media;
  if (stored && typeof stored.localOnly === 'boolean') {
    return { localOnly: stored.localOnly, source: 'config' };
  }
  return { localOnly: resolveMediaLocalOnly(undefined), source: 'default' };
}

/**
 * Read the policy for this daemon. An unreadable app-config resolves to the
 * default (on), never to off: a broken file must not open the hosted routes.
 */
export async function readMediaLocalOnly(
  dataDir: string,
  env: NodeJS.ProcessEnv = process.env,
): Promise<{ localOnly: boolean; source: MediaLocalOnlySource }> {
  const config = await readAppConfig(dataDir).catch(() => null);
  return resolveMediaLocalOnlyState(config, env);
}

/**
 * Which ComfyUI image model can run: `comfyui-agent` when ComfyUI is set up in
 * media-config and the ai& key for its planner and vision model resolves
 * (stored or `AIAND_API_KEY`), `comfyui-workflow` when only the server is set
 * up, null when ComfyUI is not configured. Env-only keys do not opt a user in
 * (see `hasStoredProviderEntry`).
 */
export async function resolveComfyImageModel(projectRoot: string): Promise<LocalComfyImageModel | null> {
  const stored = await hasStoredProviderEntry(projectRoot, 'comfyui').catch(() => false);
  if (!stored) return null;
  const creds = await resolveProviderConfig(projectRoot, 'comfyui').catch(() => null);
  return creds?.apiKey ? COMFYUI_AGENT_MODEL : COMFYUI_WORKFLOW_MODEL;
}

/** Prompt input for both sides of the prompt fork. */
export async function resolveLocalImageGenerationForPrompt(input: {
  projectRoot: string;
  config: Pick<AppConfigPrefs, 'media'> | null | undefined;
  env?: NodeJS.ProcessEnv;
}): Promise<LocalImageGenerationPromptInput> {
  const { localOnly } = resolveMediaLocalOnlyState(input.config, input.env);
  return { localOnly, comfyuiModel: await resolveComfyImageModel(input.projectRoot) };
}

/** Thrown by `generateMedia` and the generate route for a refused dispatch. */
export class MediaLocalOnlyError extends Error {
  readonly code: MediaLocalOnlyCode;
  readonly status = 403;
  readonly retryable = false;
  readonly localModel: string | undefined;

  constructor(code: MediaLocalOnlyCode, message: string, localModel?: string) {
    super(message);
    this.name = 'MediaLocalOnlyError';
    this.code = code;
    this.localModel = localModel;
  }
}

/**
 * Throw {@link MediaLocalOnlyError} when the policy refuses this dispatch.
 * `providerId` is the catalogue provider of the resolved model (or `fal` for a
 * raw `fal-ai/*` path, `aihubmix` for a live AIHubMix id).
 */
export async function assertMediaDispatchAllowed(input: {
  projectRoot: string;
  localOnly: boolean;
  surface: string;
  providerId: string;
  model: string;
}): Promise<void> {
  if (isMediaProviderAllowedByPolicy(input)) return;
  const comfyuiModel = input.surface === 'image'
    ? await resolveComfyImageModel(input.projectRoot)
    : null;
  const denial = mediaLocalOnlyDenial({ ...input, comfyuiModel });
  if (denial) throw new MediaLocalOnlyError(denial.code, denial.message, denial.localModel);
}

export type AnnotatedMediaModel = MediaModel & MediaModelReadiness;

export interface MediaCatalogResponse {
  providers: typeof MEDIA_PROVIDERS;
  readiness: Record<string, MediaProviderReadiness>;
  policy: MediaCatalogPolicy & { source: MediaLocalOnlySource };
  image: AnnotatedMediaModel[];
  video: AnnotatedMediaModel[];
  audio: Record<AudioKind, AnnotatedMediaModel[]>;
}

/**
 * Whether each provider has what it needs to run: a key (stored, env, or the
 * borrowed auth file `readMaskedConfig` already resolves), a base URL for
 * `custom-image`, a stored entry for ComfyUI, a Vela sign-in for `vela`, and
 * nothing for the local HyperFrames renderer. Providers with no renderer
 * (`integrated: false`) are never configured.
 */
export async function readMediaProviderConfigured(
  projectRoot: string,
  config: Pick<AppConfigPrefs, 'agentCliEnv'> | null | undefined,
  env: NodeJS.ProcessEnv = process.env,
): Promise<Record<string, boolean>> {
  const masked = await readMaskedConfig(projectRoot).catch(() => null);
  const comfyStored = await hasStoredProviderEntry(projectRoot, 'comfyui').catch(() => false);
  let velaSignedIn = false;
  try {
    velaSignedIn = readVelaLoginStatus(env, config?.agentCliEnv?.amr ?? {}).sessionState === 'authenticated';
  } catch {
    velaSignedIn = false;
  }
  const out: Record<string, boolean> = {};
  for (const provider of MEDIA_PROVIDERS) {
    const entry = masked?.providers[provider.id];
    let configured: boolean;
    if (!provider.integrated) configured = false;
    else if (provider.id === 'comfyui') configured = comfyStored;
    else if (provider.id === 'hyperframes') configured = true;
    else if (provider.id === 'stub') configured = env.OD_MEDIA_ALLOW_STUBS === '1' || env.OD_MEDIA_ALLOW_STUBS === 'true';
    else if (provider.id === 'vela') configured = velaSignedIn;
    else if (provider.id === 'custom-image') configured = Boolean(entry?.baseUrl);
    else configured = Boolean(entry?.configured);
    out[provider.id] = configured;
  }
  return out;
}

function annotate(
  models: readonly MediaModel[],
  surface: 'image' | 'video' | 'audio',
  localOnly: boolean,
  configured: Record<string, boolean>,
  comfyuiModel: LocalComfyImageModel | null,
): AnnotatedMediaModel[] {
  return models.map((model) => {
    const { default: _default, ...rest } = model;
    let isConfigured = configured[model.provider] === true;
    // comfyui-agent also needs the ai& key for its planner and vision model.
    if (model.id === COMFYUI_AGENT_MODEL) isConfigured = comfyuiModel === COMFYUI_AGENT_MODEL;
    const allowed = isMediaProviderAllowedByPolicy({ providerId: model.provider, surface, localOnly });
    return { ...rest, configured: isConfigured, allowed, ready: isConfigured && allowed };
  });
}

/**
 * Mark exactly one default per list, and only on a ready model: the
 * catalogue's own default when it is ready, else (for images) the ready
 * ComfyUI model, else the first ready model. A list with no ready model has
 * no default.
 */
function withReadyDefault(
  models: AnnotatedMediaModel[],
  catalogue: readonly MediaModel[],
  preferred: readonly string[] = [],
): AnnotatedMediaModel[] {
  const ready = models.filter((m) => m.ready);
  const catalogueDefaults = catalogue.filter((m) => m.default).map((m) => m.id);
  const pick = [...preferred, ...catalogueDefaults]
    .map((id) => ready.find((m) => m.id === id))
    .find(Boolean) ?? ready[0];
  return models.map((m) => (pick && m.id === pick.id ? { ...m, default: true } : m));
}

/**
 * The catalogue with readiness. `view: 'agent'` (what `od media models`
 * shows an agent) drops image and video models the policy refuses; `full`
 * (Settings) keeps them, marked `allowed: false`.
 */
export async function buildMediaCatalog(input: {
  projectRoot: string;
  config: Pick<AppConfigPrefs, 'media' | 'agentCliEnv'> | null | undefined;
  view: 'agent' | 'full';
  env?: NodeJS.ProcessEnv;
}): Promise<MediaCatalogResponse> {
  const env = input.env ?? process.env;
  const { localOnly, source } = resolveMediaLocalOnlyState(input.config, env);
  const configured = await readMediaProviderConfigured(input.projectRoot, input.config, env);
  const comfyuiModel = await resolveComfyImageModel(input.projectRoot);
  const readiness: Record<string, MediaProviderReadiness> = {};
  for (const provider of MEDIA_PROVIDERS) {
    readiness[provider.id] = {
      configured: configured[provider.id] === true,
      local: isLocalMediaProvider(provider.id),
      // Audio-only providers stay allowed; the policy covers image and video.
      allowed: isMediaProviderAllowedByPolicy({ providerId: provider.id, surface: 'image', localOnly })
        || !providerServesImageOrVideo(provider.id),
    };
  }
  const keep = (m: AnnotatedMediaModel) => input.view === 'full' || m.allowed;
  const imagePreferred = comfyuiModel ? [comfyuiModel] : [];
  const image = withReadyDefault(
    annotate(IMAGE_MODELS, 'image', localOnly, configured, comfyuiModel),
    IMAGE_MODELS,
    imagePreferred,
  ).filter(keep);
  const video = withReadyDefault(
    annotate(VIDEO_MODELS, 'video', localOnly, configured, comfyuiModel),
    VIDEO_MODELS,
  ).filter(keep);
  const audio = {} as Record<AudioKind, AnnotatedMediaModel[]>;
  for (const kind of Object.keys(AUDIO_MODELS_BY_KIND) as AudioKind[]) {
    audio[kind] = withReadyDefault(
      annotate(AUDIO_MODELS_BY_KIND[kind], 'audio', localOnly, configured, comfyuiModel),
      AUDIO_MODELS_BY_KIND[kind],
    );
  }
  return {
    providers: MEDIA_PROVIDERS,
    readiness,
    policy: { localOnly, view: input.view, source },
    image,
    video,
    audio,
  };
}

const IMAGE_OR_VIDEO_PROVIDERS: ReadonlySet<string> = new Set(
  [...IMAGE_MODELS, ...VIDEO_MODELS].map((m) => m.provider),
);

function providerServesImageOrVideo(providerId: string): boolean {
  return IMAGE_OR_VIDEO_PROVIDERS.has(providerId);
}
