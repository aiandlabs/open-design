// Per-model token limits for a BYOK run, read from the provider's own
// `/v1/models` list. ai& (aiand.com) lists `max_output_tokens` and
// `context_window` on every entry of its public OpenAI-shape list, which the
// same host serves to both its OpenAI and Anthropic presets. Only hosts known
// to publish limits on a public list are asked, so a run against any other
// provider makes no extra request and keeps its defaults.
//
// The request carries no credentials: the list is public, and a key is never
// sent to fetch metadata.

import type { ByokChatProviderConfig } from '@open-design/contracts';

import { validateUserProviderBaseUrl } from '../connectionTest.js';
import { listedContextWindow } from './model-list-fields.js';

export interface ByokModelLimits {
  output?: number;
  context?: number;
}

interface ResolveOptions {
  fetchImpl?: typeof fetch;
  validateBaseUrl?: (baseUrl: string) => Promise<{ error?: string; parsed?: URL; forbidden?: boolean }>;
  timeoutMs?: number;
}

const LIMITS_TIMEOUT_MS = 3_000;
const CACHE_TTL_MS = 10 * 60_000;
const PROTOCOLS_WITH_LISTED_LIMITS = new Set<ByokChatProviderConfig['protocol']>(['openai', 'anthropic']);
/** Hosts whose public `/v1/models` list carries per-model limits. */
const HOSTS_WITH_LISTED_LIMITS = new Set(['api.aiand.com']);

const cache = new Map<string, { at: number; byModel: Map<string, ByokModelLimits> | null }>();

export function clearByokModelLimitsCache(): void {
  cache.clear();
}

function positiveInteger(raw: unknown): number | undefined {
  const n = typeof raw === 'string' ? Number(raw) : raw;
  return typeof n === 'number' && Number.isInteger(n) && n > 0 ? n : undefined;
}

/** `max_output_tokens` and `context_window` of one list entry, when present. */
export function listedModelLimits(item: unknown): ByokModelLimits {
  if (!item || typeof item !== 'object') return {};
  const output = positiveInteger((item as { max_output_tokens?: unknown }).max_output_tokens);
  const context = listedContextWindow(item);
  return {
    ...(output !== undefined ? { output } : {}),
    ...(context !== undefined ? { context } : {}),
  };
}

function modelsUrl(baseUrl: string): string {
  const url = new URL(baseUrl.trim().replace(/\/+$/, ''));
  const pathname = url.pathname.replace(/\/+$/, '');
  url.pathname = /\/v\d+(\/|$)/.test(pathname) ? `${pathname}/models` : `${pathname}/v1/models`;
  return url.toString();
}

async function fetchLimitsList(
  url: string,
  fetchImpl: typeof fetch,
  timeoutMs: number,
): Promise<Map<string, ByokModelLimits> | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchImpl(url, { method: 'GET', signal: controller.signal, redirect: 'error' });
    if (!response.ok) return null;
    const body = (await response.json()) as { data?: unknown };
    if (!Array.isArray(body?.data)) return null;
    const byModel = new Map<string, ByokModelLimits>();
    for (const item of body.data) {
      const id = item && typeof item === 'object' ? (item as { id?: unknown }).id : undefined;
      if (typeof id !== 'string' || !id) continue;
      const limits = listedModelLimits(item);
      if (limits.output !== undefined || limits.context !== undefined) byModel.set(id, limits);
    }
    return byModel;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * The listed limits for `model` at the provider's base URL, or null when the
 * provider does not list them. Never throws; one list request per base URL is
 * cached for ten minutes.
 */
export async function resolveByokModelLimits(
  provider: Pick<ByokChatProviderConfig, 'protocol' | 'baseUrl'> & { apiKey?: string },
  model: string | null | undefined,
  options: ResolveOptions = {},
): Promise<ByokModelLimits | null> {
  const modelId = typeof model === 'string' ? model.trim() : '';
  const baseUrl = typeof provider?.baseUrl === 'string' ? provider.baseUrl.trim() : '';
  if (!modelId || !baseUrl || !PROTOCOLS_WITH_LISTED_LIMITS.has(provider.protocol)) return null;
  let url: string;
  try {
    url = modelsUrl(baseUrl);
  } catch {
    return null;
  }
  if (!HOSTS_WITH_LISTED_LIMITS.has(new URL(url).hostname.toLowerCase())) return null;
  const cached = cache.get(url);
  let byModel: Map<string, ByokModelLimits> | null;
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) {
    byModel = cached.byModel;
  } else {
    const validate = options.validateBaseUrl ?? validateUserProviderBaseUrl;
    const validated = await validate(baseUrl).catch(() => ({ error: 'invalid' }));
    if (validated.error || !('parsed' in validated) || !validated.parsed) return null;
    byModel = await fetchLimitsList(url, options.fetchImpl ?? fetch, options.timeoutMs ?? LIMITS_TIMEOUT_MS);
    cache.set(url, { at: Date.now(), byModel });
  }
  return byModel?.get(modelId) ?? null;
}
