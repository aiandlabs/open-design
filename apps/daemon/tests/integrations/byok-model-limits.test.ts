import { afterEach, describe, expect, it } from 'vitest';

import {
  clearByokModelLimitsCache,
  listedModelLimits,
  resolveByokModelLimits,
} from '../../src/integrations/byok-model-limits.js';
import { buildOpenCodeByokProviderConfig } from '../../src/runtimes/byok-opencode.js';

// Shape of ai&'s public `/v1/models` entries (OpenAI shape), trimmed.
const AIAND_MODELS = {
  object: 'list',
  data: [
    {
      id: 'moonshotai/kimi-k3',
      object: 'model',
      context_window: 1_048_576,
      max_output_tokens: 131_072,
      capabilities: ['vision', 'reasoning', 'tool_calling'],
    },
    { id: 'qwen/qwen3.8-27b', object: 'model', context_window: 262_144, max_output_tokens: 32_768 },
    { id: 'no-limits/model', object: 'model' },
  ],
};

const allowAll = async () => ({ parsed: new URL('https://api.aiand.com/v1') });

function fakeFetch(body: unknown, status = 200) {
  const calls: Array<{ url: string; init: RequestInit | undefined }> = [];
  const fetchImpl = (async (url: string | URL, init?: RequestInit) => {
    calls.push({ url: String(url), init });
    return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
  }) as typeof fetch;
  return { calls, fetchImpl };
}

afterEach(() => clearByokModelLimitsCache());

describe('listedModelLimits', () => {
  it('reads max_output_tokens and context_window from a list entry', () => {
    expect(listedModelLimits(AIAND_MODELS.data[0])).toEqual({ output: 131_072, context: 1_048_576 });
    expect(listedModelLimits({ id: 'x' })).toEqual({});
    expect(listedModelLimits({ id: 'x', max_output_tokens: -1, context_window: 'big' })).toEqual({});
  });
});

describe('resolveByokModelLimits', () => {
  it('returns the listed limits for the run model, without sending the key', async () => {
    const { calls, fetchImpl } = fakeFetch(AIAND_MODELS);
    const limits = await resolveByokModelLimits(
      { protocol: 'openai', baseUrl: 'https://api.aiand.com/v1', apiKey: 'sk-secret' },
      'moonshotai/kimi-k3',
      { fetchImpl, validateBaseUrl: allowAll },
    );
    expect(limits).toEqual({ output: 131_072, context: 1_048_576 });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe('https://api.aiand.com/v1/models');
    expect(JSON.stringify(calls[0]!.init?.headers ?? {})).not.toContain('sk-secret');
  });

  it('uses the same list for the Anthropic shape (ai& serves both from one host)', async () => {
    const { calls, fetchImpl } = fakeFetch(AIAND_MODELS);
    const limits = await resolveByokModelLimits(
      { protocol: 'anthropic', baseUrl: 'https://api.aiand.com', apiKey: 'sk-secret' },
      'qwen/qwen3.8-27b',
      { fetchImpl, validateBaseUrl: allowAll },
    );
    expect(limits).toEqual({ output: 32_768, context: 262_144 });
    expect(calls[0]!.url).toBe('https://api.aiand.com/v1/models');
  });

  it('caches per base URL, so one list request serves several runs', async () => {
    const { calls, fetchImpl } = fakeFetch(AIAND_MODELS);
    const provider = { protocol: 'openai' as const, baseUrl: 'https://api.aiand.com/v1', apiKey: '' };
    await resolveByokModelLimits(provider, 'moonshotai/kimi-k3', { fetchImpl, validateBaseUrl: allowAll });
    await resolveByokModelLimits(provider, 'qwen/qwen3.8-27b', { fetchImpl, validateBaseUrl: allowAll });
    expect(calls).toHaveLength(1);
  });

  it('returns null when the model is not listed, has no limits, or the list fails', async () => {
    const ok = fakeFetch(AIAND_MODELS);
    const provider = { protocol: 'openai' as const, baseUrl: 'https://api.aiand.com/v1', apiKey: '' };
    expect(await resolveByokModelLimits(provider, 'missing/model', { fetchImpl: ok.fetchImpl, validateBaseUrl: allowAll })).toBeNull();
    expect(await resolveByokModelLimits(provider, 'no-limits/model', { fetchImpl: ok.fetchImpl, validateBaseUrl: allowAll })).toBeNull();
    clearByokModelLimitsCache();
    const denied = fakeFetch({ error: 'unauthorized' }, 401);
    expect(await resolveByokModelLimits(provider, 'moonshotai/kimi-k3', { fetchImpl: denied.fetchImpl, validateBaseUrl: allowAll })).toBeNull();
    clearByokModelLimitsCache();
    const thrower = (async () => { throw new Error('offline'); }) as typeof fetch;
    expect(await resolveByokModelLimits(provider, 'moonshotai/kimi-k3', { fetchImpl: thrower, validateBaseUrl: allowAll })).toBeNull();
  });

  it('asks only hosts that publish limits, skips other protocols, and honours the SSRF guard', async () => {
    const { calls, fetchImpl } = fakeFetch(AIAND_MODELS);
    expect(await resolveByokModelLimits(
      { protocol: 'google', baseUrl: 'https://generativelanguage.googleapis.com', apiKey: 'k' },
      'moonshotai/kimi-k3',
      { fetchImpl, validateBaseUrl: allowAll },
    )).toBeNull();
    expect(await resolveByokModelLimits(
      { protocol: 'openai', baseUrl: 'https://api.openai.com/v1', apiKey: 'k' },
      'gpt-5',
      { fetchImpl, validateBaseUrl: allowAll },
    )).toBeNull();
    expect(await resolveByokModelLimits(
      { protocol: 'openai', baseUrl: 'https://api.aiand.com/v1', apiKey: '' },
      'moonshotai/kimi-k3',
      { fetchImpl, validateBaseUrl: async () => ({ error: 'forbidden', forbidden: true }) },
    )).toBeNull();
    expect(calls).toHaveLength(0);
  });
});

describe('buildOpenCodeByokProviderConfig limits', () => {
  const provider = { protocol: 'openai' as const, apiKey: 'sk', baseUrl: 'https://api.aiand.com/v1' };
  const modelLimit = (out: ReturnType<typeof buildOpenCodeByokProviderConfig>, model: string) =>
    (out?.config as { provider: Record<string, { models: Record<string, { limit: unknown }> }> })
      .provider['open-design-byok']!.models[model]!.limit;

  it('uses the listed output and context limits when given', () => {
    const out = buildOpenCodeByokProviderConfig(provider, 'moonshotai/kimi-k3', { output: 131_072, context: 1_048_576 });
    expect(modelLimit(out, 'moonshotai/kimi-k3')).toEqual({ context: 1_048_576, output: 131_072 });
  });

  it('keeps the defaults for a limit the list does not give', () => {
    const out = buildOpenCodeByokProviderConfig(provider, 'moonshotai/kimi-k3', { output: 32_768 });
    expect(modelLimit(out, 'moonshotai/kimi-k3')).toEqual({ context: 128_000, output: 32_768 });
    expect(modelLimit(buildOpenCodeByokProviderConfig(provider, 'x'), 'x')).toEqual({ context: 128_000, output: 16_384 });
  });

  it('never sets an output limit at or above the context limit', () => {
    const out = buildOpenCodeByokProviderConfig(provider, 'm', { output: 262_144, context: 262_144 });
    const limit = modelLimit(out, 'm') as { context: number; output: number };
    expect(limit.output).toBeLessThan(limit.context);
  });
});
