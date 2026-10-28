import { describe, expect, it } from 'vitest';
import {
  FAST_MODEL_BY_PROTOCOL,
  SUGGESTED_MODELS_BY_PROTOCOL,
} from '../../src/state/apiProtocols';
import { KNOWN_PROVIDERS } from '../../src/state/config';

describe('apiProtocols table consistency', () => {
  it('FAST_MODEL_BY_PROTOCOL.google is one of the live suggested models', () => {
    expect(SUGGESTED_MODELS_BY_PROTOCOL.google).toContain(FAST_MODEL_BY_PROTOCOL.google);
  });

  it('keeps the Ollama Cloud picker current with recent cloud models', () => {
    const recentCloudModels = [
      'glm-5.2',
      'kimi-k2.7-code',
    ];
    const ollamaCloudProvider = KNOWN_PROVIDERS.find(
      (provider) => provider.protocol === 'ollama' && provider.baseUrl === 'https://ollama.com',
    );

    expect(ollamaCloudProvider?.preferredModels).toBeDefined();
    for (const model of recentCloudModels) {
      expect(SUGGESTED_MODELS_BY_PROTOCOL.ollama).toContain(model);
      expect(ollamaCloudProvider?.preferredModels).toContain(model);
    }
  });

  it('keeps the Atlas Cloud preset wired to OpenAI-compatible chat models', () => {
    const atlasCloudProvider = KNOWN_PROVIDERS.find(
      (provider) =>
        provider.protocol === 'openai' &&
        provider.baseUrl === 'https://api.atlascloud.ai/v1',
    );

    expect(atlasCloudProvider).toMatchObject({
      label: 'Atlas Cloud',
      apiKeyConsoleLink: {
        host: 'atlascloud.ai',
        url: 'https://atlascloud.ai/?utm_source=open_design&utm_medium=provider_preset&utm_campaign=atlascloud_byok',
      },
    });
    expect(atlasCloudProvider?.preferredModels).toContain('qwen/qwen3.5-flash');
    expect(atlasCloudProvider?.preferredModels).toContain('deepseek-ai/deepseek-v4-flash');
  });

  it('keeps the ai& presets on one catalogue across both API shapes', () => {
    const openai = KNOWN_PROVIDERS.find((provider) => provider.label === 'ai& — OpenAI');
    const anthropic = KNOWN_PROVIDERS.find((provider) => provider.label === 'ai& — Anthropic');
    const consoleLink = { host: 'console.aiand.com', url: 'https://console.aiand.com' };

    expect(openai).toMatchObject({
      protocol: 'openai',
      baseUrl: 'https://api.aiand.com/v1',
      apiKeyConsoleLink: consoleLink,
    });
    // Origin only: the Anthropic SDK, the daemon proxy and BYOK OpenCode
    // each add /v1 before /messages.
    expect(anthropic).toMatchObject({
      protocol: 'anthropic',
      baseUrl: 'https://api.aiand.com',
      apiKeyConsoleLink: consoleLink,
    });
    expect(anthropic?.preferredModels).toEqual(openai?.preferredModels);
    // The live GET /v1/models list is the catalogue; the presets keep one
    // fallback id for when that list cannot be fetched.
    expect(openai?.publicModelList).toBe(true);
    expect(anthropic?.publicModelList).toBe(true);
    expect(openai?.preferredModels).toEqual(['moonshotai/kimi-k3']);
  });
});
