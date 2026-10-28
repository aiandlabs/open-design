// Model lists that carry `name`, `capabilities` and `context_window`
// (ai&'s does, on both API shapes' parsers): labels, the vision flag, and
// the chat filter that must trust declared capabilities over id heuristics.

import { promises as dnsPromises } from 'node:dns';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { listProviderModels } from '../src/integrations/provider-models.js';

let seen: { url: string; headers: Record<string, string> }[] = [];

function serve(body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: unknown, init?: RequestInit) => {
      seen.push({ url: String(input), headers: (init?.headers ?? {}) as Record<string, string> });
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    }),
  );
}

beforeEach(() => {
  seen = [];
  vi.spyOn(dnsPromises, 'lookup').mockImplementation((async () => [
    { address: '203.0.113.10', family: 4 },
  ]) as unknown as typeof dnsPromises.lookup);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// Shaped like the live https://api.aiand.com/v1/models response.
const AIAND_OPENAI_LIST = {
  object: 'list',
  data: [
    { id: 'deepseek-ai/deepseek-v4-pro', name: 'deepseek-ai/DeepSeek-V4-Pro', capabilities: ['reasoning', 'tool_calling'], context_window: 1048576 },
    { id: 'moonshotai/kimi-k3', name: 'moonshotai/kimi-k3', capabilities: ['reasoning', 'tool_calling', 'vision', 'document'], context_window: 262144 },
    // Ids that the old id heuristics would drop ("video", "image", "tts").
    { id: 'lab/omni-video-chat', name: 'Omni Video Chat', capabilities: ['chat', 'vision', 'video'] },
    { id: 'lab/image-reasoner', name: 'Image Reasoner', capabilities: ['reasoning', 'vision'] },
    // A list entry that declares no text capability is not a chat model.
    { id: 'lab/painter', name: 'Painter', capabilities: ['image_generation'] },
    // No capabilities: fall back to the id heuristics.
    { id: 'lab/plain-chat' },
    { id: 'lab/text-embedding-small' },
  ],
};

describe('listProviderModels with listed fields', () => {
  it('uses the listed name as the label and carries vision and context window', async () => {
    serve(AIAND_OPENAI_LIST);
    const res = await listProviderModels({ protocol: 'openai', baseUrl: 'https://api.aiand.com/v1', apiKey: 'k-test' });
    expect(res.ok).toBe(true);
    expect(seen[0]).toEqual({ url: 'https://api.aiand.com/v1/models', headers: { authorization: 'Bearer k-test' } });
    expect(res.models).toEqual([
      { id: 'deepseek-ai/deepseek-v4-pro', label: 'deepseek-ai/DeepSeek-V4-Pro', metadata: { contextWindowTokens: 1048576, vision: false } },
      { id: 'moonshotai/kimi-k3', label: 'moonshotai/kimi-k3', metadata: { contextWindowTokens: 262144, vision: true } },
      { id: 'lab/omni-video-chat', label: 'Omni Video Chat', metadata: { vision: true } },
      { id: 'lab/image-reasoner', label: 'Image Reasoner', metadata: { vision: true } },
      { id: 'lab/plain-chat', label: 'lab/plain-chat' },
    ]);
  });

  it('sends no credentials when the key is blank', async () => {
    serve(AIAND_OPENAI_LIST);
    const res = await listProviderModels({ protocol: 'openai', baseUrl: 'https://api.aiand.com/v1', apiKey: '  ' });
    expect(res.ok).toBe(true);
    expect(seen[0]!.headers).toEqual({});
  });

  it('parses the same fields on an Anthropic-shape list, and sends no key when blank', async () => {
    // ai&'s Anthropic-shape list carries display_name only today; capabilities
    // are read when present so both shapes behave alike.
    serve({
      data: [
        { id: 'a/vision', type: 'model', display_name: 'A Vision', capabilities: ['vision', 'tool_calling'] },
        { id: 'a/text', type: 'model', display_name: 'A Text' },
        { id: 'a/painter', type: 'model', display_name: 'Painter', capabilities: ['image_generation'] },
      ],
    });
    const res = await listProviderModels({ protocol: 'anthropic', baseUrl: 'https://api.aiand.com', apiKey: '' });
    expect(seen[0]!.url).toBe('https://api.aiand.com/v1/models?limit=1000');
    expect(seen[0]!.headers).toEqual({ 'anthropic-version': '2023-06-01' });
    expect(res.models).toEqual([
      { id: 'a/vision', label: 'A Vision', metadata: { vision: true } },
      { id: 'a/text', label: 'A Text' },
    ]);
  });
});
