import { describe, expect, it } from 'vitest';

import {
  loadsModelsBeforeKey,
  orderVisionModelsFirst,
  providerListsModelsWithoutKey,
  suggestedIdsWithLiveList,
} from '../../src/components/byok/public-model-list';
import { KNOWN_PROVIDERS } from '../../src/state/config';

const aiand = KNOWN_PROVIDERS.find((p) => p.label === 'ai& — OpenAI');
const openai = KNOWN_PROVIDERS.find((p) => p.label === 'OpenAI');

describe('public model list rules', () => {
  it('only presets that declare a public list load before a key', () => {
    expect(providerListsModelsWithoutKey(aiand)).toBe(true);
    expect(providerListsModelsWithoutKey(openai)).toBe(false);
    expect(providerListsModelsWithoutKey(undefined)).toBe(false);
    expect(loadsModelsBeforeKey(aiand, '')).toBe(true);
    expect(loadsModelsBeforeKey(aiand, '  ')).toBe(true);
    // With a key, the normal commit-on-blur flow fetches with that key.
    expect(loadsModelsBeforeKey(aiand, 'sk-x')).toBe(false);
    expect(loadsModelsBeforeKey(openai, '')).toBe(false);
  });

  it('drops a public-list preset\'s fallback ids once the live list has loaded', () => {
    expect(suggestedIdsWithLiveList(aiand, ['moonshotai/kimi-k3'], 0)).toEqual(['moonshotai/kimi-k3']);
    expect(suggestedIdsWithLiveList(aiand, ['moonshotai/kimi-k3'], 4)).toEqual([]);
    expect(suggestedIdsWithLiveList(openai, ['gpt-4o'], 4)).toEqual(['gpt-4o']);
  });

  it('puts vision models first and keeps the order otherwise', () => {
    const models = [
      { id: 'a', metadata: { vision: false } },
      { id: 'b', metadata: { vision: true } },
      { id: 'c' },
      { id: 'd', metadata: { vision: true } },
    ];
    expect(orderVisionModelsFirst(models).map((m) => m.id)).toEqual(['b', 'd', 'a', 'c']);
    const unknown = [{ id: 'x' }, { id: 'y', metadata: { cost: 'low' as const } }];
    expect(orderVisionModelsFirst(unknown)).toEqual(unknown);
  });
});
