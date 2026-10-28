// Picker rules for providers whose model list is public (`publicModelList`
// on KnownProvider, today ai&): the list loads before a key is entered, the
// live list replaces the preset's fallback suggestions once it arrives, and
// models the list marks as vision-capable come first.

import type { KnownProvider } from '../../state/config';
import type { ProviderModelOption } from '../../types';

export function providerListsModelsWithoutKey(
  provider: Pick<KnownProvider, 'publicModelList'> | null | undefined,
): boolean {
  return provider?.publicModelList === true;
}

/**
 * Whether the picker should load the list as soon as the preset is selected,
 * without waiting for the key field to be committed. Only while the key is
 * blank: once the user types a key, the normal commit-on-blur flow fetches
 * the list again with that key.
 */
export function loadsModelsBeforeKey(
  provider: Pick<KnownProvider, 'publicModelList'> | null | undefined,
  apiKey: string,
): boolean {
  return providerListsModelsWithoutKey(provider) && !apiKey.trim();
}

/**
 * Suggestions to merge under the live list. A public-list provider's
 * preferred ids are a fallback for when the list is missing, so they are
 * dropped once the live list has loaded; an id the live list lacks is never
 * offered.
 */
export function suggestedIdsWithLiveList(
  provider: Pick<KnownProvider, 'publicModelList'> | null | undefined,
  suggestedIds: readonly string[],
  liveModelCount: number,
): string[] {
  if (providerListsModelsWithoutKey(provider) && liveModelCount > 0) return [];
  return [...suggestedIds];
}

/**
 * Vision-capable models first, keeping the list's order inside each group.
 * Models whose `vision` is unknown sort with the non-vision ones, so a list
 * without capability data keeps its order.
 */
export function orderVisionModelsFirst<T extends Pick<ProviderModelOption, 'metadata'>>(
  models: readonly T[],
): T[] {
  if (!models.some((model) => model.metadata?.vision === true)) return [...models];
  return [
    ...models.filter((model) => model.metadata?.vision === true),
    ...models.filter((model) => model.metadata?.vision !== true),
  ];
}
