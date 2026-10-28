// Optional fields some `/v1/models` lists carry beyond `id`: a display
// `name`, a `capabilities` array and a `context_window`. ai& (aiand.com)
// sends all three on its OpenAI-shape list. Providers that do not send them
// are unaffected: every reader here returns undefined for a missing field.

/** Capabilities that mean a model answers chat/completions or messages. */
const TEXT_GENERATION_CAPABILITIES = new Set(['chat', 'reasoning', 'tool_calling']);

/** Lower-cased `capabilities` strings, or undefined when the entry has no such array. */
export function listedCapabilities(item: unknown): string[] | undefined {
  if (!item || typeof item !== 'object') return undefined;
  const raw = (item as { capabilities?: unknown }).capabilities;
  if (!Array.isArray(raw)) return undefined;
  return raw
    .filter((c): c is string => typeof c === 'string')
    .map((c) => c.trim().toLowerCase())
    .filter(Boolean);
}

/**
 * Whether a listed model generates text, judged from its declared
 * capabilities. Undefined when the entry declares none, so the caller falls
 * back to its id heuristics. ai& does not list `chat` for every chat model,
 * so `reasoning` or `tool_calling` also count.
 */
export function declaresTextGeneration(capabilities: readonly string[] | undefined): boolean | undefined {
  if (!capabilities || capabilities.length === 0) return undefined;
  return capabilities.some((c) => TEXT_GENERATION_CAPABILITIES.has(c));
}

/** The entry's display `name`, when it is a non-empty string. */
export function listedDisplayName(item: unknown): string | undefined {
  if (!item || typeof item !== 'object') return undefined;
  const name = (item as { name?: unknown }).name;
  return typeof name === 'string' && name.trim() ? name.trim() : undefined;
}

/** A positive integer `context_window`, when present. */
export function listedContextWindow(item: unknown): number | undefined {
  if (!item || typeof item !== 'object') return undefined;
  const raw = (item as { context_window?: unknown }).context_window;
  const n = typeof raw === 'string' ? Number(raw) : raw;
  return typeof n === 'number' && Number.isInteger(n) && n > 0 ? n : undefined;
}
