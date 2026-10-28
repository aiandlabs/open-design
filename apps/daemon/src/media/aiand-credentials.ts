// ai& API credentials for daemon features other than the ComfyUI provider
// itself (today: visual check). The key resolves exactly as the comfyui
// media provider resolves it — `resolveProviderConfig('comfyui')` reads
// OD_COMFYUI_AIAND_API_KEY / AIAND_API_KEY and then the stored Settings →
// Media providers → ComfyUI entry, and `resolveComfyConfig` applies the
// comfy-agent defaults (base URL from AIAND_BASE_URL, API shape from the
// entry's model settings). One resolution, two consumers.

import type { ApiShape } from '@open-design/comfy-agent';

import { resolveProviderConfig } from './config.js';
import { resolveComfyConfig } from './comfyui.js';

export interface AiandLlmCredentials {
  apiKey: string | null;
  baseUrl: string;
  shape: ApiShape;
}

export async function resolveAiandLlmCredentials(
  projectRoot: string,
  env: NodeJS.ProcessEnv = process.env,
): Promise<AiandLlmCredentials> {
  const entry = await resolveProviderConfig(projectRoot, 'comfyui');
  const cfg = resolveComfyConfig(
    {
      ...(entry.apiKey ? { apiKey: entry.apiKey } : {}),
      ...(entry.model ? { model: entry.model } : {}),
    },
    env,
  );
  return {
    apiKey: cfg.aiandApiKey ?? null,
    baseUrl: cfg.aiandBaseUrl,
    shape: cfg.apiShape,
  };
}
