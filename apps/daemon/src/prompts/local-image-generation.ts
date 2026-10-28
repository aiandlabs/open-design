/**
 * Legacy-prompt wiring for the local image generation rule.
 *
 * The rule text has one source, `renderLocalImageGenerationRule` in
 * `@open-design/contracts` (`prompts/local-image-generation.ts`), which OD Next
 * also carries as its `local-image-generation` instruction block. This file
 * only places that text in the legacy media contract and narrows the
 * contract's catalogue lists when "Local image generation only" is on.
 *
 * Nothing here changes a prompt when the policy is off and ComfyUI is not
 * configured: every function returns its input unchanged (or '') then.
 */
import {
  isLocalMediaProvider,
  renderLocalImageGenerationRule,
  type LocalImageGenerationPromptInput,
} from '@open-design/contracts';

import { IMAGE_MODELS, VIDEO_MODELS } from '../media/models.js';

function fmtList(ids: string[]): string {
  return ids.map((id) => `\`${id}\``).join(', ');
}

const ALLOWED_IDS_MARKER = '\n### Allowed model IDs (per surface)';

// The catalogue lines and default-model bullets of MEDIA_GENERATION_CONTRACT
// that name hosted models. Under local-only they are swapped for the local
// route. Kept as exact strings so a wording change in the contract fails
// `apps/daemon/tests/prompts/local-image-generation.test.ts` instead of
// silently leaving a hosted default in place.
export const CONTRACT_IMAGE_IDS_LINE = `- **image**:   ${fmtList(IMAGE_MODELS.map((m) => m.id))}`;
export const CONTRACT_VIDEO_IDS_LINE = `- **video**:   ${fmtList(VIDEO_MODELS.map((m) => m.id))}`;
export const CONTRACT_IMAGE_DEFAULT_BULLETS = `   - **Image, best quality (user says "best", "highest quality", "most realistic")**:
     use \`flux-pro-ultra\` — but tell the user it takes 60–180s
   - **Image, default / no preference stated**: use an explicitly named model
     in the current user message, then the run-scoped BYOK image default, then the project metadata's
     \`imageModel\` if set; otherwise use \`vela/gpt-image-2\``;

/** True when the rule adds text to this run's prompt. */
export function hasLocalImageGenerationRule(
  input: LocalImageGenerationPromptInput | null | undefined,
): boolean {
  return renderLocalImageGenerationRule(input) !== '';
}

/**
 * The media-surface contract with the local image section inserted before the
 * catalogue lists, and, under local-only, the lists and image defaults
 * narrowed to the local route.
 */
export function applyLocalImageGenerationToMediaContract(
  contract: string,
  input: LocalImageGenerationPromptInput | null | undefined,
): string {
  const rule = renderLocalImageGenerationRule(input);
  if (!rule || !input) return contract;
  const heading = input.comfyuiModel
    ? '### Image generation (local ComfyUI)'
    : '### Image generation (not available)';
  const section = `\n${heading}\n\n${rule}\n`;
  let next = contract.includes(ALLOWED_IDS_MARKER)
    ? contract.replace(ALLOWED_IDS_MARKER, `${section}${ALLOWED_IDS_MARKER}`)
    : `${contract}\n${section}`;
  if (!input.localOnly) return next;

  const localImageIds = IMAGE_MODELS
    .filter((m) => isLocalMediaProvider(m.provider))
    .filter((m) => input.comfyuiModel === 'comfyui-agent' || m.id !== 'comfyui-agent')
    .map((m) => m.id);
  const localVideoIds = VIDEO_MODELS.filter((m) => isLocalMediaProvider(m.provider)).map((m) => m.id);
  next = next.replace(
    CONTRACT_IMAGE_IDS_LINE,
    input.comfyuiModel
      ? `- **image**:   ${fmtList(localImageIds)} (Local image generation only is on; hosted image models are refused)`
      : '- **image**:   none (Local image generation only is on and ComfyUI is not configured)',
  );
  next = next.replace(
    CONTRACT_VIDEO_IDS_LINE,
    `- **video**:   ${fmtList(localVideoIds)} (Local image generation only is on; hosted video models are refused)`,
  );
  next = next.replace(
    CONTRACT_IMAGE_DEFAULT_BULLETS,
    input.comfyuiModel
      ? `   - **Image**: use \`${input.comfyuiModel}\` as the Image generation section above says. A hosted model named in the project metadata or run defaults cannot run while Local image generation only is on.`
      : '   - **Image**: none can run; follow the Image generation section above.',
  );
  return next;
}

/** The non-media legacy section (prototype, deck, and other HTML runs). */
export { renderLocalImageGenerationSection } from '@open-design/contracts';
