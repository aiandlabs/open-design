import { SETTINGS_MEDIA_PROVIDERS_PATH } from '../settings-nav.js';

/**
 * Which local image route a run has, as the host resolved it:
 *
 * - `comfyui-agent` — ComfyUI is configured and the ai& API key for the
 *   planner and vision model resolves.
 * - `comfyui-workflow` — ComfyUI is configured but the ai& key is missing, so
 *   only the single-pass workflow model can run.
 * - `null` — ComfyUI is not configured.
 */
export type LocalComfyImageModel = 'comfyui-agent' | 'comfyui-workflow';

export interface LocalImageGenerationPromptInput {
  /** Settings → Media providers → "Local image generation only". */
  localOnly: boolean;
  comfyuiModel: LocalComfyImageModel | null;
}

const OD = '"$OD_NODE_BIN" "$OD_BIN"';

/**
 * The one source of the image-generation routing rule for the local ComfyUI
 * provider and the local-only policy.
 *
 * Legacy (`apps/daemon/src/prompts/media-contract.ts` and the non-media
 * dispatch section in `system.ts`) and OD Next (the `local-image-generation`
 * instruction block in `od-next-strategy.ts`) both render this text. It
 * returns '' when the policy is off and ComfyUI is not configured, so those
 * runs keep their prompts byte-identical to the build without the feature.
 *
 * The wording stays clear of the vocabulary OD Next's planning/Build guard
 * rejects (`FORBIDDEN_POST_BUILD_SEMANTICS` in `od-next-strategy.ts`);
 * `packages/contracts/tests/local-image-generation-prompt.test.ts` holds it
 * to that.
 */
export function renderLocalImageGenerationRule(
  input: LocalImageGenerationPromptInput | null | undefined,
): string {
  if (!input) return '';
  const { localOnly, comfyuiModel } = input;
  if (!localOnly && !comfyuiModel) return '';

  if (!comfyuiModel) {
    return [
      'Image generation: not available in this run.',
      'The user turned on "Local image generation only" in Settings, and no local ComfyUI server is configured. The daemon refuses every hosted image and video model (`vela/*`, OpenAI, Nano Banana, fal, OpenRouter, and the rest of the media catalogue), so do not run `media generate` for images or generative video (even when the project metadata or run defaults name a model), do not try other model ids, and do not ask the user for provider API keys.',
      `For each image the task needs, use a stock photo from \`${OD} media stock-search\`, an asset the user supplied, or a designed placeholder (CSS or SVG shapes, gradients, type) that keeps the planned size and position.`,
      `In the reply, say in one sentence that generated images were not available, so the design uses stock photos or placeholders, and that ComfyUI can be set up under ${SETTINGS_MEDIA_PROVIDERS_PATH}.`,
    ].join('\n');
  }

  const lines: string[] = [];
  lines.push(
    `Image generation: the user has a local ComfyUI server configured. Make every raster image this task needs (hero, illustration, icon, texture, product shot) with \`--model ${comfyuiModel}\`:`,
  );
  if (comfyuiModel === 'comfyui-agent') {
    lines.push(
      '```bash',
      `${OD} media generate --project "$OD_PROJECT_ID" --surface image --model comfyui-agent \\`,
      '  --output <planned-file>.png --prompt "<plain description of the asset>" \\',
      '  [--size <W>x<H>] [--transparent] [--negative "<things that must not appear>"] [--max-rounds <1-3>]',
      '```',
      '- Put a plain description of the asset in `--prompt`; comfyui-agent writes the Stable Diffusion prompt itself.',
      '- `--size <W>x<H>` sets an exact pixel size (for example `--size 1600x900` for a hero). Without it, `--aspect 1:1|16:9|9:16|4:3|3:4` picks the size.',
      '- `--transparent` makes a cut-out on a transparent background (icons, stickers, product cut-outs).',
      '- `--negative "<text>"` names things that must not appear.',
      '- comfyui-agent compares each result with the request and edits the graph to fix what is wrong, for up to `--max-rounds` rounds (default 3; use 1 or 2 for a quick draft).',
      '- Use `--model comfyui-workflow --workflow <file.json>` only when the user gives a ComfyUI workflow, or for one plain generation with no repair rounds.',
    );
  } else {
    lines.push(
      '```bash',
      `${OD} media generate --project "$OD_PROJECT_ID" --surface image --model comfyui-workflow \\`,
      '  --output <planned-file>.png --prompt "<Stable Diffusion prompt>" \\',
      '  [--size <W>x<H>] [--negative "<things that must not appear>"] [--workflow <file.json>]',
      '```',
      '- comfyui-workflow runs one generation. Write `--prompt` as a Stable Diffusion prompt: subject first, then style, lighting and composition.',
      '- `--size <W>x<H>` sets an exact pixel size; without it, `--aspect 1:1|16:9|9:16|4:3|3:4` picks the size. `--negative "<text>"` names things that must not appear.',
      `- comfyui-agent (repair rounds, \`--transparent\`) needs the ai& API key under ${SETTINGS_MEDIA_PROVIDERS_PATH} → ComfyUI, which is not set; do not use it in this run.`,
    );
  }
  lines.push(
    `- The server is shared and jobs queue, so a round can take a minute or more. When the command prints a \`taskId\`, run \`${OD} media wait <taskId> --since <n>\` until it exits 0 (done) or 5 (failed); exit 2 means it is still running.`,
    '- The command also writes `<name>.trace.json` beside the image. Leave it in the project; do not read it back or describe it.',
  );
  if (localOnly) {
    lines.push(
      `- "Local image generation only" is on in Settings: the daemon refuses every hosted image and video model (\`vela/*\`, OpenAI, Nano Banana, fal, OpenRouter, and the rest of the media catalogue). ${comfyuiModel} is the only image route, even when the project metadata or run defaults name a hosted model, so do not try other image models and do not ask for provider API keys. For video, only \`hyperframes-html\` (HTML motion rendered locally) can run.`,
      `- If a ComfyUI job does not succeed, act on its \`error.nextStep\`; when no image can be made, use a stock photo from \`${OD} media stock-search\` or a designed placeholder instead.`,
    );
  }
  return lines.join('\n');
}

/** Legacy-prompt section form: a horizontal rule and a heading around the rule. */
export function renderLocalImageGenerationSection(
  input: LocalImageGenerationPromptInput | null | undefined,
): string {
  const rule = renderLocalImageGenerationRule(input);
  if (!rule) return '';
  const heading = input?.comfyuiModel
    ? '## Image generation (local ComfyUI)'
    : '## Image generation (not available)';
  return `\n\n---\n\n${heading}\n\n${rule}\n`;
}
