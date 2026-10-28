import { describe, expect, it } from 'vitest';

import type { LocalImageGenerationPromptInput, OdNextStrategyRequestRecipeV2 } from '@open-design/contracts';

import {
  CONTRACT_IMAGE_DEFAULT_BULLETS,
  CONTRACT_IMAGE_IDS_LINE,
  CONTRACT_VIDEO_IDS_LINE,
} from '../../src/prompts/local-image-generation.js';
import { MEDIA_GENERATION_CONTRACT, renderMediaGenerationContract } from '../../src/prompts/media-contract.js';
import { composeSystemPrompt, type ComposeInput } from '../../src/prompts/system.js';

const recipe: OdNextStrategyRequestRecipeV2 = {
  recipe: 'od-next-plan-build-v2',
  strategyId: 'od-next-strategy',
  strategyVersion: '2.0.0',
  snapshotId: 'snapshot-local-image',
  packageHash: 'a'.repeat(64),
  taskProfileDigest: 'b'.repeat(64),
  taskProfileVersion: '2.0.0',
  taskType: 'prototype',
  executionProfile: 'filesystem',
  coreStrategy: '# Core\n\nKeep route and execution facts locked.',
  generalOrchestration: '# Orchestration\n\nPrepare a Design Spec and Full Plan, then Build.',
  taskSkill: '# Prototype\n\nProduce the declared editable prototype.',
  activeStages: [
    { name: 'discovery', atoms: [{ name: 'discovery-question-form' }] },
    { name: 'plan', atoms: [{ name: 'direction-picker' }, { name: 'todo-write' }] },
    { name: 'generate', atoms: [{ name: 'file-write' }, { name: 'live-artifact' }] },
  ],
};

const OFF: LocalImageGenerationPromptInput = { localOnly: false, comfyuiModel: null };
const ON_NO_COMFY: LocalImageGenerationPromptInput = { localOnly: true, comfyuiModel: null };
const ON_COMFY: LocalImageGenerationPromptInput = { localOnly: true, comfyuiModel: 'comfyui-agent' };
const OFF_COMFY: LocalImageGenerationPromptInput = { localOnly: false, comfyuiModel: 'comfyui-agent' };

const SCENARIOS: ReadonlyArray<readonly [string, ComposeInput]> = [
  ['classic design', { skillMode: 'prototype', metadata: { kind: 'prototype' } as any }],
  ['slim design', { skillMode: 'prototype', promptCoreVariant: 'slim', metadata: { kind: 'prototype' } as any }],
  ['slim deck', { skillMode: 'deck', promptCoreVariant: 'slim', metadata: { kind: 'deck' } as any }],
  ['classic design with media hint', { skillMode: 'prototype', mediaHintSignal: true, metadata: { kind: 'prototype' } as any }],
  ['classic plain stream', { streamFormat: 'plain', skillMode: 'prototype' }],
  ['slim ask mode', { promptCoreVariant: 'slim', sessionMode: 'chat' as any }],
  ['slim image', { promptCoreVariant: 'slim', skillMode: 'image', metadata: { kind: 'image' } as any }],
  ['classic video', { skillMode: 'video', metadata: { kind: 'video' } as any }],
];

describe('policy off and ComfyUI not configured: every prompt is byte-identical', () => {
  it.each(SCENARIOS)('%s', (_label, input) => {
    const baseline = composeSystemPrompt(input);
    expect(composeSystemPrompt({ ...input, localImageGeneration: OFF })).toBe(baseline);
    expect(composeSystemPrompt({ ...input, localImageGeneration: undefined })).toBe(baseline);
    expect(baseline).not.toContain('Image generation (local ComfyUI)');
    expect(baseline).not.toContain('Local image generation only');
  });

  it('the OD Next fork', () => {
    expect(composeSystemPrompt({ odNextStrategyRecipe: recipe, localImageGeneration: OFF })).toBe(
      composeSystemPrompt({ odNextStrategyRecipe: recipe }),
    );
  });

  it('the media contract', () => {
    expect(renderMediaGenerationContract(undefined, undefined, { localImageGeneration: OFF })).toBe(
      renderMediaGenerationContract(undefined, undefined),
    );
    expect(renderMediaGenerationContract(undefined, undefined)).toBe(MEDIA_GENERATION_CONTRACT);
  });
});

describe('legacy media contract (image / video projects)', () => {
  it('still carries the exact catalogue lines the local-only rewrite targets', () => {
    expect(MEDIA_GENERATION_CONTRACT).toContain(CONTRACT_IMAGE_IDS_LINE);
    expect(MEDIA_GENERATION_CONTRACT).toContain(CONTRACT_VIDEO_IDS_LINE);
    expect(MEDIA_GENERATION_CONTRACT).toContain(CONTRACT_IMAGE_DEFAULT_BULLETS);
  });

  it('local-only with ComfyUI: names comfyui-agent and its flags, narrows the lists and the image default', () => {
    const prompt = composeSystemPrompt({ ...SCENARIOS[6]![1], localImageGeneration: ON_COMFY });
    expect(prompt).toContain('### Image generation (local ComfyUI)');
    expect(prompt).toContain('--model comfyui-agent');
    expect(prompt).toContain('--size <W>x<H>');
    expect(prompt).toContain('--transparent');
    expect(prompt).toContain('--negative');
    expect(prompt).toContain('"Local image generation only" is on');
    expect(prompt).not.toContain(CONTRACT_IMAGE_IDS_LINE);
    expect(prompt).not.toContain(CONTRACT_VIDEO_IDS_LINE);
    expect(prompt).not.toContain(CONTRACT_IMAGE_DEFAULT_BULLETS);
    expect(prompt).toContain('- **image**:   `comfyui-agent`, `comfyui-workflow`');
    expect(prompt).toContain('- **video**:   `hyperframes-html`');
    expect(prompt).not.toContain('otherwise use `vela/gpt-image-2`');
  });

  it('local-only without ComfyUI: says there is no image generation', () => {
    const prompt = composeSystemPrompt({ ...SCENARIOS[6]![1], localImageGeneration: ON_NO_COMFY });
    expect(prompt).toContain('### Image generation (not available)');
    expect(prompt).toContain('Image generation: not available in this run.');
    expect(prompt).toContain('- **image**:   none');
    expect(prompt).not.toContain('--model comfyui-agent');
  });

  it('ComfyUI configured with the policy off: adds the section, keeps the full catalogue', () => {
    const prompt = composeSystemPrompt({ ...SCENARIOS[6]![1], localImageGeneration: OFF_COMFY });
    expect(prompt).toContain('--model comfyui-agent');
    expect(prompt).toContain(CONTRACT_IMAGE_IDS_LINE);
    expect(prompt).not.toContain('"Local image generation only" is on');
  });
});

describe('legacy non-media runs (prototype, deck)', () => {
  it.each([0, 1, 2])('%#: carries the section without any media vocabulary in the request', (index) => {
    const input = SCENARIOS[index]!;
    const withComfy = composeSystemPrompt({ ...input[1], localImageGeneration: ON_COMFY });
    expect(withComfy).toContain('## Image generation (local ComfyUI)');
    expect(withComfy).toContain('media generate --project "$OD_PROJECT_ID" --surface image --model comfyui-agent');
    const without = composeSystemPrompt({ ...input[1], localImageGeneration: ON_NO_COMFY });
    expect(without).toContain('## Image generation (not available)');
    expect(without).toContain('media stock-search');
    expect(without).toContain('designed placeholder');
  });

  it('the dispatch hint example uses the local model and defers model choice to the section', () => {
    const prompt = composeSystemPrompt({ ...SCENARIOS[3]![1], localImageGeneration: ON_COMFY });
    expect(prompt).toContain('IMAGE_MODEL="comfyui-agent"');
    expect(prompt).not.toContain('IMAGE_MODEL="flux-pro-ultra"');
    expect(prompt).not.toContain('For the best fal image model use `--model flux-pro-ultra`.');
    expect(prompt).toContain('the "Image generation" section of this prompt decides which model may run');
  });

  it('ask mode and text-artifact runs get nothing (no shell to run od)', () => {
    for (const index of [4, 5]) {
      const input = SCENARIOS[index]![1];
      expect(composeSystemPrompt({ ...input, localImageGeneration: ON_COMFY })).not.toContain('## Image generation');
    }
  });
});

describe('OD Next fork', () => {
  const BLOCK = '<od-next-context kind="instruction" name="local-image-generation">';

  it('carries the same rule as an instruction block', () => {
    const withComfy = composeSystemPrompt({ odNextStrategyRecipe: recipe, localImageGeneration: ON_COMFY });
    expect(withComfy).toContain(BLOCK);
    expect(withComfy).toContain('--model comfyui-agent');
    const without = composeSystemPrompt({ odNextStrategyRecipe: recipe, localImageGeneration: ON_NO_COMFY });
    expect(without).toContain(BLOCK);
    expect(without).toContain('Image generation: not available in this run.');
  });
});
