// One-shot workflow run: take a user graph (API or UI format) or the default
// base graph, fill the prompt / size / seed slots through typed edits, validate
// against the server, run it, and return the image. No LLM involved.

import { randomInt } from "node:crypto";

import type { ComfyClient } from "./client.js";
import { GraphEditError } from "./errors.js";
import { applyEdit, describeDoc, type GraphEdit } from "./edits.js";
import { buildTxt2Img, chooseModel, defaultSampling, detectFamily, docFromGraph, planGenerationSize, type GraphDoc } from "./graphs.js";
import { coverToSize, decodePng, encodePng } from "./image.js";
import type { ApiWorkflow, OutputImage, ProgressEvent } from "./types.js";
import { toApiWorkflow } from "./ui-convert.js";
import { assertValidGraph } from "./validate.js";

export type WorkflowRunInput = {
  /** API- or UI-format workflow JSON. Omit to use the default text-to-image base graph. */
  workflow?: unknown;
  prompt?: string;
  negative?: string;
  width?: number;
  height?: number;
  seed?: number;
  checkpoint?: string;
  /** Crop/resize the output to exactly width × height (default true when both are given). */
  exactSize?: boolean;
  timeoutMs?: number;
  onProgress?: (e: ProgressEvent) => void;
  signal?: AbortSignal;
};

export type WorkflowRunResult = {
  png: Buffer;
  images: OutputImage[];
  graph: ApiWorkflow;
  promptId: string;
  durationMs: number;
  settings: Record<string, unknown>;
  /** Slots that were requested but that this graph has no node for. */
  unfilled: string[];
};

export async function runWorkflow(comfy: ComfyClient, input: WorkflowRunInput): Promise<WorkflowRunResult> {
  const info = await comfy.getObjectInfo();
  const seed = input.seed ?? randomInt(0, 2 ** 32 - 1);
  let doc: GraphDoc;
  const unfilled: string[] = [];

  if (input.workflow !== undefined && input.workflow !== null) {
    doc = docFromGraph(toApiWorkflow(input.workflow, info));
    const tryEdit = (edit: GraphEdit, slot: string) => {
      try {
        doc = applyEdit(doc, edit, info);
      } catch (err) {
        if (err instanceof GraphEditError && /has no .* node/.test(err.message)) unfilled.push(slot);
        else throw err;
      }
    };
    if (input.prompt) tryEdit({ op: "set_prompt", text: input.prompt }, "prompt");
    if (input.negative) tryEdit({ op: "set_negative_prompt", text: input.negative }, "negative");
    if (input.checkpoint) tryEdit({ op: "set_checkpoint", ckpt_name: input.checkpoint }, "checkpoint");
    if (input.width && input.height) {
      const family = doc.family !== "unknown" ? doc.family : detectFamily(String(describeDoc(doc).checkpoint ?? ""));
      const gen = planGenerationSize(family, input.width, input.height);
      tryEdit({ op: "set_size", width: gen.width, height: gen.height }, "size");
    }
    tryEdit({ op: "set_seed", seed }, "seed");
  } else {
    if (!input.prompt) throw new GraphEditError("run_workflow", "a prompt is required when no workflow is supplied");
    const choice = chooseModel(info, input.checkpoint);
    const family = choice.family;
    const gen = planGenerationSize(family, input.width ?? 1024, input.height ?? 1024);
    doc = buildTxt2Img({
      checkpoint: choice.model,
      family,
      ...(choice.textEncoder ? { textEncoder: choice.textEncoder } : {}),
      ...(choice.vae ? { vae: choice.vae } : {}),
      positive: input.prompt,
      negative: input.negative ?? "blurry, low quality, watermark, text",
      width: gen.width,
      height: gen.height,
      filenamePrefix: "comfy-agent/workflow",
      ...defaultSampling(family, info),
      seed,
    });
  }

  assertValidGraph(doc.graph, info);
  const run = await comfy.run(doc.graph, {
    validate: false,
    ...(input.timeoutMs ? { timeoutMs: input.timeoutMs } : {}),
    ...(input.onProgress ? { onProgress: input.onProgress } : {}),
    ...(input.signal ? { signal: input.signal } : {}),
  });
  const last = run.images.at(-1)!;
  let png = last.bytes;
  const exact = input.exactSize ?? Boolean(input.width && input.height);
  if (exact && input.width && input.height) {
    const img = decodePng(png);
    if (img.width !== input.width || img.height !== input.height) {
      png = encodePng(coverToSize(img, input.width, input.height));
    }
  }
  return {
    png,
    images: run.images,
    graph: doc.graph,
    promptId: run.promptId,
    durationMs: run.durationMs,
    settings: describeDoc(doc),
    unfilled,
  };
}
