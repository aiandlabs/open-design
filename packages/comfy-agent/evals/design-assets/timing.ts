// ComfyUI execution time per prompt, separate from time spent waiting in the
// queue. When other clients share the server, the wall time a row measures
// includes their jobs; the history timestamps do not.

import { readFile } from "node:fs/promises";

import type { ComfyClient } from "../../src/client.js";
import type { HistoryEntry } from "../../src/types.js";
import type { ResultRow } from "./results.js";

/** Milliseconds from `execution_start` to `execution_success` (or error/interrupt), or undefined if absent. */
export function executionMs(entry: HistoryEntry | null | undefined): number | undefined {
  const msgs = entry?.status?.messages ?? [];
  const ts = (name: string) => {
    const m = msgs.find(([n]) => n === name);
    const t = m?.[1]?.timestamp;
    return typeof t === "number" ? t : undefined;
  };
  const start = ts("execution_start");
  const end = ts("execution_success") ?? ts("execution_error") ?? ts("execution_interrupted");
  return start !== undefined && end !== undefined && end >= start ? end - start : undefined;
}

/** Prompt ids a row generated, from the row itself or from its trace file. */
export async function promptIdsOf(row: ResultRow): Promise<string[]> {
  if (row.promptIds?.length) return row.promptIds;
  if (!row.tracePath) return [];
  try {
    const trace = JSON.parse(await readFile(row.tracePath, "utf8")) as { promptId?: string; rounds?: { generation?: { promptId?: string } }[] };
    if (typeof trace.promptId === "string") return [trace.promptId];
    return (trace.rounds ?? []).map((r) => r.generation?.promptId).filter((p): p is string => typeof p === "string");
  } catch {
    return [];
  }
}

/**
 * Sum execution time over a row's prompts and derive queue wait. Returns
 * undefined when any prompt is missing from the server history.
 */
export async function comfyTiming(comfy: ComfyClient, row: ResultRow): Promise<Pick<ResultRow, "promptIds" | "comfyExecMs" | "comfyQueueMs"> | undefined> {
  const ids = await promptIdsOf(row);
  if (!ids.length) return undefined;
  let exec = 0;
  for (const id of ids) {
    const ms = executionMs(await comfy.getHistory(id).catch(() => null));
    if (ms === undefined) return undefined;
    exec += ms;
  }
  return { promptIds: ids, comfyExecMs: exec, comfyQueueMs: Math.max(0, row.comfyMs - exec) };
}

/** The checkpoint a row generated with, from the row or its trace (raw: `checkpoint`; plan/loop: `server.checkpoint`). */
export async function checkpointOf(row: ResultRow): Promise<string | undefined> {
  if (row.checkpoint) return row.checkpoint;
  if (!row.tracePath) return undefined;
  try {
    const trace = JSON.parse(await readFile(row.tracePath, "utf8")) as { checkpoint?: unknown; server?: { checkpoint?: unknown } };
    const c = trace.checkpoint ?? trace.server?.checkpoint;
    return typeof c === "string" && c ? c : undefined;
  } catch {
    return undefined;
  }
}

/** Planner and critic of a plan/loop row, from the row or its trace (`models`). Raw rows use neither. */
export async function llmModelsOf(row: ResultRow): Promise<ResultRow["models"]> {
  if (row.models) return row.models;
  if (row.arm === "raw" || !row.tracePath) return undefined;
  try {
    const trace = JSON.parse(await readFile(row.tracePath, "utf8")) as { models?: { planner?: unknown; critic?: unknown } };
    const { planner, critic } = trace.models ?? {};
    return typeof planner === "string" && typeof critic === "string" ? { planner, critic } : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Image models already in a run that differ from the one new rows would use.
 * A run that mixes image models compares models, not arms.
 */
export function foreignCheckpoints(existing: readonly (string | undefined)[], current: string): string[] {
  return [...new Set(existing.filter((c): c is string => Boolean(c) && c !== current))];
}
