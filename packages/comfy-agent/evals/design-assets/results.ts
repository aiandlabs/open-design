// One row per task × arm × seed in results.jsonl, plus resumability.

import { appendFile, readFile } from "node:fs/promises";

import type { CheckResult } from "../../src/loop.js";

/**
 * raw: the request as the prompt, one image. plan: the planner's prompt and
 * references, one image. loop: plan, critic and repair rounds. The ablation
 * arms are the loop with one feature off: chroma keying instead of matting,
 * no shape references, no set (consistency) references.
 */
export const ARMS = ["raw", "plan", "loop", "loop-chroma", "loop-noref", "loop-noset"] as const;
export type Arm = (typeof ARMS)[number];

export const ARM_LABEL: Record<Arm, string> = {
  raw: "A raw",
  plan: "B plan",
  loop: "C loop",
  "loop-chroma": "C− chroma key",
  "loop-noref": "C− no shape refs",
  "loop-noset": "C− no set refs",
};

/** Arms a run uses when --arms is not given. The ablations are opt-in. */
export const DEFAULT_ARMS: readonly Arm[] = ["raw", "plan", "loop"];

/** Whether an arm measures anything on a task (an ablation only runs where its feature is used). */
export function armApplies(arm: Arm, task: { transparent: boolean; group: string; set?: string }): boolean {
  if (arm === "loop-chroma") return task.transparent;
  if (arm === "loop-noref") return task.group === "icon";
  if (arm === "loop-noset") return Boolean(task.set);
  return true;
}

export type JudgedItem = { id: string; text: string; pass: boolean; evidence?: string };

export type JudgeResult = {
  model: string;
  shape: string;
  items: JudgedItem[];
  /** Set when the judge could not grade the image; the row is then not scored. */
  error?: string;
  /** Items the model did not return a usable grade for (graded as fail). */
  ungraded?: string[];
  retries: number;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
  latencyMs: number;
};

export type RowError = { class: string; code?: string; message: string; stage: "generate" | "judge" | "timeout" };

export type ResultRow = {
  taskId: string;
  group: string;
  arm: Arm;
  seed: number;
  transparent: boolean;
  width: number;
  height: number;
  status: "ok" | "error";
  error?: RowError;
  /** ComfyUI checkpoint (image model) the arm generated with. */
  checkpoint?: string;
  /** Transparent tasks: how the subject was cut out. Rows from before matting have none (chroma keying). */
  cutout?: "matte" | "chroma-key";
  /** Icon set the task belongs to. */
  set?: string;
  /** References the returned image was generated with (ids). */
  references?: { shape: string[]; set: string[] };
  /** Planner and critic the arm used (absent for raw, which uses neither). */
  models?: { planner: string; critic: string };
  /** Rounds of generation (1 for raw and plan). */
  rounds: number;
  stopReason?: string;
  wallMs: number;
  /** Time inside ComfyUI runs as the client saw it: queue wait plus execution. */
  comfyMs: number;
  /** ComfyUI prompt ids, in round order. */
  promptIds?: string[];
  /** Execution time from ComfyUI history timestamps (excludes queue wait). */
  comfyExecMs?: number;
  /** comfyMs - comfyExecMs: time spent waiting behind other clients' jobs. */
  comfyQueueMs?: number;
  /** LLM use by the arm itself (planner + critic); the judge is not included. */
  llm: { inputTokens: number; outputTokens: number; costUsd: number; estimatedCostUsd: number; reportedCostUsd: number | null };
  /** Every deterministic check the loop computes. */
  checks: CheckResult[];
  /** Ids of the checks that count toward a full pass. */
  scoredChecks: string[];
  judge?: JudgeResult;
  judge2?: JudgeResult;
  imagePath?: string;
  tracePath?: string;
  /** What the judge was shown (the cut-out sheet for transparent assets). */
  judgeViewPath?: string;
  finishedAt: string;
};

/**
 * Which code checks count toward a full pass: size always; alpha and the
 * cut-out's residue check (matte_residue or key_residue) when transparency is
 * required; reference_copy whenever references were used.
 */
export function scoredCheckIds(transparent: boolean, cutout: "matte" | "chroma-key" = "chroma-key", usedReferences = false): string[] {
  return [
    "size",
    ...(transparent ? ["alpha", cutout === "matte" ? "matte_residue" : "key_residue"] : []),
    ...(usedReferences ? ["reference_copy"] : []),
  ];
}

export function rowKey(r: { taskId: string; arm: Arm; seed: number }): string {
  return `${r.taskId}|${r.arm}|${r.seed}`;
}

/** Read results.jsonl. Later rows with the same key replace earlier ones; malformed lines are skipped. */
export async function readResults(file: string): Promise<ResultRow[]> {
  let text: string;
  try {
    text = await readFile(file, "utf8");
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw err;
  }
  return parseResults(text);
}

export function parseResults(text: string): ResultRow[] {
  const byKey = new Map<string, ResultRow>();
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    let row: ResultRow;
    try {
      row = JSON.parse(line) as ResultRow;
    } catch {
      continue; // a line cut short by a crash
    }
    if (!row || typeof row.taskId !== "string" || !ARMS.includes(row.arm)) continue;
    byKey.delete(rowKey(row));
    byKey.set(rowKey(row), row);
  }
  return [...byKey.values()];
}

export async function appendResult(file: string, row: ResultRow): Promise<void> {
  await appendFile(file, `${JSON.stringify(row)}\n`);
}

export type Job = { taskId: string; arm: Arm; seed: number };

/**
 * Jobs still to generate. Rows already present are skipped; with
 * `retryFailed`, rows whose generation errored are run again; with
 * `rerunStopReasons`, rows whose loop stopped for one of those reasons (for
 * example "time-limit" while the shared GPU was busy) are run again. A row with an
 * image but a failed or missing grade is not regenerated: see {@link judgesToRedo}.
 */
export function pendingJobs(jobs: Job[], existing: ResultRow[], options: { retryFailed?: boolean; rerunStopReasons?: string[] } = {}): Job[] {
  const done = new Map(existing.map((r) => [rowKey(r), r]));
  return jobs.filter((j) => {
    const r = done.get(rowKey(j));
    if (!r) return true;
    if (options.retryFailed && r.status === "error") return true;
    return Boolean(r.stopReason && options.rerunStopReasons?.includes(r.stopReason));
  });
}

/** Judges to (re)run on a generated row: a failed primary grade, or a missing/failed second grade on a subset task. */
export function judgesToRedo(row: ResultRow, inSubset: boolean, judge2Enabled: boolean): ("judge" | "judge2")[] {
  if (row.status !== "ok" || !row.judgeViewPath) return [];
  const out: ("judge" | "judge2")[] = [];
  if (!row.judge || row.judge.error) out.push("judge");
  if (inSubset && judge2Enabled && (!row.judge2 || row.judge2.error)) out.push("judge2");
  return out;
}
