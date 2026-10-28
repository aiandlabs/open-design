// Design-asset eval runner.
//
//   pnpm --filter @open-design/comfy-agent eval:design-assets --out <dir> [flags]
//
// Runs every selected task under the raw / plan / loop arms with the same
// seeds, grades each final image with an independent judge (and a second
// judge on a fixed subset), and writes results.jsonl, summary.{md,json},
// contact-sheet.html and judge-subset.html into --out. Rerunning with the
// same --out resumes: finished rows are skipped, missing grades are redone.
//
// Needs COMFYUI_URL (default http://127.0.0.1:8188) and AIAND_API_KEY.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";

import { ComfyClient } from "../../src/client.js";
import { resolveComfyAgentConfig, parseApiShape } from "../../src/config.js";
import { AiandClient, type ApiShape } from "../../src/llm.js";
import { runArm } from "./arms.js";
import { DEFAULT_JUDGE2_MODEL, DEFAULT_JUDGE_MODEL, judgeImage, judgeView } from "./judge.js";
import { ARMS, appendResult, judgesToRedo, pendingJobs, readResults, type Arm, type Job, type ResultRow } from "./results.js";
import { writeReports } from "./report.js";
import { comfyTiming } from "./timing.js";
import { DEFAULT_RELIABILITY_SUBSET, loadTasks, selectTasks, TASKS_PATH, type EvalTask } from "./tasks.js";

const HELP = `Usage: eval:design-assets --out <dir> [options]

  --out <dir>             run directory (required); rerun with the same dir to resume
  --tasks id,id           only these task ids
  --groups g,g            only these groups
  --arms raw,plan,loop    arms to run (default all three)
  --seeds 1,2             seeds; every arm uses the same seeds (default 1)
  --max-rounds <n>        rounds for the loop arm (default 4)
  --judge <model>         judge model (default ${DEFAULT_JUDGE_MODEL})
  --judge2 <model|none>   second judge on the reliability subset (default ${DEFAULT_JUDGE2_MODEL})
  --subset id,id          reliability subset (default ${DEFAULT_RELIABILITY_SUBSET.join(",")})
  --shape openai|anthropic  API shape for planner, critic and judges (default from COMFY_AGENT_API_SHAPE, else openai)
  --planner <model>       planner model (default from config)
  --critic <model>        critic model (default from config)
  --task-timeout-min <n>  deadline per task x arm (default 25)
  --judge-concurrency <n> judge calls in flight while the next image generates (default 4)
  --retry-failed          rerun rows whose generation failed
  --rerun-stop-reasons r,r  rerun rows whose loop stopped for these reasons (e.g. time-limit,
                          after other clients kept the shared ComfyUI busy)
  --report-only           only rebuild summary and contact sheets from results.jsonl
  --tasks-file <path>     task set (default evals/design-assets/tasks.json)
`;

const list = (v: string | undefined) => (v ? v.split(",").map((s) => s.trim()).filter(Boolean) : undefined);

function log(message: string): void {
  process.stderr.write(`[${new Date().toISOString().slice(11, 19)}] ${message}\n`);
}

/** Bounded concurrency for judge calls. */
function limiter(max: number) {
  let active = 0;
  const waiting: (() => void)[] = [];
  return async <T>(fn: () => Promise<T>): Promise<T> => {
    if (active >= max) await new Promise<void>((r) => waiting.push(r));
    active += 1;
    try {
      return await fn();
    } finally {
      active -= 1;
      waiting.shift()?.();
    }
  };
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      out: { type: "string" },
      tasks: { type: "string" },
      groups: { type: "string" },
      arms: { type: "string" },
      seeds: { type: "string" },
      "max-rounds": { type: "string" },
      judge: { type: "string" },
      judge2: { type: "string" },
      subset: { type: "string" },
      shape: { type: "string" },
      planner: { type: "string" },
      critic: { type: "string" },
      "task-timeout-min": { type: "string" },
      "judge-concurrency": { type: "string" },
      "retry-failed": { type: "boolean" },
      "rerun-stop-reasons": { type: "string" },
      "report-only": { type: "boolean" },
      "tasks-file": { type: "string" },
      help: { type: "boolean" },
    },
    allowPositionals: false,
  });
  if (values.help) {
    process.stdout.write(HELP);
    return;
  }
  if (!values.out) throw new Error(`--out is required\n\n${HELP}`);
  const outDir = path.resolve(values.out);
  const resultsFile = path.join(outDir, "results.jsonl");
  await mkdir(outDir, { recursive: true });

  const allTasks = await loadTasks(values["tasks-file"] ? path.resolve(values["tasks-file"]) : TASKS_PATH);
  const tasks = selectTasks(allTasks, { ...(list(values.tasks) ? { ids: list(values.tasks)! } : {}), ...(list(values.groups) ? { groups: list(values.groups)! } : {}) });
  const arms = (list(values.arms) ?? [...ARMS]) as Arm[];
  const badArm = arms.find((a) => !ARMS.includes(a));
  if (badArm) throw new Error(`unknown arm "${badArm}"; use ${ARMS.join(", ")}`);
  const seeds = (list(values.seeds) ?? ["1"]).map((s) => {
    const n = Number(s);
    if (!Number.isInteger(n) || n < 0) throw new Error(`invalid seed "${s}"`);
    return n;
  });
  const subset = new Set(list(values.subset) ?? DEFAULT_RELIABILITY_SUBSET);
  const judgeModel = values.judge ?? DEFAULT_JUDGE_MODEL;
  const judge2Model = values.judge2 === "none" ? undefined : (values.judge2 ?? DEFAULT_JUDGE2_MODEL);

  if (!values["report-only"]) {
    const shapeFlag = values.shape ? parseApiShape(values.shape) : undefined;
    if (values.shape && !shapeFlag) throw new Error(`invalid --shape "${values.shape}"`);
    const cfg = resolveComfyAgentConfig({
      ...(shapeFlag ? { apiShape: shapeFlag } : {}),
      ...(values.planner ? { plannerModel: values.planner } : {}),
      ...(values.critic ? { criticModel: values.critic } : {}),
    });
    const shape: ApiShape = cfg.apiShape;
    const comfy = new ComfyClient({ baseUrl: cfg.comfyUrl });
    const llm = new AiandClient({ ...(cfg.aiandApiKey ? { apiKey: cfg.aiandApiKey } : {}), baseUrl: cfg.aiandBaseUrl, shape });
    await comfy.getObjectInfo();
    const timeoutMs = Number(values["task-timeout-min"] ?? 25) * 60 * 1000;
    const maxRounds = Number(values["max-rounds"] ?? 4);
    if (!Number.isInteger(maxRounds) || maxRounds < 1) throw new Error("--max-rounds must be a positive integer");
    const runJudge = limiter(Number(values["judge-concurrency"] ?? 4));
    let writeChain = Promise.resolve();
    const write = (row: ResultRow) => (writeChain = writeChain.then(() => appendResult(resultsFile, row)));
    const taskById = new Map(allTasks.map((t) => [t.id, t]));

    const grade = async (row: ResultRow, which: ("judge" | "judge2")[]): Promise<ResultRow> => {
      const task = taskById.get(row.taskId)!;
      const view = await readFile(row.judgeViewPath!);
      const next = { ...row };
      await Promise.all(
        which.map(async (w) => {
          const model = w === "judge" ? judgeModel : judge2Model!;
          const res = await runJudge(() => judgeImage(task, view, { llm, model, shape }));
          next[w] = res;
          log(`${row.taskId} ${row.arm} s${row.seed}: ${w} ${model} ${res.error ? `error: ${res.error}` : `${res.items.filter((i) => i.pass).length}/${res.items.length}`}`);
        }),
      );
      next.finishedAt = new Date().toISOString();
      return next;
    };

    // Never let a grading failure reject: it would surface as an unhandled
    // rejection while the next image is generating. The row is still written
    // without grades, and the next resume grades it.
    const gradeAndWrite = (row: ResultRow, which: ("judge" | "judge2")[]): Promise<void> =>
      grade(row, which)
        .catch((err) => {
          log(`${row.taskId} ${row.arm} s${row.seed}: grading failed: ${err instanceof Error ? err.message : String(err)}`);
          return row;
        })
        .then(write);

    log(`config: planner ${cfg.plannerModel}, critic ${cfg.criticModel}, judge ${judgeModel}, judge2 ${judge2Model ?? "none"}, shape ${shape}, comfy ${cfg.comfyUrl}`);

    // Rows that already have an image: fill in ComfyUI execution time from the
    // server history if it is missing, and redo missing or failed grades.
    const existing = await readResults(resultsFile);
    const inFlight: Promise<void>[] = [];
    const selected = new Set(tasks.map((t) => t.id));
    for (const row of existing) {
      if (!selected.has(row.taskId)) continue;
      const timing = row.status === "ok" && row.comfyExecMs === undefined ? await comfyTiming(comfy, row) : undefined;
      const current = timing ? { ...row, ...timing } : row;
      const redo = judgesToRedo(current, subset.has(row.taskId), Boolean(judge2Model));
      if (redo.length) inFlight.push(gradeAndWrite(current, redo));
      else if (timing) await write(current);
    }

    const rerunStopReasons = list(values["rerun-stop-reasons"]) ?? [];
    const jobs: Job[] = tasks.flatMap((t) => seeds.flatMap((seed) => arms.map((arm) => ({ taskId: t.id, arm, seed }))));
    const todo = pendingJobs(jobs, existing, { retryFailed: Boolean(values["retry-failed"]), rerunStopReasons });
    log(`${jobs.length} jobs, ${jobs.length - todo.length} already done, ${todo.length} to run`);
    let stopping = false;
    process.once("SIGINT", () => {
      stopping = true;
      log("SIGINT: finishing the current job, then stopping");
    });

    for (const [i, job] of todo.entries()) {
      if (stopping) break;
      const task = taskById.get(job.taskId) as EvalTask;
      const workDir = path.join(outDir, "images", job.taskId, `${job.arm}-s${job.seed}`);
      log(`(${i + 1}/${todo.length}) ${job.taskId} ${job.arm} seed ${job.seed}`);
      // Generation is strictly one at a time: ComfyUI is one local GPU, and
      // overlapping runs would make per-task ComfyUI time meaningless.
      const out = await runArm(job.arm, task, job.seed, {
        comfy,
        llm,
        plannerModel: cfg.plannerModel,
        criticModel: cfg.criticModel,
        shape,
        ...(cfg.plannerReasoningEffort ? { plannerReasoningEffort: cfg.plannerReasoningEffort } : {}),
        ...(cfg.criticReasoningEffort ? { criticReasoningEffort: cfg.criticReasoningEffort } : {}),
        loopMaxRounds: maxRounds,
        timeoutMs,
        workDir,
        onProgress: (m) => {
          if (!/step \d+\/\d+/.test(m)) log(`  ${m}`);
        },
      });
      let row: ResultRow = { ...out.row, finishedAt: new Date().toISOString() };
      if (row.status === "ok") {
        const timing = await comfyTiming(comfy, row);
        if (timing) row = { ...row, ...timing };
      }
      if (row.status === "error") {
        log(`  failed: ${row.error?.class}: ${row.error?.message}`);
        await write(row);
        continue;
      }
      const queued = row.comfyQueueMs ? `, ${(row.comfyQueueMs / 1000).toFixed(0)} s queued behind other ComfyUI clients` : "";
      log(`  generated in ${(row.wallMs / 1000).toFixed(0)} s${queued}, ${row.rounds} round(s), ${row.stopReason ?? ""}, $${row.llm.costUsd.toFixed(4)}`);
      row.judgeViewPath = path.join(workDir, "judge-view.png");
      await writeFile(row.judgeViewPath, judgeView(out.image!, task.transparent));
      // Grading overlaps with the next generation.
      const which: ("judge" | "judge2")[] = ["judge", ...(judge2Model && subset.has(task.id) ? (["judge2"] as const) : [])];
      inFlight.push(gradeAndWrite(row, which));
    }
    await Promise.all(inFlight);
    await writeChain;
  }

  const written = await writeReports(outDir, allTasks, await readResults(resultsFile));
  log(`wrote ${written.join(", ")} in ${outDir}`);
}

main().catch((err) => {
  process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
  process.exitCode = 1;
});
