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

import { existsSync, readFileSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";

import { ComfyClient } from "../../src/client.js";
import { chooseModel } from "../../src/graphs.js";
import { resolveComfyAgentConfig, parseApiShape } from "../../src/config.js";
import { AiandClient, type ApiShape } from "../../src/llm.js";
import { runArm } from "./arms.js";
import { JUDGE2_PREFERENCES, JUDGE_PREFERENCES, judgeImage, judgeView, modelFamily, selectJudgeModel } from "./judge.js";
import { ARMS, DEFAULT_ARMS, appendResult, armApplies, judgesToRedo, pendingJobs, readResults, type Arm, type Job, type ResultRow } from "./results.js";
import { appendSetResult, judgeSet, readSetResults, setMembers, setReferencesFor } from "./sets.js";
import { decodePng } from "../../src/image.js";
import { chooseBackgroundRemovalModel } from "../../src/object-info.js";
import { ReferenceLibrary } from "../../src/references.js";
import { writeReports } from "./report.js";
import { checkpointOf, comfyTiming, foreignCheckpoints, llmModelsOf } from "./timing.js";
import { DEFAULT_RELIABILITY_SUBSET, loadTasks, selectTasks, TASKS_PATH, type EvalTask } from "./tasks.js";

const HELP = `Usage: eval:design-assets --out <dir> [options]

  --out <dir>             run directory (required); rerun with the same dir to resume
  --tasks id,id           only these task ids
  --groups g,g            only these groups
  --arms raw,plan,loop    arms to run (default raw,plan,loop). Ablations, each run only on the
                          tasks it measures: loop-chroma (chroma key instead of matting;
                          transparent tasks), loop-noref (no shape references; icon tasks),
                          loop-noset (no style references from the set; set tasks)
  --reference-cache <dir> cache for emoji / Material Symbols references (default <out>/reference-cache)
  --no-references         no shape references for icon tasks in any arm
  --seeds 1,2             seeds; every arm uses the same seeds (default 1)
  --max-rounds <n>        rounds for the loop arm (default 4)
  --judge <model>         judge model; default: picked from the ai& catalog, preferring
                          ${JUDGE_PREFERENCES.join(", ")}, never from the planner's or critic's family
  --judge2 <model|none>   second judge on the reliability subset; default: picked the same way,
                          preferring ${JUDGE2_PREFERENCES.join(", ")}, also not the first judge's family
  --subset id,id          reliability subset (default ${DEFAULT_RELIABILITY_SUBSET.join(",")})
  --shape openai|anthropic  API shape for planner, critic and judges (default from COMFY_AGENT_API_SHAPE, else openai)
  --planner <model>       planner model (default: config, else picked from the ai& catalog)
  --critic <model>        critic model (default: config, else picked from the ai& catalog)
  --task-timeout-min <n>  deadline per task x arm (default 25)
  --judge-concurrency <n> judge calls in flight while the next image generates (default 4)
  --retry-failed          rerun rows whose generation failed
  --rerun-stop-reasons r,r  rerun rows whose loop stopped for these reasons (e.g. time-limit,
                          after other clients kept the shared ComfyUI busy)
  --report-only           only rebuild summary and contact sheets from results.jsonl
  --allow-mixed-models    resume a run even though the server default image model, or the
                          planner, critic or judges the catalog picks, changed since it started
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
      "allow-mixed-models": { type: "boolean" },
      "tasks-file": { type: "string" },
      "reference-cache": { type: "string" },
      "no-references": { type: "boolean" },
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
  const arms = (list(values.arms) ?? [...DEFAULT_ARMS]) as Arm[];
  const badArm = arms.find((a) => !ARMS.includes(a));
  if (badArm) throw new Error(`unknown arm "${badArm}"; use ${ARMS.join(", ")}`);
  const seeds = (list(values.seeds) ?? ["1"]).map((s) => {
    const n = Number(s);
    if (!Number.isInteger(n) || n < 0) throw new Error(`invalid seed "${s}"`);
    return n;
  });
  const subset = new Set(list(values.subset) ?? DEFAULT_RELIABILITY_SUBSET);

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

    // Resolve every model once against the live ai& catalog, so all tasks and
    // arms use the same ids. Judges come from families the judged models do not use.
    const plannerModel = await llm.resolveModel("planner", cfg.plannerModel);
    const criticModel = await llm.resolveModel("critic", cfg.criticModel);
    const catalog = await llm.catalog.list();
    const familyOf = (id: string) => modelFamily(id, catalog.find((m) => m.id === id)?.provider);
    const judgeModel = selectJudgeModel(catalog, {
      ...(values.judge ? { configured: values.judge } : {}),
      preferred: JUDGE_PREFERENCES,
      excludeFamilies: [familyOf(plannerModel), familyOf(criticModel)],
    });
    const judge2Model =
      values.judge2 === "none"
        ? undefined
        : selectJudgeModel(catalog, {
            ...(values.judge2 ? { configured: values.judge2 } : {}),
            preferred: JUDGE2_PREFERENCES,
            excludeFamilies: [familyOf(plannerModel), familyOf(criticModel), familyOf(judgeModel)],
          });
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

    log(`config: planner ${plannerModel}, critic ${criticModel}, judge ${judgeModel}, judge2 ${judge2Model ?? "none"}, shape ${shape}, comfy ${cfg.comfyUrl}`);

    // A resumed run must keep the models it started with; otherwise the
    // comparison would be between models, not arms. Refuse to mix unless asked.
    const existing = await readResults(resultsFile);
    const objectInfo = await comfy.getObjectInfo();
    const checkpoint = chooseModel(objectInfo).model;
    // Transparent tasks are cut out with the matting model when installed; rows
    // from before matting (no cutout field) were chroma keyed.
    const cutoutNow = chooseBackgroundRemovalModel(objectInfo) ? "matte" : "chroma-key";
    const existingCutouts = existing.filter((r) => r.status === "ok" && r.transparent && r.arm !== "loop-chroma").map((r) => r.cutout ?? "chroma-key");
    const existingCheckpoints = await Promise.all(existing.filter((r) => r.status === "ok").map((r) => checkpointOf(r)));
    const existingLlms = await Promise.all(existing.filter((r) => r.status === "ok").map((r) => llmModelsOf(r)));
    const mixed = [
      ["image model", foreignCheckpoints(existingCheckpoints, checkpoint), checkpoint],
      ["cut-out method", foreignCheckpoints(existingCutouts, cutoutNow), cutoutNow],
      ["planner", foreignCheckpoints(existingLlms.map((m) => m?.planner), plannerModel), plannerModel],
      ["critic", foreignCheckpoints(existingLlms.map((m) => m?.critic), criticModel), criticModel],
      ["judge", foreignCheckpoints(existing.map((r) => r.judge?.model), judgeModel), judgeModel],
      ["second judge", judge2Model ? foreignCheckpoints(existing.map((r) => r.judge2?.model), judge2Model) : [], judge2Model ?? ""],
    ] as const;
    const conflicts = mixed.filter(([, found]) => found.length > 0).map(([what, found, now]) => `${what}: run has ${found.join(", ")}, now ${now}`);
    if (conflicts.length) {
      if (!values["allow-mixed-models"]) {
        throw new Error(`${outDir} was started with other models (${conflicts.join("; ")}). Use a new --out, or pass --allow-mixed-models.`);
      }
      log(`warning: mixing models in one run: ${conflicts.join("; ")}`);
    }
    log(`image model (server default checkpoint): ${checkpoint}`);

    // Rows that already have an image: fill in the checkpoint and ComfyUI
    // execution time if missing, and redo missing or failed grades.
    const inFlight: Promise<void>[] = [];
    const selected = new Set(tasks.map((t) => t.id));
    for (const row of existing) {
      if (!selected.has(row.taskId)) continue;
      const timing = row.status === "ok" && row.comfyExecMs === undefined ? await comfyTiming(comfy, row) : undefined;
      const rowCheckpoint = row.status === "ok" && !row.checkpoint ? await checkpointOf(row) : undefined;
      const rowModels = row.status === "ok" && !row.models ? await llmModelsOf(row) : undefined;
      const changed = Boolean(timing || rowCheckpoint || rowModels);
      const current: ResultRow = {
        ...row,
        ...(timing ?? {}),
        ...(rowCheckpoint ? { checkpoint: rowCheckpoint } : {}),
        ...(rowModels ? { models: rowModels } : {}),
      };
      const redo = judgesToRedo(current, subset.has(row.taskId), Boolean(judge2Model));
      if (redo.length) inFlight.push(gradeAndWrite(current, redo));
      else if (changed) await write(current);
    }

    const rerunStopReasons = list(values["rerun-stop-reasons"]) ?? [];
    const jobs: Job[] = tasks.flatMap((t) => seeds.flatMap((seed) => arms.filter((arm) => armApplies(arm, t)).map((arm) => ({ taskId: t.id, arm, seed }))));
    // Shape references for icon tasks: fetched from pinned sources and cached.
    const library = values["no-references"]
      ? undefined
      : new ReferenceLibrary({ cacheDir: path.resolve(values["reference-cache"] ?? path.join(outDir, "reference-cache")) });
    // Rows known so far (existing and new), for set references and set judging.
    const known: ResultRow[] = [...existing];
    const setResultsFile = path.join(outDir, "set-results.jsonl");
    const judgedSets = new Set((await readSetResults(setResultsFile)).filter((j) => !j.error).map((j) => `${j.set}|${j.arm}|${j.seed}`));
    const readImage = (p: string) => (existsSync(p) ? readFileSync(p) : undefined);
    const judgeSetIfComplete = async (set: string, arm: Arm, seed: number) => {
      const key = `${set}|${arm}|${seed}`;
      if (judgedSets.has(key)) return;
      const members = setMembers(allTasks, set);
      const rows = members.map((m) => known.find((r) => r.taskId === m.id && r.arm === arm && r.seed === seed && r.status === "ok" && r.imagePath));
      if (rows.some((r) => !r || !existsSync(r.imagePath!))) return;
      const images = rows.map((r) => ({ id: r!.taskId, image: decodePng(readFileSync(r!.imagePath!)) }));
      const j = await runJudge(() => judgeSet(set, arm, seed, images, { llm, model: judgeModel, shape }));
      judgedSets.add(key);
      await appendSetResult(setResultsFile, j);
      log(`set ${set} ${arm} s${seed}: ${j.error ? `error: ${j.error}` : `score ${j.score}/5${j.outliers?.length ? `, off-style: ${j.outliers.join(", ")}` : ""}`}`);
    };
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
        plannerModel,
        criticModel,
        shape,
        ...(cfg.plannerReasoningEffort ? { plannerReasoningEffort: cfg.plannerReasoningEffort } : {}),
        ...(cfg.criticReasoningEffort ? { criticReasoningEffort: cfg.criticReasoningEffort } : {}),
        loopMaxRounds: maxRounds,
        timeoutMs,
        workDir,
        ...(library && task.group === "icon" ? { library } : {}),
        setReferences: setReferencesFor(task, job.arm, job.seed, allTasks, known, readImage),
        onProgress: (m) => {
          if (!/step \d+\/\d+/.test(m)) log(`  ${m}`);
        },
      });
      let row: ResultRow = { ...out.row, finishedAt: new Date().toISOString() };
      if (row.status === "ok") {
        const timing = await comfyTiming(comfy, row);
        if (timing) row = { ...row, ...timing };
      }
      known.push(row);
      if (row.status === "error") {
        log(`  failed: ${row.error?.class}: ${row.error?.message}`);
        await write(row);
        continue;
      }
      if (task.set) inFlight.push(judgeSetIfComplete(task.set, job.arm, job.seed).catch((err) => log(`set ${task.set}: ${String(err)}`)));
      const queued = row.comfyQueueMs ? `, ${(row.comfyQueueMs / 1000).toFixed(0)} s queued behind other ComfyUI clients` : "";
      log(`  generated in ${(row.wallMs / 1000).toFixed(0)} s${queued}, ${row.rounds} round(s), ${row.stopReason ?? ""}, $${row.llm.costUsd.toFixed(4)}`);
      row.judgeViewPath = path.join(workDir, "judge-view.png");
      await writeFile(row.judgeViewPath, judgeView(out.image!, task.transparent));
      // Grading overlaps with the next generation.
      const which: ("judge" | "judge2")[] = ["judge", ...(judge2Model && subset.has(task.id) ? (["judge2"] as const) : [])];
      inFlight.push(gradeAndWrite(row, which));
    }
    // Sets finished in an earlier run but never judged (for example after a crash).
    for (const t of tasks) {
      if (!t.set) continue;
      for (const seed of seeds) for (const arm of arms) if (armApplies(arm, t)) inFlight.push(judgeSetIfComplete(t.set, arm, seed).catch(() => undefined));
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
