// MCP stdio server exposing the ComfyUI client and the asset loop as tools:
//   comfy_inspect        — server, nodes summary, installed models
//   comfy_run_workflow   — run a given graph (API or UI format) once
//   comfy_generate_asset — the planner/critic loop
//   comfy_get_trace      — the full trace of an earlier comfy_generate_asset

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

import { ComfyClient } from "./client.js";
import { resolveComfyAgentConfig, type ComfyAgentConfig } from "./config.js";
import { ComfyAgentError } from "./errors.js";
import { availableBaseGraphs } from "./graphs.js";
import { decodePng, encodePng, fitWithin } from "./image.js";
import { AiandClient } from "./llm.js";
import { generateAsset, type AssetTrace } from "./loop.js";
import { summarizeCapabilities } from "./object-info.js";
import { runWorkflow } from "./workflow-run.js";

export const TOOLS = [
  {
    name: "comfy_inspect",
    description: "Describe the configured ComfyUI server: version, devices, installed checkpoints/LoRAs/upscale models, samplers, and which base graphs it can run.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
  },
  {
    name: "comfy_run_workflow",
    description: "Run one ComfyUI workflow (API or UI format JSON) once. Optional prompt/negative/size/seed fill the matching slots. Without a workflow, runs the default text-to-image graph. Writes the PNG and returns its path.",
    inputSchema: {
      type: "object",
      properties: {
        workflow: { type: "object", description: "Workflow JSON (API format preferred)." },
        prompt: { type: "string" },
        negative: { type: "string" },
        width: { type: "integer" },
        height: { type: "integer" },
        seed: { type: "integer" },
        checkpoint: { type: "string" },
        output_path: { type: "string", description: "Where to write the PNG (default: output dir)." },
      },
      additionalProperties: false,
    },
  },
  {
    name: "comfy_generate_asset",
    description: "Generate an image asset with a plan → generate → check → critique → repair loop. Returns the best image path, the stop reason, the checklist score and a trace id for comfy_get_trace.",
    inputSchema: {
      type: "object",
      properties: {
        prompt: { type: "string" },
        width: { type: "integer" },
        height: { type: "integer" },
        transparent: { type: "boolean" },
        style: { type: "string", description: "Style and palette notes." },
        negative: { type: "string", description: "Things the image must not contain." },
        checkpoint: { type: "string" },
        seed: { type: "integer" },
        max_rounds: { type: "integer" },
        max_time_s: { type: "number" },
        max_cost_usd: { type: "number" },
        output_path: { type: "string" },
      },
      required: ["prompt", "width", "height"],
      additionalProperties: false,
    },
  },
  {
    name: "comfy_get_trace",
    description: "Return the full JSON trace (per-round graph, image paths, checks, critic output, edits, usage/cost, timing) of a comfy_generate_asset run.",
    inputSchema: {
      type: "object",
      properties: { trace_id: { type: "string" } },
      required: ["trace_id"],
      additionalProperties: false,
    },
  },
] as const;

type ToolResult = { content: ({ type: "text"; text: string } | { type: "image"; data: string; mimeType: string })[]; isError?: boolean };

export type McpHandlerContext = {
  config: ComfyAgentConfig;
  outDir: string;
  comfy?: ComfyClient;
  llm?: AiandClient;
};

export function createToolHandler(ctx: McpHandlerContext) {
  const comfy = ctx.comfy ?? new ComfyClient({ baseUrl: ctx.config.comfyUrl });
  let llm = ctx.llm;
  const traces = new Map<string, AssetTrace>();
  const getLlm = () => {
    llm ??= new AiandClient({
      ...(ctx.config.aiandApiKey ? { apiKey: ctx.config.aiandApiKey } : {}),
      baseUrl: ctx.config.aiandBaseUrl,
      shape: ctx.config.apiShape,
    });
    return llm;
  };

  return async function handle(name: string, args: Record<string, unknown>): Promise<ToolResult> {
    try {
      switch (name) {
        case "comfy_inspect": {
          const [stats, info, folders] = await Promise.all([comfy.systemStats(), comfy.getObjectInfo({ refresh: true }), comfy.listModelFolders().catch(() => [])]);
          return text({ url: comfy.baseUrl, system: stats.system, devices: stats.devices, capabilities: summarizeCapabilities(info), baseGraphs: availableBaseGraphs(info), modelFolders: folders });
        }
        case "comfy_run_workflow": {
          const result = await runWorkflow(comfy, {
            ...(args.workflow !== undefined ? { workflow: args.workflow } : {}),
            ...(typeof args.prompt === "string" ? { prompt: args.prompt } : {}),
            ...(typeof args.negative === "string" ? { negative: args.negative } : {}),
            ...(typeof args.width === "number" ? { width: args.width } : {}),
            ...(typeof args.height === "number" ? { height: args.height } : {}),
            ...(typeof args.seed === "number" ? { seed: args.seed } : {}),
            ...(typeof args.checkpoint === "string" ? { checkpoint: args.checkpoint } : {}),
          });
          const out = typeof args.output_path === "string" ? path.resolve(args.output_path) : path.join(ctx.outDir, `workflow-${result.promptId}.png`);
          await mkdir(path.dirname(out), { recursive: true });
          await writeFile(out, result.png);
          return {
            content: [
              { type: "text", text: JSON.stringify({ path: out, promptId: result.promptId, durationMs: result.durationMs, settings: result.settings, unfilled: result.unfilled, graph: result.graph }, null, 2) },
              preview(result.png),
            ],
          };
        }
        case "comfy_generate_asset": {
          const requestId = `${Date.now().toString(36)}`;
          const workDir = path.join(ctx.outDir, `asset-${requestId}`);
          const result = await generateAsset(
            {
              prompt: String(args.prompt ?? ""),
              width: Number(args.width),
              height: Number(args.height),
              ...(args.transparent === true ? { transparent: true } : {}),
              ...(typeof args.style === "string" ? { style: args.style } : {}),
              ...(typeof args.negative === "string" ? { negative: args.negative } : {}),
              ...(typeof args.checkpoint === "string" ? { checkpoint: args.checkpoint } : {}),
              ...(typeof args.seed === "number" ? { seed: args.seed } : {}),
            },
            {
              comfy,
              llm: getLlm(),
              plannerModel: ctx.config.plannerModel,
              criticModel: ctx.config.criticModel,
              ...(ctx.config.plannerReasoningEffort ? { plannerReasoningEffort: ctx.config.plannerReasoningEffort } : {}),
              ...(ctx.config.criticReasoningEffort ? { criticReasoningEffort: ctx.config.criticReasoningEffort } : {}),
              workDir,
              limits: {
                ...(typeof args.max_rounds === "number" ? { maxRounds: args.max_rounds } : {}),
                ...(typeof args.max_time_s === "number" ? { maxTimeMs: args.max_time_s * 1000 } : {}),
                ...(typeof args.max_cost_usd === "number" ? { maxCostUsd: args.max_cost_usd } : {}),
              },
            },
          );
          traces.set(result.trace.id, result.trace);
          const out = typeof args.output_path === "string" ? path.resolve(args.output_path) : path.join(workDir, "asset.png");
          await mkdir(path.dirname(out), { recursive: true });
          await writeFile(out, result.png);
          await writeFile(path.join(ctx.outDir, `${result.trace.id}.trace.json`), JSON.stringify(result.trace, null, 2));
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  path: out,
                  ok: result.ok,
                  traceId: result.trace.id,
                  stopReason: result.trace.stopReason,
                  rounds: result.trace.rounds.length,
                  best: result.trace.best,
                  totals: result.trace.totals,
                  checklist: result.trace.plan.checklist,
                }, null, 2),
              },
              preview(result.png),
            ],
          };
        }
        case "comfy_get_trace": {
          const id = String(args.trace_id ?? "");
          if (!/^[0-9a-f-]{36}$/i.test(id)) return error(`invalid trace_id "${id}"`);
          const cached = traces.get(id);
          if (cached) return text(cached);
          try {
            return { content: [{ type: "text", text: await readFile(path.join(ctx.outDir, `${id}.trace.json`), "utf8") }] };
          } catch {
            return error(`no trace with id ${id}`);
          }
        }
        default:
          return error(`unknown tool ${name}`);
      }
    } catch (err) {
      const payload = err instanceof ComfyAgentError ? err.toJSON() : { message: err instanceof Error ? err.message : String(err) };
      return { content: [{ type: "text", text: JSON.stringify({ error: payload }, null, 2) }], isError: true };
    }
  };
}

function text(value: unknown): ToolResult {
  return { content: [{ type: "text", text: JSON.stringify(value, null, 2) }] };
}

function error(message: string): ToolResult {
  return { content: [{ type: "text", text: JSON.stringify({ error: { message } }) }], isError: true };
}

function preview(png: Buffer): { type: "image"; data: string; mimeType: string } {
  return { type: "image", data: encodePng(fitWithin(decodePng(png), 512)).toString("base64"), mimeType: "image/png" };
}

export async function startMcpServer(options: { outDir?: string; env?: NodeJS.ProcessEnv } = {}): Promise<void> {
  const env = options.env ?? process.env;
  const config = resolveComfyAgentConfig({}, env);
  const outDir = path.resolve(options.outDir ?? env.COMFY_AGENT_OUT_DIR ?? "comfy-agent-out");
  const handle = createToolHandler({ config, outDir });
  const server = new Server({ name: "comfy-agent", version: "0.1.0" }, { capabilities: { tools: {} } });
  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS.map((t) => ({ ...t })) }));
  server.setRequestHandler(CallToolRequestSchema, async (req) => handle(req.params.name, (req.params.arguments ?? {}) as Record<string, unknown>));
  await server.connect(new StdioServerTransport());
}
