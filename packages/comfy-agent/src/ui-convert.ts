import { WorkflowFormatError } from "./errors.js";
import { widgetOrder } from "./object-info.js";
import type { ApiWorkflow, ObjectInfo } from "./types.js";
import { isApiWorkflow } from "./validate.js";

// UI-format ("workflow") JSON is what the ComfyUI editor saves: a node list
// with positions, a link table, and positional `widgets_values`. The server
// only executes API format, so we convert using the server's own
// `/object_info` widget order. Subgraphs are rejected: flattening them needs
// the frontend's expansion rules, and guessing would run a different graph
// from the one the user built.

type UiInput = { name: string; type?: string; link: number | null; widget?: { name: string } };
type UiOutput = { name?: string; type?: string; links?: number[] | null };
type UiNode = {
  id: number | string;
  type: string;
  mode?: number;
  inputs?: UiInput[];
  outputs?: UiOutput[];
  widgets_values?: unknown[] | Record<string, unknown>;
  title?: string;
};
/** [id, fromNode, fromSlot, toNode, toSlot, type] */
type UiLinkTuple = [number, number | string, number, number | string, number, string];
type UiLinkObject = { id: number; origin_id: number | string; origin_slot: number; target_id: number | string; target_slot: number; type?: string };
export type UiWorkflow = {
  nodes: UiNode[];
  links?: (UiLinkTuple | UiLinkObject)[];
  definitions?: { subgraphs?: unknown[] };
  [k: string]: unknown;
};

const MODE_MUTED = 2;
const MODE_BYPASS = 4;
// Frontend-only nodes: they never reach the server.
const VIRTUAL_NODE_TYPES = new Set(["Note", "MarkdownNote", "PrimitiveNode", "Reroute", "PrimitiveString", "PrimitiveInt"]);
const CONTROL_VALUES = new Set(["fixed", "increment", "decrement", "randomize"]);
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function isUiWorkflow(value: unknown): value is UiWorkflow {
  return !!value && typeof value === "object" && Array.isArray((value as { nodes?: unknown }).nodes);
}

export type WorkflowFormat = "api" | "ui";

export function detectWorkflowFormat(value: unknown): WorkflowFormat {
  if (isUiWorkflow(value)) return "ui";
  if (isApiWorkflow(value)) return "api";
  // Some exporters wrap the API graph as { prompt: {...} }.
  if (value && typeof value === "object" && isApiWorkflow((value as { prompt?: unknown }).prompt)) return "api";
  throw new WorkflowFormatError("unrecognised workflow JSON: expected API format ({ id: { class_type, inputs } }) or UI format ({ nodes, links })");
}

/** Accept either format and return an API-format graph. */
export function toApiWorkflow(value: unknown, info: ObjectInfo): ApiWorkflow {
  const format = detectWorkflowFormat(value);
  if (format === "api") {
    const graph = isApiWorkflow(value) ? value : (value as { prompt: ApiWorkflow }).prompt;
    return structuredClone(graph);
  }
  return convertUiWorkflow(value as UiWorkflow, info);
}

export function convertUiWorkflow(ui: UiWorkflow, info: ObjectInfo): ApiWorkflow {
  const subgraphs = ui.definitions?.subgraphs;
  if (Array.isArray(subgraphs) && subgraphs.length > 0) {
    throw new WorkflowFormatError(
      `UI workflow contains ${subgraphs.length} subgraph definition(s); subgraphs are not supported. Export the workflow in API format (Workflow → Export (API)) and pass that instead.`,
    );
  }
  for (const node of ui.nodes) {
    if (typeof node.type === "string" && UUID_RE.test(node.type)) {
      throw new WorkflowFormatError(`UI workflow node ${node.id} is a subgraph instance; subgraphs are not supported. Export in API format instead.`);
    }
  }

  const nodesById = new Map<string, UiNode>(ui.nodes.map((n) => [String(n.id), n]));
  const links = new Map<number, { from: string; fromSlot: number; type?: string }>();
  for (const raw of ui.links ?? []) {
    if (Array.isArray(raw)) {
      links.set(raw[0], { from: String(raw[1]), fromSlot: raw[2], type: raw[5] });
    } else if (raw && typeof raw === "object") {
      links.set(raw.id, { from: String(raw.origin_id), fromSlot: raw.origin_slot, type: raw.type });
    }
  }

  // Resolve a link to its real producer, walking through Reroute and bypassed nodes.
  const resolveLink = (linkId: number, depth = 0): [string, number] | null => {
    if (depth > 64) throw new WorkflowFormatError("link cycle through Reroute/bypassed nodes");
    const link = links.get(linkId);
    if (!link) return null;
    const src = nodesById.get(link.from);
    if (!src) return null;
    if (src.type === "Reroute") {
      const upstream = src.inputs?.[0]?.link;
      return upstream == null ? null : resolveLink(upstream, depth + 1);
    }
    if (src.mode === MODE_BYPASS) {
      // A bypassed node passes through the first input whose type matches the output.
      const outType = src.outputs?.[link.fromSlot]?.type ?? link.type;
      const passthrough = src.inputs?.find((i) => i.link != null && (!outType || i.type === outType));
      return passthrough?.link == null ? null : resolveLink(passthrough.link, depth + 1);
    }
    if (src.mode === MODE_MUTED) return null;
    if (src.type === "PrimitiveNode") return null; // its value is already in the target's widgets_values
    return [String(src.id), link.fromSlot];
  };

  const graph: ApiWorkflow = {};
  for (const node of ui.nodes) {
    if (VIRTUAL_NODE_TYPES.has(node.type)) continue;
    if (node.mode === MODE_MUTED || node.mode === MODE_BYPASS) continue;
    const def = info[node.type];
    if (!def) {
      throw new WorkflowFormatError(`UI workflow node ${node.id} uses class "${node.type}", which this server does not have; cannot map its widget values`);
    }
    const inputs: Record<string, unknown> = {};

    // Widgets first, consuming widgets_values positionally.
    const widgets = widgetOrder(def);
    const values = node.widgets_values;
    if (values && !Array.isArray(values)) {
      // Some custom nodes store widgets as a name → value map.
      for (const w of widgets) if (w.name in values) inputs[w.name] = values[w.name];
    } else if (Array.isArray(values)) {
      let cursor = 0;
      for (const w of widgets) {
        if (cursor >= values.length) break;
        inputs[w.name] = values[cursor];
        cursor += 1;
        if (w.controlAfterGenerate && typeof values[cursor] === "string" && CONTROL_VALUES.has(values[cursor] as string)) {
          cursor += 1;
        }
      }
    }

    // Then links, which override any widget that was converted to an input.
    for (const input of node.inputs ?? []) {
      if (input.link == null) continue;
      const resolved = resolveLink(input.link);
      const name = input.widget?.name ?? input.name;
      if (resolved) inputs[name] = resolved;
    }

    graph[String(node.id)] = {
      class_type: node.type,
      inputs,
      ...(node.title ? { _meta: { title: node.title } } : {}),
    };
  }

  // Drop links that point at nodes we removed (muted producers).
  for (const node of Object.values(graph)) {
    for (const [name, value] of Object.entries(node.inputs)) {
      if (Array.isArray(value) && value.length === 2 && typeof value[1] === "number" && !graph[String(value[0])]) {
        delete node.inputs[name];
      }
    }
  }
  return graph;
}
