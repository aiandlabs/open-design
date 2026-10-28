import { GraphValidationError, WorkflowFormatError, type GraphIssue } from "./errors.js";
import { inputSpecs } from "./object-info.js";
import type { ApiWorkflow, ObjectInfo } from "./types.js";

export function isLink(value: unknown): value is [string, number] {
  return (
    Array.isArray(value)
    && value.length === 2
    && (typeof value[0] === "string" || typeof value[0] === "number")
    && typeof value[1] === "number"
  );
}

/** Shape check only: is this an API-format workflow object? */
export function isApiWorkflow(value: unknown): value is ApiWorkflow {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const entries = Object.entries(value as Record<string, unknown>);
  if (entries.length === 0) return false;
  return entries.every(([, node]) =>
    !!node
    && typeof node === "object"
    && typeof (node as { class_type?: unknown }).class_type === "string"
    && typeof (node as { inputs?: unknown }).inputs === "object",
  );
}

export function assertApiWorkflowShape(value: unknown): ApiWorkflow {
  if (!isApiWorkflow(value)) {
    throw new WorkflowFormatError("not an API-format ComfyUI workflow: expected an object of { class_type, inputs } nodes");
  }
  return value;
}

/**
 * Check a graph against the server's `/object_info` before queueing it:
 * every class exists, every required input is present, links point at real
 * outputs of a compatible type, COMBO values (checkpoints, LoRAs, samplers)
 * are installed, and numbers sit inside the declared range.
 */
export function validateGraph(graph: ApiWorkflow, info: ObjectInfo): GraphIssue[] {
  const issues: GraphIssue[] = [];
  let hasOutputNode = false;
  for (const [nodeId, node] of Object.entries(graph)) {
    const def = info[node.class_type];
    if (!def) {
      issues.push({ nodeId, classType: node.class_type, kind: "unknown-class", message: `node class "${node.class_type}" is not installed on this server` });
      continue;
    }
    if (def.api_node) {
      issues.push({ nodeId, classType: node.class_type, kind: "cloud-node", message: `node class "${node.class_type}" is a partner API node that calls a cloud service; only local nodes may run` });
      continue;
    }
    if (def.output_node) hasOutputNode = true;
    for (const spec of inputSpecs(def)) {
      if (spec.type === AUTOGROW) {
        checkAutogrow(graph, info, nodeId, node.class_type, node.inputs, spec.name, def, issues);
        continue;
      }
      const present = Object.prototype.hasOwnProperty.call(node.inputs, spec.name);
      const value = node.inputs[spec.name];
      if (!present || value === undefined || value === null) {
        if (spec.required) {
          issues.push({ nodeId, classType: node.class_type, input: spec.name, kind: "missing-input", message: `required input "${spec.name}" (${spec.type}) is missing` });
        }
        continue;
      }
      if (isLink(value)) {
        const [srcId, slot] = [String(value[0]), value[1]];
        const src = graph[srcId];
        if (!src) {
          issues.push({ nodeId, classType: node.class_type, input: spec.name, kind: "bad-link", message: `links to node "${srcId}", which is not in the graph` });
          continue;
        }
        const srcDef = info[src.class_type];
        if (!srcDef) continue; // reported on the source node itself
        const outType = srcDef.output[slot];
        if (outType === undefined) {
          issues.push({ nodeId, classType: node.class_type, input: spec.name, kind: "bad-link", message: `links to output ${slot} of ${src.class_type}, which has ${srcDef.output.length} outputs` });
          continue;
        }
        if (!typesCompatible(outType, spec.type)) {
          issues.push({ nodeId, classType: node.class_type, input: spec.name, kind: "link-type", message: `expects ${spec.type} but is linked to ${src.class_type} output ${slot} (${outType})` });
        }
        continue;
      }
      if (!spec.isWidget && !spec.forceInput) {
        issues.push({ nodeId, classType: node.class_type, input: spec.name, kind: "bad-type", message: `expects a ${spec.type} link, got an inline value` });
        continue;
      }
      if (spec.type === "COMBO") {
        const options = spec.options ?? [];
        if (!spec.imageUpload && !options.includes(String(value))) {
          issues.push({
            nodeId,
            classType: node.class_type,
            input: spec.name,
            kind: "enum-value",
            message: options.length === 0
              ? `"${String(value)}" is not available: this server has no values for ${spec.name}`
              : `"${String(value)}" is not available on this server`,
            allowed: options.slice(0, 50),
          });
        }
        continue;
      }
      if (spec.type === "INT" || spec.type === "FLOAT") {
        if (typeof value !== "number" || !Number.isFinite(value)) {
          issues.push({ nodeId, classType: node.class_type, input: spec.name, kind: "bad-type", message: `expects a number, got ${JSON.stringify(value)}` });
          continue;
        }
        if (spec.type === "INT" && !Number.isInteger(value)) {
          issues.push({ nodeId, classType: node.class_type, input: spec.name, kind: "bad-type", message: `expects an integer, got ${value}` });
        }
        if ((spec.min !== undefined && value < spec.min) || (spec.max !== undefined && value > spec.max)) {
          issues.push({ nodeId, classType: node.class_type, input: spec.name, kind: "out-of-range", message: `${value} is outside [${spec.min ?? "-inf"}, ${spec.max ?? "inf"}]` });
        }
        continue;
      }
      if (spec.type === "STRING" && typeof value !== "string") {
        issues.push({ nodeId, classType: node.class_type, input: spec.name, kind: "bad-type", message: `expects a string, got ${typeof value}` });
      }
      if (spec.type === "BOOLEAN" && typeof value !== "boolean") {
        issues.push({ nodeId, classType: node.class_type, input: spec.name, kind: "bad-type", message: `expects a boolean, got ${typeof value}` });
      }
    }
  }
  if (Object.keys(graph).length > 0 && !hasOutputNode && issues.every((i) => i.kind !== "unknown-class")) {
    issues.push({ nodeId: "*", kind: "no-output-node", message: "graph has no output node (for example SaveImage), so ComfyUI would run nothing" });
  }
  return issues;
}

export function assertValidGraph(graph: ApiWorkflow, info: ObjectInfo): void {
  const issues = validateGraph(graph, info);
  if (issues.length > 0) throw new GraphValidationError(issues);
}

/**
 * Image generation stays on the local machine: refuse any graph that uses a
 * partner API node. Runs even when full validation is skipped.
 */
export function assertLocalOnly(graph: ApiWorkflow, info: ObjectInfo): void {
  const issues = validateGraph(graph, info).filter((i) => i.kind === "cloud-node");
  if (issues.length > 0) throw new GraphValidationError(issues);
}

const AUTOGROW = "COMFY_AUTOGROW_V3";

/**
 * Autogrow inputs (for example `images` on TextEncodeMingImageEdit) expand to
 * dotted sockets in API format: `images.image_1`, `images.image_2`, ... The
 * template names the allowed suffixes, the socket type and the minimum count.
 */
function checkAutogrow(
  graph: ApiWorkflow,
  info: ObjectInfo,
  nodeId: string,
  classType: string,
  inputs: Record<string, unknown>,
  name: string,
  def: ObjectInfo[string],
  issues: GraphIssue[],
): void {
  const raw = def.input.required?.[name] ?? def.input.optional?.[name];
  const template = ((raw?.[1] ?? {}) as { template?: { input?: Record<string, Record<string, unknown[]>>; names?: string[]; prefix?: string; min?: number; max?: number } }).template ?? {};
  const first = Object.values(template.input ?? {}).flatMap((group) => Object.values(group ?? {}))[0];
  const socketType = String(first?.[0] ?? "*");
  const allowed = template.names ?? (template.prefix ? Array.from({ length: template.max ?? 10 }, (_, i) => `${template.prefix}${i}`) : []);
  const keys = Object.keys(inputs).filter((k) => k.startsWith(`${name}.`));
  for (const key of keys) {
    const suffix = key.slice(name.length + 1);
    if (allowed.length && !allowed.includes(suffix)) {
      issues.push({ nodeId, classType, input: key, kind: "bad-link", message: `"${key}" is not one of ${name}.{${allowed.join(",")}}` });
      continue;
    }
    const value = inputs[key];
    if (!isLink(value)) {
      issues.push({ nodeId, classType, input: key, kind: "bad-type", message: `expects a ${socketType} link, got an inline value` });
      continue;
    }
    const src = graph[String(value[0])];
    if (!src) {
      issues.push({ nodeId, classType, input: key, kind: "bad-link", message: `links to node "${String(value[0])}", which is not in the graph` });
      continue;
    }
    const outType = info[src.class_type]?.output[value[1]];
    if (outType !== undefined && !typesCompatible(outType, socketType)) {
      issues.push({ nodeId, classType, input: key, kind: "link-type", message: `expects ${socketType} but is linked to ${src.class_type} output ${value[1]} (${outType})` });
    }
  }
  if (keys.length < (template.min ?? 0)) {
    issues.push({ nodeId, classType, input: name, kind: "missing-input", message: `needs at least ${template.min} ${name} input(s)` });
  }
}

function typesCompatible(output: string, input: string): boolean {
  if (output === "*" || input === "*") return true;
  if (output === input) return true;
  // ComfyUI allows comma-separated unions on either side ("IMAGE,MASK").
  const outs = output.split(",").map((s) => s.trim());
  const ins = input.split(",").map((s) => s.trim());
  return outs.some((o) => ins.includes(o));
}
