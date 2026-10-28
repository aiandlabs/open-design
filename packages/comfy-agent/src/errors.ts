// Every failure mode the ComfyUI client and the asset loop can hit has its own
// class with a stable `code`. Callers branch on the class (or the code), never
// on message text. In particular nothing here is ever reported as a timeout
// unless the deadline really passed: a refused connection, a graph the server
// rejected, a node that raised, an empty output and a failed `/view` fetch are
// all different problems with different fixes.

export type ComfyErrorCode =
  | "COMFY_CONNECTION"
  | "COMFY_HTTP"
  | "COMFY_VALIDATION"
  | "COMFY_EXECUTION"
  | "COMFY_MISSING_OUTPUT"
  | "COMFY_OUTPUT_FETCH"
  | "COMFY_TIMEOUT"
  | "COMFY_INTERRUPTED"
  | "WORKFLOW_FORMAT"
  | "GRAPH_VALIDATION"
  | "GRAPH_EDIT"
  | "LLM_HTTP"
  | "LLM_FORMAT"
  | "LLM_CONFIG"
  | "LLM_MODEL"
  | "IMAGE_FORMAT";

export class ComfyAgentError extends Error {
  readonly code: ComfyErrorCode;
  constructor(code: ComfyErrorCode, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.code = code;
    this.name = new.target.name;
  }
  toJSON(): Record<string, unknown> {
    return { name: this.name, code: this.code, message: this.message };
  }
}

/** The server could not be reached at all (refused, DNS, reset before a response). */
export class ComfyConnectionError extends ComfyAgentError {
  readonly url: string;
  constructor(url: string, cause: unknown) {
    const reason = describeNetworkCause(cause);
    super("COMFY_CONNECTION", `cannot reach ComfyUI at ${url}: ${reason}`, { cause });
    this.url = url;
  }
  override toJSON() {
    return { ...super.toJSON(), url: this.url };
  }
}

/** A non-2xx response that has no more specific meaning. */
export class ComfyHttpError extends ComfyAgentError {
  readonly status: number;
  readonly url: string;
  readonly body: string;
  constructor(url: string, status: number, body: string) {
    super("COMFY_HTTP", `ComfyUI ${url} returned HTTP ${status}: ${body.slice(0, 300)}`);
    this.status = status;
    this.url = url;
    this.body = body;
  }
  override toJSON() {
    return { ...super.toJSON(), status: this.status, url: this.url };
  }
}

export type ComfyNodeError = {
  classType?: string;
  errors: { type?: string; message: string; details?: string; inputName?: string }[];
};

/** `POST /prompt` answered 400: the server refused the graph before queueing it. */
export class ComfyValidationError extends ComfyAgentError {
  readonly error: { type?: string; message: string; details?: string };
  readonly nodeErrors: Record<string, ComfyNodeError>;
  constructor(error: { type?: string; message: string; details?: string }, nodeErrors: Record<string, ComfyNodeError>) {
    const nodes = Object.entries(nodeErrors)
      .map(([id, e]) => `${id}${e.classType ? ` (${e.classType})` : ""}: ${e.errors.map((x) => x.details ? `${x.message} [${x.details}]` : x.message).join("; ")}`)
      .join(" | ");
    super("COMFY_VALIDATION", `ComfyUI rejected the workflow: ${error.message}${nodes ? ` — ${nodes}` : ""}`);
    this.error = error;
    this.nodeErrors = nodeErrors;
  }
  override toJSON() {
    return { ...super.toJSON(), error: this.error, nodeErrors: this.nodeErrors };
  }
}

/** A node raised while the prompt ran (websocket `execution_error` or history status). */
export class ComfyExecutionError extends ComfyAgentError {
  readonly promptId: string;
  readonly nodeId: string | undefined;
  readonly nodeType: string | undefined;
  readonly exceptionType: string | undefined;
  readonly exceptionMessage: string;
  constructor(input: { promptId: string; nodeId?: string; nodeType?: string; exceptionType?: string; exceptionMessage: string }) {
    super(
      "COMFY_EXECUTION",
      `ComfyUI execution failed at node ${input.nodeId ?? "?"}${input.nodeType ? ` (${input.nodeType})` : ""}: ${input.exceptionType ? `${input.exceptionType}: ` : ""}${input.exceptionMessage}`,
    );
    this.promptId = input.promptId;
    this.nodeId = input.nodeId;
    this.nodeType = input.nodeType;
    this.exceptionType = input.exceptionType;
    this.exceptionMessage = input.exceptionMessage;
  }
  override toJSON() {
    return {
      ...super.toJSON(),
      promptId: this.promptId,
      nodeId: this.nodeId,
      nodeType: this.nodeType,
      exceptionType: this.exceptionType,
    };
  }
}

/** The prompt finished successfully but produced no image the caller can use. */
export class ComfyMissingOutputError extends ComfyAgentError {
  readonly promptId: string;
  constructor(promptId: string, detail: string) {
    super("COMFY_MISSING_OUTPUT", `ComfyUI prompt ${promptId} completed without an image output: ${detail}`);
    this.promptId = promptId;
  }
}

/** The prompt produced an image but `GET /view` could not return it. */
export class ComfyOutputFetchError extends ComfyAgentError {
  readonly filename: string;
  readonly status: number | undefined;
  constructor(filename: string, status: number | undefined, detail: string, cause?: unknown) {
    super("COMFY_OUTPUT_FETCH", `could not fetch ComfyUI output ${filename}${status ? ` (HTTP ${status})` : ""}: ${detail}`, { cause });
    this.filename = filename;
    this.status = status;
  }
}

/**
 * A deadline passed. `phase` says which: `queue` when the server never
 * started the prompt (other prompts ahead of it in a shared queue), `run`
 * when it started but did not finish in time.
 */
export class ComfyTimeoutError extends ComfyAgentError {
  readonly promptId: string;
  readonly timeoutMs: number;
  readonly lastStatus: string;
  readonly phase: "queue" | "run";
  constructor(promptId: string, timeoutMs: number, lastStatus: string, phase: "queue" | "run" = "run") {
    super(
      "COMFY_TIMEOUT",
      phase === "queue"
        ? `ComfyUI prompt ${promptId} was still waiting in the queue after ${timeoutMs} ms`
        : `ComfyUI prompt ${promptId} did not finish within ${timeoutMs} ms of starting (last status: ${lastStatus})`,
    );
    this.promptId = promptId;
    this.timeoutMs = timeoutMs;
    this.lastStatus = lastStatus;
    this.phase = phase;
  }
}

/** The prompt was interrupted (by the caller's abort signal or by another client). */
export class ComfyInterruptedError extends ComfyAgentError {
  readonly promptId: string;
  constructor(promptId: string, detail: string) {
    super("COMFY_INTERRUPTED", `ComfyUI prompt ${promptId} was interrupted: ${detail}`);
    this.promptId = promptId;
  }
}

/** Workflow JSON could not be read or converted (unknown format, subgraphs, ...). */
export class WorkflowFormatError extends ComfyAgentError {
  constructor(message: string) {
    super("WORKFLOW_FORMAT", message);
  }
}

export type GraphIssue = {
  nodeId: string;
  classType?: string;
  input?: string;
  kind:
    | "unknown-class"
    | "cloud-node"
    | "missing-input"
    | "bad-link"
    | "link-type"
    | "enum-value"
    | "out-of-range"
    | "bad-type"
    | "no-output-node";
  message: string;
  allowed?: string[];
};

/** Our own pre-queue check against `/object_info` found problems. */
export class GraphValidationError extends ComfyAgentError {
  readonly issues: GraphIssue[];
  constructor(issues: GraphIssue[]) {
    super(
      "GRAPH_VALIDATION",
      `workflow does not match this ComfyUI server: ${issues.map((i) => `${i.nodeId}${i.classType ? `(${i.classType})` : ""}${i.input ? `.${i.input}` : ""}: ${i.message}`).join("; ")}`,
    );
    this.issues = issues;
  }
  override toJSON() {
    return { ...super.toJSON(), issues: this.issues };
  }
}

/** A typed edit could not be applied (unknown sampler, missing LoRA, no sampler node, ...). */
export class GraphEditError extends ComfyAgentError {
  readonly op: string;
  constructor(op: string, message: string) {
    super("GRAPH_EDIT", `edit ${op}: ${message}`);
    this.op = op;
  }
}

export class LlmHttpError extends ComfyAgentError {
  readonly status: number;
  readonly body: string;
  constructor(url: string, status: number, body: string) {
    super("LLM_HTTP", `ai& API ${url} returned HTTP ${status}: ${body.slice(0, 400)}`);
    this.status = status;
    this.body = body;
  }
}

export class LlmFormatError extends ComfyAgentError {
  readonly raw: string;
  constructor(message: string, raw: string) {
    super("LLM_FORMAT", message);
    this.raw = raw.slice(0, 2000);
  }
}

export class LlmConfigError extends ComfyAgentError {
  constructor(message: string) {
    super("LLM_CONFIG", message);
  }
}

export type ModelSelectionReason = "unknown" | "missing-capability" | "deprecated" | "none-capable";

/**
 * No usable model for a role: the configured id is not in the ai& catalog,
 * lacks a capability the role needs, is deprecated, or (with nothing
 * configured) no listed model has the capabilities. `choices` lists the ids
 * that would work.
 */
export class ModelSelectionError extends ComfyAgentError {
  readonly role: string;
  readonly model: string | undefined;
  readonly reason: ModelSelectionReason;
  readonly missing: string[];
  readonly required: string[];
  readonly choices: string[];
  readonly deprecationDate: string | undefined;
  constructor(input: {
    role: string;
    model?: string;
    reason: ModelSelectionReason;
    missing?: readonly string[];
    required: readonly string[];
    choices: readonly string[];
    deprecationDate?: string;
  }) {
    const need = input.required.join(", ");
    const valid = input.choices.length > 0 ? `valid ${input.role} models: ${input.choices.join(", ")}` : "the ai& catalog lists no valid model";
    const what =
      input.reason === "unknown" ? `${input.role} model "${input.model}" is not in the ai& model list`
        : input.reason === "missing-capability" ? `${input.role} model "${input.model}" lacks ${(input.missing ?? []).join(", ")} (the ${input.role} needs ${need})`
          : input.reason === "deprecated" ? `${input.role} model "${input.model}" is deprecated${input.deprecationDate ? ` since ${input.deprecationDate}` : ""}`
            : `no ai& model has ${need} for the ${input.role}`;
    super("LLM_MODEL", `${what}; ${valid}`);
    this.role = input.role;
    this.model = input.model;
    this.reason = input.reason;
    this.missing = [...(input.missing ?? [])];
    this.required = [...input.required];
    this.choices = [...input.choices];
    this.deprecationDate = input.deprecationDate;
  }
  override toJSON() {
    return {
      ...super.toJSON(),
      role: this.role,
      ...(this.model ? { model: this.model } : {}),
      reason: this.reason,
      ...(this.missing.length ? { missing: this.missing } : {}),
      required: this.required,
      choices: this.choices,
    };
  }
}

export class ImageFormatError extends ComfyAgentError {
  constructor(message: string, cause?: unknown) {
    super("IMAGE_FORMAT", message, { cause });
  }
}

function describeNetworkCause(cause: unknown): string {
  const chain: string[] = [];
  let cur: unknown = cause;
  for (let depth = 0; cur && depth < 4; depth += 1) {
    if (typeof cur === "object" && cur !== null) {
      const code = (cur as { code?: unknown }).code;
      const message = (cur as { message?: unknown }).message;
      if (typeof code === "string") chain.push(code);
      else if (typeof message === "string" && message) chain.push(message);
      cur = (cur as { cause?: unknown }).cause;
    } else {
      chain.push(String(cur));
      break;
    }
  }
  return chain.length > 0 ? Array.from(new Set(chain)).join(" / ") : "network error";
}

export function isConnectionRefused(err: unknown): boolean {
  let cur: unknown = err;
  for (let depth = 0; cur && depth < 4; depth += 1) {
    if (typeof cur === "object" && cur !== null && (cur as { code?: unknown }).code === "ECONNREFUSED") return true;
    cur = typeof cur === "object" && cur !== null ? (cur as { cause?: unknown }).cause : undefined;
  }
  return false;
}
