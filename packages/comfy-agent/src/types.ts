// Wire shapes for the ComfyUI HTTP API. Only what this package reads is
// modelled; everything else passes through as `unknown`.

/** A link from another node's output: `[sourceNodeId, outputIndex]`. */
export type ApiLink = [string, number];

export type ApiNode = {
  class_type: string;
  inputs: Record<string, unknown>;
  _meta?: { title?: string };
};

/** API-format workflow ("prompt"): node id → node. This is what `/prompt` accepts. */
export type ApiWorkflow = Record<string, ApiNode>;

/** One input spec as `/object_info` reports it: `[type, options?]`. */
export type ObjectInfoInputSpec = [unknown, Record<string, unknown>?] | [unknown];

export type ObjectInfoNode = {
  input: {
    required?: Record<string, ObjectInfoInputSpec>;
    optional?: Record<string, ObjectInfoInputSpec>;
    hidden?: Record<string, unknown>;
  };
  input_order?: { required?: string[]; optional?: string[] };
  output: string[];
  output_is_list?: boolean[];
  output_name?: string[];
  name: string;
  display_name?: string;
  category?: string;
  output_node?: boolean;
  /** True for ComfyUI partner/API nodes, which call a cloud service. */
  api_node?: boolean;
};

export type ObjectInfo = Record<string, ObjectInfoNode>;

export type SystemStats = {
  system: { os?: string; comfyui_version?: string; python_version?: string; [k: string]: unknown };
  devices: { name?: string; type?: string; vram_total?: number; vram_free?: number; [k: string]: unknown }[];
};

export type ImageRef = { filename: string; subfolder: string; type: string };

export type HistoryEntry = {
  prompt?: unknown;
  outputs: Record<string, { images?: (ImageRef & { metadata?: Record<string, unknown> })[]; [k: string]: unknown }>;
  status?: {
    status_str?: string;
    completed?: boolean;
    messages?: [string, Record<string, unknown>][];
  };
};

export type OutputImage = ImageRef & { nodeId: string; bytes: Buffer };

export type RunResult = {
  promptId: string;
  images: OutputImage[];
  durationMs: number;
  /** How completion was observed. */
  via: "websocket" | "polling";
};

export type ProgressEvent =
  | { type: "queued"; promptId: string; number?: number }
  | { type: "executing"; promptId: string; nodeId: string }
  | { type: "progress"; promptId: string; nodeId?: string; value: number; max: number }
  | { type: "cached"; promptId: string; nodeIds: string[] }
  | { type: "done"; promptId: string };
