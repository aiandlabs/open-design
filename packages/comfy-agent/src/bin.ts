// `comfy-agent-mcp`: stdio MCP server.
//
// Environment:
//   COMFYUI_URL                 ComfyUI base URL (default http://127.0.0.1:8188)
//   AIAND_API_KEY               ai& API key (needed for comfy_generate_asset)
//   AIAND_BASE_URL              ai& API base (default https://api.aiand.com/v1)
//   COMFY_AGENT_PLANNER_MODEL   planner model id (unset: picked from /v1/models)
//   COMFY_AGENT_CRITIC_MODEL    critic model id (unset: picked from /v1/models)
//   COMFY_AGENT_API_SHAPE       openai | anthropic
//   COMFY_AGENT_OUT_DIR         where images and traces are written

import { startMcpServer } from "./mcp.js";

startMcpServer().catch((err: unknown) => {
  process.stderr.write(`comfy-agent-mcp: ${err instanceof Error ? err.stack ?? err.message : String(err)}\n`);
  process.exit(1);
});
