# YOLOZU LLM integrations (MCP-first)

This project standardizes LLM integrations around one backend implementation.

## 1) Common base (highest priority): YOLOZU MCP server

Start server:

```bash
python3 tools/run_mcp_server.py
```

Exposed tools (minimum):
- `doctor`
- `validate_predictions`
- `validate_dataset`
- `eval_coco`
- `run_scenarios`
- `convert_dataset` (optional but available)

Return format policy:
- Always machine-readable JSON with stable top-level keys:
  - `ok` (bool)
  - `tool` (string)
  - `summary` (short sentence)
  - `exit_code` (int)
  - `stdout` / `stderr` (string)
  - optional parsed JSON artifacts (e.g. `report_json`)

This format is designed so Claude/Copilot/other MCP-capable clients can summarize consistently.

## 2) OpenAI (ChatGPT) routes

### A. MCP route (recommended)

Use the same YOLOZU MCP server as remote MCP endpoint.

- Reuses one implementation across LLMs.
- Keeps command behavior and outputs identical to local CLI semantics.

### B. GPT Actions route (OpenAPI)

Start REST endpoint:

```bash
python3 tools/run_actions_api.py
```

OpenAPI schema:
- `http://<host>:8080/openapi.json`

Main endpoints:
- `POST /doctor`
- `POST /validate/predictions`
- `POST /validate/dataset`
- `POST /eval/coco`
- `POST /run/scenarios`
- `POST /convert/dataset`

Recommendation: ship MCP first, add Actions only when ChatGPT Actions integration is required.

## 3) Copilot routes

### A. Copilot Extensions (skillsets / agent)

Define skill endpoints that forward to:
- YOLOZU MCP server (preferred), or
- YOLOZU Actions API.

### B. VS Code extension route

Implement participant/commands that invoke the same backend (MCP/API) instead of re-implementing CLI logic.

This avoids duplicate business logic and keeps output parity between Copilot and other LLM clients.

## 4) Gemini route

Gemini can use the same backend in two ways:

### A. MCP route (recommended)

Connect Gemini-capable MCP client/runtime to YOLOZU MCP server:

```bash
python3 tools/run_mcp_server.py
```

Use the same core tools (`doctor`, `validate_predictions`, `validate_dataset`, `eval_coco`, `run_scenarios`, `convert_dataset`) with identical JSON outputs.

### B. API/tool-calling route

Expose the FastAPI/OpenAPI endpoint and register tool/function calls against it:

```bash
python3 tools/run_actions_api.py
```

Schema endpoint:
- `http://<host>:8080/openapi.json`

This keeps Gemini, OpenAI, and Copilot integrations aligned on one implementation.
