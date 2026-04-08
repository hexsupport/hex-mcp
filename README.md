# HexagonML ModelManager MCP Server

A modular MCP server for interacting with the HexagonML ModelManager API, with full tool discoverability and clean architecture.

## Architecture

```
server/
├── config.py                  # Environment configuration and server setup
├── clients.py                 # ModelManager API client factory and context
├── utils.py                   # Common utilities (response handling, validation)
├── validators.py              # Payload validation functions
├── base.py                    # Base classes and common patterns
├── model_tools.py             # Model management tools
├── usecase_tools.py           # Usecase management tools
├── modelcard_tools.py         # Model card management tools
├── forecasting_tools.py       # Forecasting tools
├── forecasting_prompts.py     # MCP prompt templates for forecast presentation
├── main.py                    # Main entry point (standalone mode)
└── server_simple.py           # FastMCP CLI entry point
```

## Available Tools

### Model Management
| Tool | Description |
|---|---|
| `add_model` | Upload a new ML model |
| `update_model` | Update model metadata or configuration |
| `delete_model` | Permanently delete a model |
| `get_latest_metrics` | Retrieve model performance metrics |

### Usecase Management
| Tool | Description |
|---|---|
| `add_usecase` | Create a new usecase (supports Forecasting type) |
| `update_usecase` | Update usecase configuration |
| `delete_usecase` | Permanently delete a usecase |
| `get_usecase_data` | List all usecases |

### Model Cards
| Tool | Description |
|---|---|
| `create_modelcard` | Generate a model card for a model |
| `get_modelcard_data` | Retrieve model card data with optional filtering |

### Forecasting
| Tool | Description |
|---|---|
| `get_forecast` | Retrieve forecasts for a usecase |

### MCP Prompts
| Prompt | Description |
|---|---|
| `forecast_presentation_guide` | Instructs the LLM how to narrate forecast results to users |

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `SECRET_KEY` | Yes | — | API secret key for ModelManager authentication |
| `MM_API_BASE_URL` | Yes | — | Base URL of the ModelManager API |
| `OUTPUT_DIR` | Yes | — | Directory for generated output files |
| `HOST` | No | `0.0.0.0` | Network interface the server listens on |
| `PORT` | No | `9000` | Port the server listens on |

> **Docker note:** `MM_API_BASE_URL=http://127.0.0.1:8000` will not resolve from inside a container. Use `http://host.docker.internal:8000` on Mac/Windows, or `http://172.17.0.1:8000` on Linux.

## Running the Server

### Option 1: FastMCP CLI (recommended)
```bash
fastmcp run server/server_simple.py --transport http --host 0.0.0.0 --port 9000
```

### Option 2: Standalone Python
```bash
python server/main.py
```

### Option 3: Development / Inspector
```bash
fastmcp dev server/server_simple.py
```

### Option 4: Docker Compose
```bash
# Build and start
docker compose up --build

# Run in background
docker compose up --build -d

# Stop
docker compose down
```

### Option 5: Docker (manual)
```bash
# Build
docker build -t hex-mm-mcp:latest .

# Run
docker run -d --name hex-mm-mcp \
  -p 9000:9000 \
  --env-file .env \
  hex-mm-mcp:latest
```

## MCP Client Configuration

### Local (stdio / HTTP)
```json
{
  "mcpServers": {
    "hex-mm-mcp": {
      "command": "fastmcp",
      "args": [
        "run",
        "/path/to/hex-mm-mcp/server/server_simple.py",
        "--transport", "http",
        "--host", "127.0.0.1",
        "--port", "9000"
      ]
    }
  }
}
```

### Docker
```json
{
  "mcpServers": {
    "hex-mm-mcp": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i", "--network=host",
        "-e", "SECRET_KEY",
        "-e", "MM_API_BASE_URL",
        "-e", "OUTPUT_DIR",
        "hex-mm-mcp:latest"
      ],
      "env": {
        "SECRET_KEY": "your-secret-key",
        "MM_API_BASE_URL": "http://host.docker.internal:8000",
        "OUTPUT_DIR": "/tmp/mm-output"
      }
    }
  }
}
```

## LLM Presentation Guidance

The `get_forecast` tool uses two complementary mechanisms to guide how the LLM presents forecast data to users.

### Embedded `_llm_instructions`

Every `get_forecast` response includes a `_llm_instructions` field. The LLM reads this automatically as part of the tool result and follows the rules when composing its reply — no client configuration needed.

Key rules embedded in every response:
- Lead with the summary (date range, value range min/max/avg)
- Explain `lower_bound`/`upper_bound` as a confidence interval
- Reference `last_actual_update` to indicate data freshness
- Flag `resolved: false` as an unresolvable request
- Avoid echoing raw JSON back to the user

### MCP Prompt (`forecast_presentation_guide`)

A named MCP prompt is also registered on the server. MCP-aware clients (Claude Desktop, MCP Inspector) can fetch it and inject it as a system prompt at the start of a conversation, priming the LLM for the whole session rather than per-call.

To use it in a client that supports MCP prompts:
```
prompts/get  →  forecast_presentation_guide
```

| Mechanism | When it fires | Client support needed |
|---|---|---|
| `_llm_instructions` | Every `get_forecast` call | None — automatic |
| `forecast_presentation_guide` | On explicit prompt request | MCP prompt-capable client |

## Development

### Adding a New Tool

1. Create a new module (e.g., `server/new_tools.py`)
2. Import `mcp` from `config` and use the `@mcp.tool` decorator
3. Import the module in `server/server_simple.py`

```python
# server/new_tools.py
from fastmcp import Context
from config import mcp
from clients import get_mm_client
from utils import safe_response_to_dict, create_error_response
import asyncio

@mcp.tool(
    name="new_tool",
    description="Description of the tool",
    tags={"category", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"},
)
async def new_tool(ctx: Context, required_param: str, optional_param: str = None) -> dict:
    """Tool docstring shown in MCP playgrounds."""
    if not required_param or not required_param.strip():
        return create_error_response(
            message="required_param is required",
            error_type="ValidationError"
        )

    await ctx.info("Starting operation")
    await ctx.report_progress(progress=20, total=100)

    try:
        client = get_mm_client(ctx, 'client_type')
        await ctx.report_progress(progress=40, total=100)

        response = await asyncio.to_thread(client.api_method, required_param)
        await ctx.report_progress(progress=80, total=100)

        result = safe_response_to_dict(response)
        await ctx.info("Operation completed")
        await ctx.report_progress(progress=100, total=100)
        return result

    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message="A validation error occurred. Please check your input.",
            error_type="ValidationError"
        )
    except Exception as e:
        await ctx.error(f"Operation failed: {str(e)}")
        return create_error_response(
            message="An internal error occurred. Please try again.",
            error_type="InternalError"
        )
```

## Troubleshooting

**Port already in use**
```bash
kill -9 $(lsof -t -i:9000)
# or use a different port
fastmcp run server/server_simple.py --port 9001
```

**Missing environment variables**  
The server exits at startup with a message listing which of `SECRET_KEY`, `MM_API_BASE_URL`, and `OUTPUT_DIR` are missing.

**Connection refused / API errors**  
- Confirm the ModelManager API is reachable at `MM_API_BASE_URL`
- When running in Docker, see the Docker note in the Configuration section above
- Verify `SECRET_KEY` is correct and has the necessary permissions

## License

MIT License — see the LICENSE file for details.
