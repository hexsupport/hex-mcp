# HexagonML ModelManager MCP Architecture
## Client Discussion Outline

---

## 1. EXECUTIVE SUMMARY

### What is an MCP (Model Context Protocol)?
- Open standard for building secure, composable integrations between applications and AI
- Allows Claude (or other LLMs) to access specialized tools and data sources
- Clean separation of concerns: LLM makes decisions, MCP server provides capabilities

### Project Overview: HexagonML ModelManager MCP Server
- **Purpose**: Enable Claude and other AI clients to interact with HexagonML's ModelManager API
- **Status**: Production-ready, fully documented, deployable via FastMCP, Docker, or standalone
- **Value Proposition**: Extends Claude's capabilities with model management, forecasting, and model card generation

---

## 2. ARCHITECTURE OVERVIEW

### High-Level Design

```
┌─────────────────────────────────────┐
│  Claude Desktop / MCP-Aware Clients │
│  (User requests AI assistance)      │
└──────────────┬──────────────────────┘
               │ MCP Protocol (stdio/HTTP)
               ▼
┌─────────────────────────────────────┐
│  MCP Server (hex-mm-mcp)            │
│  ├─ config.py (setup & validation)  │
│  ├─ clients.py (API client factory) │
│  ├─ base.py (base classes)          │
│  └─ [tool modules] (see section 3)  │
└──────────────┬──────────────────────┘
               │ REST/HTTPS
               ▼
┌─────────────────────────────────────┐
│  HexagonML ModelManager API         │
│  (Models, Usecases, Forecasts)      │
└─────────────────────────────────────┘
```

### Key Architectural Principles

1. **Modular Tool Organization**: Each domain (models, usecases, forecasts, etc.) has its own module
2. **Centralized Configuration**: Single source of truth for env vars, API credentials, output settings
3. **Client Factory Pattern**: `clients.py` creates authenticated API clients on demand
4. **Async/Await First**: All tools use async patterns for non-blocking I/O
5. **Structured Error Handling**: Consistent error responses with type codes
6. **LLM-Aware Responses**: Built-in `_llm_instructions` guide how Claude presents data to users
7. **Progress Reporting**: Tools report progress via MCP context for long operations

---

## 3. TOOL CATEGORIES & CAPABILITIES

### Category 1: Model Management
**Module**: `model_tools.py` (14 KB)

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `add_model` | Model file, metadata (name, version, tags) | Model ID, creation timestamp | Onboard new ML models into the platform |
| `update_model` | Model ID, new metadata | Confirmation + updated fields | Modify model descriptions, tags, or configuration |
| `delete_model` | Model ID | Confirmation | Remove outdated or deprecated models |
| `get_latest_metrics` | Model ID (optional filters) | Performance metrics: accuracy, precision, recall, F1 | Monitor model performance over time |

**Typical Client Request**: "Show me the accuracy of our top-performing model and update its documentation."

---

### Category 2: Usecase Management
**Module**: `usecase_tools.py` (16 KB)

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `add_usecase` | Usecase name, type (Forecasting, Classification, etc.), model ID | Usecase ID | Create new ML applications for business problems |
| `update_usecase` | Usecase ID, new config | Updated usecase data | Modify thresholds, retrain frequency, or model selection |
| `delete_usecase` | Usecase ID | Confirmation | Retire completed projects or experiments |
| `get_usecase_data` | (optional filters) | List of all usecases with metadata | Inventory and track all active ML applications |

**Typical Client Request**: "Create a forecasting usecase for our supply chain, then show me all active usecases."

---

### Category 3: Model Cards
**Module**: `modelcard_tools.py` (11 KB)

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `create_modelcard` | Model ID, optional: sections to include, format | Generated model card (Markdown/JSON) | Generate documentation for model accountability & transparency |
| `get_modelcard_data` | Model ID (optional filters: section names) | Structured model card data with metadata | Retrieve existing model cards for auditing or compliance |

**Typical Client Request**: "Create a model card for our risk prediction model that includes fairness metrics."

**Why Model Cards Matter**:
- Regulatory compliance (Model Governance, MLOps best practices)
- Stakeholder transparency (what does this model do? what are its limitations?)
- Onboarding new team members (living documentation)

---

### Category 4: Forecasting
**Module**: `forecasting_tools.py` (5.9 KB)

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `get_forecast` | Usecase ID, optional: date range, confidence intervals | Forecast values + confidence bounds + metadata + `_llm_instructions` | Retrieve predictions and ask Claude to interpret them |

**Special Feature**: Every forecast response includes `_llm_instructions` that automatically guide Claude's narrative:
- Start with summary (date range, value range, min/max/avg)
- Explain confidence intervals as upper/lower bounds
- Reference data freshness (`last_actual_update`)
- Flag unresolvable requests (`resolved: false`)
- Avoid echoing raw JSON

**Typical Client Request**: "What's our forecast for Q2 revenue? Include confidence intervals and explain any trends you see."

---

### Category 5: MCP Prompts (System-Level Guidance)
**Module**: `forecasting_prompts.py` (1.3 KB)

| Prompt | Purpose |
|--------|---------|
| `forecast_presentation_guide` | System prompt injected at conversation start (MCP-aware clients only). Primes Claude for the entire session on how to narrate forecast data |

**When it activates**:
- `_llm_instructions` (in tool response): fires on every `get_forecast` call—automatic, no client config needed
- `forecast_presentation_guide` (MCP prompt): client must explicitly request `prompts/get` — more powerful for multi-turn conversations

---

## 4. CORE INFRASTRUCTURE

### Configuration Layer (`config.py`)
- **Validates** required env vars at startup: `SECRET_KEY`, `MM_API_BASE_URL`, `OUTPUT_DIR`
- **Reports** config status without exposing secrets (for debugging)
- **Initializes** FastMCP server instance (shared across all tool modules)

### Client Factory (`clients.py`)
- **Creates authenticated API clients** on-demand (context-aware)
- **Caches** clients during a request session
- **Handles** retry logic and timeout management
- **Supports** multiple client types (future extensibility)

### Base Classes (`base.py`)
- **Standardized tool signatures** for consistency
- **Common validation patterns** (required fields, format checking)
- **Error response templates** (JSON structure, status codes)
- **Async utilities** for concurrent operations

### Utilities (`utils.py`)
- **Response parsing**: Convert API responses to Python dicts
- **Error mapping**: Translate API errors to user-friendly messages
- **File handling**: Save outputs to `OUTPUT_DIR`
- **Logging**: Structured logging for debugging

### Response Handlers (`response_handlers.py`)
- **Rich response formatting** for forecasts and model cards
- **Data transformation** pipelines (API → LLM-ready format)
- **Validation** of response structure before returning to client

### Validators (`validators.py`)
- **Payload validation** before API calls
- **Type checking** (required fields, optional fields)
- **Business logic validation** (e.g., usecase type must match model type)

---

## 5. DEPLOYMENT OPTIONS

### Option 1: FastMCP CLI (Recommended for Production)
```bash
fastmcp run server/server_simple.py --transport http --host 0.0.0.0 --port 9000
```
- **Pros**: Single command, auto-restart on crash, minimal config
- **Cons**: Requires FastMCP installation
- **Best for**: Production servers, CI/CD pipelines

### Option 2: Docker Compose (Recommended for Teams)
```bash
docker compose up --build
```
- **Pros**: Isolated environment, reproducible across machines, volume mounts for persistence
- **Cons**: Requires Docker, slightly slower startup
- **Best for**: Shared development, staging/QA environments

### Option 3: Docker Manual (Maximum Control)
```bash
docker build -t hex-mm-mcp:latest .
docker run -d --name hex-mm-mcp -p 9000:9000 --env-file .env hex-mm-mcp:latest
```
- **Pros**: Fine-grained control over networking, volumes, env vars
- **Cons**: Manual startup/stop, no auto-restart without additional tooling
- **Best for**: Custom Kubernetes deployments, complex networking

### Option 4: Standalone Python (Development)
```bash
python server/main.py
```
- **Pros**: Direct inspection, easy debugging with IDE
- **Cons**: No auto-restart, fragile for production
- **Best for**: Local development, testing

### Option 5: FastMCP Dev Inspector (Debugging)
```bash
fastmcp dev server/server_simple.py
```
- **Pros**: Interactive playground, live tool testing, schema validation
- **Cons**: Not for production
- **Best for**: Testing new tools, API exploration

---

## 6. CLIENT INTEGRATION

### For Claude Desktop Users

Add to `~/.config/claude/claude.json` (Mac) or `%APPDATA%\Claude\claude.json` (Windows):

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

### For Custom LLM Applications

Use any MCP-compatible client library:
- **Python**: `anthropic-sdk` with MCP support
- **Node.js**: `@modelcontextprotocol/sdk`
- **HTTP direct**: POST to `localhost:9000/` with MCP protocol messages

---

## 7. SECURITY & GOVERNANCE

### Authentication
- **API Key Required**: `SECRET_KEY` env var (passed to ModelManager API)
- **No direct user auth**: Server trusts whoever can reach the MCP endpoint (network-level security)
- **Recommended**: Run behind VPN or SSH tunnel, restrict firewall rules

### Data Handling
- **Output Files**: Saved to `OUTPUT_DIR` (configurable, can be mounted to S3/NFS)
- **Secrets**: Never logged; only shown as `*****` in status reports
- **Audit Trail**: Server logs all tool calls (timestamp, tool name, parameters, result)

### Compliance
- **Model Card Generation**: Supports documentation for regulatory frameworks (GDPR, FDA, SOX)
- **No data retention**: Forecasts and model cards computed on-demand, not stored by server
- **Extensible**: Can add rate limiting, request signing, or audit webhooks

---

## 8. DEVELOPMENT & EXTENSIBILITY

### Adding a New Tool (3-Step Process)

**Step 1: Create module** (`server/new_feature_tools.py`)
```python
from fastmcp import Context
from config import mcp
from clients import get_mm_client
from utils import safe_response_to_dict, create_error_response

@mcp.tool(
    name="my_new_tool",
    description="What this tool does",
    tags={"category", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"},
)
async def my_new_tool(ctx: Context, required_param: str, optional_param: str = None) -> dict:
    """Full docstring visible in MCP playgrounds."""
    # Validation
    if not required_param or not required_param.strip():
        return create_error_response(message="required_param is required", error_type="ValidationError")
    
    # Implementation
    await ctx.info("Starting operation")
    await ctx.report_progress(progress=20, total=100)
    
    try:
        client = get_mm_client(ctx, 'client_type')
        await ctx.report_progress(progress=40, total=100)
        
        response = await asyncio.to_thread(client.api_method, required_param)
        result = safe_response_to_dict(response)
        
        await ctx.report_progress(progress=100, total=100)
        return result
    except ValueError as e:
        return create_error_response(message=f"Validation error: {str(e)}", error_type="ValidationError")
    except Exception as e:
        return create_error_response(message="An internal error occurred", error_type="InternalError")
```

**Step 2: Import in** `server/server_simple.py`
```python
import server.new_feature_tools  # noqa: F401
```

**Step 3: Restart server** — FastMCP auto-discovers new tools

### Code Organization Best Practices
- **One tool module per domain** (models, usecases, forecasts, etc.)
- **Reuse base classes** from `base.py` for consistency
- **Use client factory** (`get_mm_client`) to avoid credential duplication
- **Test with FastMCP inspector** before deploying
- **Document with docstrings** — they appear in tool schemas

---

## 9. TESTING & QUALITY ASSURANCE

### Unit Tests
```bash
pytest tests/ -v
```
- Test each tool's validation logic
- Mock API responses
- Verify error handling paths

### Integration Tests
- Test against a staging ModelManager API
- Verify end-to-end tool flows
- Check response format compliance

### Manual Testing
```bash
fastmcp dev server/server_simple.py
```
- Interactive playground in the browser
- Test each tool with real parameters
- Inspect full request/response payloads

---

## 10. OPERATIONAL CONSIDERATIONS

### Monitoring
- **Metrics to track**:
  - Request latency (p50, p95, p99)
  - Error rates by tool
  - API response times
  - Tool usage frequency
  
### Logging
- **Structure**: JSON logs with timestamp, tool name, parameters (sanitized), duration, status
- **Aggregation**: Forward to ELK, Datadog, or CloudWatch for analysis

### Common Troubleshooting

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| Port 9000 already in use | Another process claims the port | `kill -9 $(lsof -t -i:9000)` or use `--port 9001` |
| "Connection refused" to ModelManager | Wrong `MM_API_BASE_URL` or API down | Check Docker bridge networking (use `host.docker.internal` on Mac) |
| Missing env var errors | `.env` file not loaded or incomplete | Copy `.env.example`, fill in values, verify with `echo $SECRET_KEY` |
| Tool not appearing in Claude | Config out of sync or tool import failed | Restart server, check `fastmcp dev` playground for new tools |
| Slow forecasts | API latency or large dataset | Add request timeout config, check ModelManager performance |

---

## 11. ROADMAP & FUTURE ENHANCEMENTS

### Short Term (Q2 2026)
- [ ] Add batch forecasting tool (get forecasts for multiple usecases in one call)
- [ ] Implement tool-level rate limiting
- [ ] Add webhook support for async notifications

### Medium Term (Q3 2026)
- [ ] Multi-workspace support (segregate models/usecases by client org)
- [ ] Streaming response support for large model cards
- [ ] Advanced filtering/search on usecase and model listings

### Long Term (Q4 2026+)
- [ ] Model versioning with automatic rollback
- [ ] A/B testing framework integration
- [ ] Custom metric calculation engine
- [ ] LLM-guided model selection (Claude recommends best model for usecase)

---

## 12. KEY TALKING POINTS FOR CLIENT

### 1. **Extensibility Without Complexity**
- New tools added in minutes; existing tools unaffected
- Clear patterns borrowed from successful tools
- No monolithic codebase to maintain

### 2. **LLM-First Design**
- Responses include `_llm_instructions` to guide Claude's behavior
- System prompts primed for forecasting use cases
- Structured outputs (JSON) for reliable parsing

### 3. **Production-Ready Infrastructure**
- Multiple deployment options (FastMCP, Docker, Kubernetes-ready)
- Built-in error handling and validation
- Monitoring hooks for observability

### 4. **Security & Compliance**
- API key–based authentication
- No stored credentials or data
- Audit trail for regulatory requirements (SOX, HIPAA, GDPR)

### 5. **Developer Experience**
- Interactive testing playground (FastMCP inspector)
- Clear documentation and examples
- Standardized tool structure reduces onboarding time

### 6. **Cost Efficiency**
- On-demand computation (no standing resources)
- Optional Docker deployment (run on existing infrastructure)
- Minimal dependencies (FastMCP, Python 3.10+)

---

## 13. Q&A PREPARATION

### Expected Client Questions

**Q: How does this compare to direct API calls?**
> A: Direct API calls require the client to know all endpoints and auth details. MCP abstracts that away—Claude can discover tools without documentation, and we control the API surface. Also, MCP enables LLM-specific optimizations like `_llm_instructions`.

**Q: Can this scale to millions of requests?**
> A: The server is stateless, so horizontal scaling is straightforward. Add more instances behind a load balancer. Bottleneck is typically the backend ModelManager API, not the MCP layer.

**Q: What if the ModelManager API changes?**
> A: Tools are tightly coupled to API responses. We'd need to update tool modules. However, the base classes and patterns mean changes are localized—no full rewrite needed.

**Q: Is this secure for production?**
> A: Yes, with caveats: run behind VPN/SSH tunnel, use strong API keys, enable audit logging. MCP doesn't encrypt in transit (that's HTTPS's job), so assume TLS 1.3 minimum.

**Q: How do we monitor tool usage?**
> A: Log aggregation (ELK, Datadog). Each tool call is logged with duration and status. We can also add custom metrics (e.g., "forecast requests per hour").

**Q: What's the cost?**
> A: Server cost (your infrastructure) + ModelManager API cost (per your contract). MCP adds minimal overhead (<5% latency increase typically).

**Q: Can we white-label this for our customers?**
> A: Yes. Ship the Docker image to customers with their own credentials. They connect to their ModelManager instance. No code changes needed.

---

## APPENDIX: Quick Reference

### Environment Variables
```bash
SECRET_KEY=your-api-key-here
MM_API_BASE_URL=http://your-modelmanager:8000
OUTPUT_DIR=/tmp/mm-output
HOST=0.0.0.0
PORT=9000
```

### Directory Structure
```
server/
├── config.py              # Environment setup & FastMCP init
├── clients.py             # API client factory
├── base.py                # Base classes & patterns
├── utils.py               # Common utilities
├── validators.py          # Input validation
├── model_tools.py         # Model CRUD & metrics
├── usecase_tools.py       # Usecase CRUD & config
├── modelcard_tools.py     # Model card generation
├── forecasting_tools.py   # Forecast retrieval
├── forecasting_prompts.py # System prompt templates
├── response_handlers.py   # Response formatting
├── health.py              # Health check endpoint
├── server_simple.py       # FastMCP CLI entry point
└── main.py                # Standalone entry point
```

### Tool Invocation Examples (via Claude)
```
"Show me the latest metrics for model abc123"
→ Calls: get_latest_metrics(model_id="abc123")

"Create a forecasting usecase called 'Q2_Revenue' using our best model"
→ Calls: get_latest_metrics(), then add_usecase(name="Q2_Revenue", type="Forecasting", model_id=...)

"Generate a model card for our risk model and include fairness metrics"
→ Calls: create_modelcard(model_id=..., sections=["performance", "fairness", "limitations"])

"What's our forecast for next quarter? Explain the confidence intervals."
→ Calls: get_forecast(usecase_id=...) → Claude reads _llm_instructions → Narrates results
```

---

## Document Metadata
- **Created**: April 2026
- **Project**: HexagonML ModelManager MCP Server
- **Audience**: C-suite, Product Managers, Technical Stakeholders
- **Status**: Ready for client presentation
