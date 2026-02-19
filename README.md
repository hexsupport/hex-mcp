## HexagonML ModelManager MCP Server

This is a HexagonML MCP server that provides a Model context protocol interface for HexagonML ModelManager tools.

### Local Development

#### Prerequisites

- **Python**: `python3`
- **Virtualenv**: recommended (project uses `.venv` in examples)

#### Environment variables

The server reads configuration from environment variables (and will also load a local `.env` file automatically).

- **`SECRET_KEY`**: secret key for ModelManager API auth
- **`MM_API_BASE_URL`**: base URL for ModelManager API (example: `http://localhost:8000`)
- **`OUTPUT_DIR`**: directory where generated HTML outputs are written
- **`HOST`** (optional): defaults to `0.0.0.0`
- **`PORT`** (optional): defaults to `9000`

Example `.env`:

```bash
SECRET_KEY=your-secret-key
MM_API_BASE_URL=http://localhost:8000
OUTPUT_DIR=./output
HOST=0.0.0.0
PORT=9000
```

#### Run the server

From the repo root:

```bash
python3 server/mm_mcp_server.py
```

To run with the FastMCP inspector (dev mode):

```bash
fastmcp dev server/mm_mcp_server.py
```

#### Troubleshooting

- **Port in use (FastMCP inspector)**: If you see `Proxy Server PORT IS IN USE` (commonly `6277`), stop the previous inspector process and retry.
- **Missing env vars**: The server will exit with a message listing missing required variables.

### Configuration For mcp integration on host (g: windsurf, vscode, claude desktop)

#### Local Configuration
```json
{
  "mcpServers": {
    "hex-mm-mcp": {
      "command": "hex-mm-mcp/.venv/bin/mcp",
      "args": ["run", "hex-mm-mcp/server/mm_mcp_server.py"]
    }
  }
}
```

#### Docker Configuration

##### For Dev (Using Local URL)
  - Run the ModelManager server in `--host 0.0.0.0 --port 8000`
    - cmd `python manage.py runserver 0.0.0.0:8000`
  - Get the hostname of your system using `hostname -I` command
    - eg: `192.168.10.75 172.17.0.1 2400:1a00:4b26:2af0:8f53:ede1:ec3a:c59b 2400:1a00:4b26:2af0:9139:c926:2fb5:6008`
    - use first ip address from the list eg: `192.168.10.75`
  - Replace `your-api-base-url` with `http://<hostIP>:8000`

```json
{
  "mcpServers": {
    "hex-mm-mcp-docker": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "--network=host",
        "-e",
        "SECRET_KEY",
        "-e",
        "MM_API_BASE_URL",
        "-v",
        "OUTPUT_DIR",
        "<image-name>:<tag>"
      ],
      "env": {
        "SECRET_KEY": "your-secret-key",
        "MM_API_BASE_URL": "your-api-base-url",
        "OUTPUT_DIR": "your-output-dir"
      }
    }
  }
}
```

```
Docker Commands

Build Image
docker build --platform=linux/amd64 -t modelmanagerdev/mcp:version_id .

Run Container
docker run --platform=linux/amd64  -d --name mm-mcp -p 9000:9000 --env-file .env modelmanagerdev/mcp:v6
```

### Model Insights Tools

The server exposes Model Insights endpoints via MCP tools.

#### Create Insight

- **Tool name**: `create_insight`
- **Input**: `data` (dict)
- **Backend**: `POST /api/mmanager-modelinsights/create_insight/`

#### Get Insights

- **Tool name**: `get_insight`
- **Input**: `usecase_id` (str)
- **Backend**: `GET /api/mmanager-modelinsights/get_insights/?usecase_id=...`
- **Response shape**:
  - If the backend returns a dict, the tool returns that dict.
  - If the backend returns a list, the tool returns `{ "data": [...] }`.

### Forecast Tools

#### Get Forecast

- **Tool name**: `get_forecast`
- **Input**: `payload` (dict)
- **Backend**: `POST /api/mmanager-forecast/get_forecast/`

### Run Command (Locally run MCP Server)
```bash
fastmcp run server/mm_mcp_server.py --transport http --host 127.0.0.1 --port 8080
```