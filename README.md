## HexagonML ModelManager MCP Server

This is a HexagonML MCP server that provides a Model context protocol interface for HexagonML ModelManager tools.

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