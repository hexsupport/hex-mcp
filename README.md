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
        "modelmanagerdev/mcp:v1"
      ],
      "env": {
        "SECRET_KEY": "your-secret-key",
        "MM_API_BASE_URL": "your-api-base-url"
      }
    }
  }
}
```
