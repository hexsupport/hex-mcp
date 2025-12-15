FROM python:3.11-slim

ARG PORT=9000

WORKDIR /hex-mm-mcp

# Install uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv
# Copy the MCP server files
COPY . .

# Install dependencies into the image env
RUN uv pip install --system .

EXPOSE ${PORT}

CMD ["fastmcp", "run", "server/mm_mcp_server.py"]