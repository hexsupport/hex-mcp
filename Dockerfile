FROM python:3.11-slim

ARG PORT=9000

WORKDIR /hex-mm-mcp

# Install uv
RUN pip install uv

# Copy the MCP server files
COPY . .

# Create and activate venv, install dependencies from pyproject.toml
RUN python -m venv venv \
    && . venv/bin/activate \
    && uv pip install .

ENV PATH="/hex-mm-mcp/venv/bin:$PATH"
ENV VIRTUAL_ENV="/hex-mm-mcp/venv"

EXPOSE ${PORT}

# Command to run the MCP server
CMD ["mcp", "run", "server/mm_mcp_server.py"]