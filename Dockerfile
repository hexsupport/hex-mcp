FROM python:3.11-slim

ARG PORT=9000
ENV PORT=${PORT}

WORKDIR /hex-mm-mcp

# Install uv
RUN pip install --no-cache-dir uv

# Copy only the project manifest first so dependency installation is cached
# separately from source changes — rebuilds only re-run this step when
# pyproject.toml changes, not on every code edit.
COPY pyproject.toml .

# Pre-install declared dependencies (project package itself excluded here)
RUN uv pip install --system --no-cache \
    fastmcp \
    python-dotenv \
    requests \
    loguru \
    ipython

# Copy the rest of the source and install the full package
COPY . .
RUN uv pip install --system --no-cache .

# Run as a non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /hex-mm-mcp
USER appuser

EXPOSE ${PORT}

CMD ["sh", "-c", "fastmcp run server/server_simple.py:mcp --no-banner --transport http --host 0.0.0.0 --port $PORT"]