# Stage 1: Build environment using the official uv image
FROM ghcr.io/astral-sh/uv:python3.10-alpine AS builder

# Set the working directory
WORKDIR /app

# Enable bytecode compilation for faster application startup
ENV UV_COMPILE_BYTECODE=1

# Copy only the files needed for dependency installation
COPY pyproject.toml uv.lock ./

# Synchronize dependencies (creates a virtual environment at /app/.venv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Stage 2: Final lightweight runtime image
FROM python:3.10-slim-bookworm

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of your application code
COPY . .

# Expose Streamlit's default communication port
EXPOSE 8501

# Configure Streamlit to run optimally inside a headless cloud container
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
