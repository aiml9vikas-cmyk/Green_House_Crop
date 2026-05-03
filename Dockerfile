# Use a slim Python image (3.11 is recommended for ML compatibility)
FROM python:3.11-slim-bookworm

# Install uv using the corrected ghcr.io path
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies (without the local project)
RUN uv sync --frozen --no-install-project

# Copy the rest of the application
COPY . .

# Install the project itself (the 'src' folder)
RUN uv sync --frozen

# Expose Streamlit port
EXPOSE 8501

# Run the application
# We use 'uv run' to ensure the locked environment is used
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
