# Dockerfile for GCP Cloud Run Deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port (Cloud Run provides PORT env var)
EXPOSE 8080
ENV PORT=8080

# Health check (disabled for Cloud Run)
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application with gunicorn + eventlet for SocketIO
CMD exec gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 300 --log-level info app:app
