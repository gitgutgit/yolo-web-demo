FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (OpenCV needs these)
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU version first (much smaller and faster)
RUN pip install --no-cache-dir --retries 5 --timeout 300 \
    torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir --retries 5 --timeout 300 -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Railway가 PORT 환경변수 제공
ENV PORT=8080

# Run with gunicorn + eventlet for SocketIO
CMD gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 300 --log-level info app:app