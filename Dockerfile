# syntax=docker/dockerfile:1

# Use a slim Python image; install TensorFlow CPU via pip
FROM python:3.10-slim

# Avoid interactive prompts; ensure UTF-8
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TZ=UTC

# System deps required for Pillow, TensorFlow, and building wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libjpeg62-turbo \
       libpng16-16 \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender1 \
       git \
       curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt ./

# Install Python deps. TensorFlow CPU will be installed from requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application code
COPY . .

# Expose the port Render will route to
EXPOSE 8000

# Healthcheck (optional but helpful)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/ || exit 1

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


