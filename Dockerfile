# AI Trading Dashboard Dockerfile
# Multi-stage build for optimized production image

# ==========================================
# Frontend Build Stage
# ==========================================
FROM node:18-alpine AS frontend-builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production --silent

# Copy source code
COPY src/ ./src/
COPY public/ ./public/
COPY index.html ./
COPY vite.config.js ./
COPY tailwind.config.js ./
COPY postcss.config.js ./

# Build frontend
RUN npm run build

# ==========================================
# Python Base Stage
# ==========================================
FROM python:3.11-slim AS python-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r trading && useradd -r -g trading trading

# ==========================================
# Production Stage
# ==========================================
FROM python-base AS production

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./backend/
COPY modules/ ./modules/
COPY envs/ ./envs/
COPY train/ ./train/
COPY utils/ ./utils/
COPY run.py ./
COPY run_dashboard.py ./

# Copy frontend build from frontend-builder stage
COPY --from=frontend-builder /app/dist ./frontend/dist

# Create necessary directories
RUN mkdir -p logs/{training,risk,simulation,strategy,position,tensorboard} \
    && mkdir -p checkpoints \
    && mkdir -p models \
    && mkdir -p data \
    && chown -R trading:trading /app

# Switch to non-root user
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Expose ports
EXPOSE 8000 6006

# Default command
CMD ["python", "backend/main.py"]

# ==========================================
# Development Stage
# ==========================================
FROM python-base AS development

WORKDIR /app

# Install Node.js for development
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy requirements and install Python dependencies (including dev dependencies)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dev dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    black \
    flake8 \
    mypy

# Copy all source code
COPY . .

# Install Node.js dependencies
RUN npm install

# Create necessary directories
RUN mkdir -p logs/{training,risk,simulation,strategy,position,tensorboard} \
    && mkdir -p checkpoints \
    && mkdir -p models \
    && mkdir -p data

# Expose ports for development
EXPOSE 8000 3000 6006

# Development command
CMD ["python", "run_dashboard.py", "--dev"]