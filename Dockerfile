# Use CUDA base image if GPU is enabled, otherwise use slim
ARG USE_GPU=0
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS gpu-base
FROM python:3.10-slim AS cpu-base
FROM ${USE_GPU}="1" ? gpu-base : cpu-base

# Set environment variables
# - PYTHONUNBUFFERED ensures Python output is sent straight to terminal (helps with logging)
# - PYTHONDONTWRITEBYTECODE prevents Python from writing .pyc files
# - DEBIAN_FRONTEND eliminates interactive prompts during package installation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    USE_GPU=${USE_GPU}

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for building Python packages
# - build-essential: provides compilers and build tools
# - git: needed for some pip packages that install directly from GitHub
# - curl: use curl for running inference with POST call from CLI
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Copy requirements file first to leverage Docker's build cache
# (Docker will only re-run steps if files have changed)
COPY requirements.txt .

# Install dependencies based on GPU availability
RUN if [ "$USE_GPU" = "1" ]; then \
        pip install --no-cache-dir torch; \
    else \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install the rest of the requirements
# --no-cache-dir reduces image size by not caching the downloaded packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
# This is done after installing dependencies to leverage Docker's build cache
COPY . .

# Expose port 8000 to allow external connections to the FastAPI server
EXPOSE 8000

# Command to run when the container starts
# This launches the FastAPI server defined in server.py
CMD ["python", "server.py"] 