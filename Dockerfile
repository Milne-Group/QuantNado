FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[dev]"

WORKDIR /workspace
