# Multi-stage build for Lumi on RunPod with CUDA.
#
# Build: docker build -f training/Dockerfile -t lumi . (from repo root)
# Run:   docker run --gpus all lumi train
#
# NOTE: this is the NVIDIA path, built from kernels/csrc_cuda/. The active local target
# is AMD/ROCm (kernels/csrc_hip/, --features hip); there is no ROCm Dockerfile yet.

# ---- Builder stage: CUDA toolkit + Rust ----
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04 AS builder

# Install Rust toolchain + build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY kernels/ kernels/
COPY training/ training/
# inference/ is a workspace member, so its manifest must exist for cargo to
# load the workspace at all. It is not built here (-p lumi), so its hip-only
# kernel dependency never activates in this CUDA image.
COPY inference/ inference/

RUN cargo build --release -p lumi --features cuda

# ---- Runtime stage: CUDA runtime only ----
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/lumi /usr/local/bin/lumi

WORKDIR /workspace
ENTRYPOINT ["lumi"]
