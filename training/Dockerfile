# Multi-stage build for Lumi on RunPod with CUDA.
#
# Build: docker build -t lumi .
# Run:   docker run --gpus all lumi train

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
COPY Cargo.toml Cargo.lock build.rs ./
COPY src/ src/
COPY csrc/ csrc/

RUN cargo build --release --features cuda

# ---- Runtime stage: CUDA runtime only ----
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/lumi /usr/local/bin/lumi

WORKDIR /workspace
ENTRYPOINT ["lumi"]
