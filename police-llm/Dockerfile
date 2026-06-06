# police-llm: OpenAI-compatible llama.cpp server
#
# Builds llama-cpp-python from source with CUDA so it runs on the DGX Spark
# GB10 (Blackwell, compute capability sm_121, CUDA 13, aarch64).
#
# Build (DGX Spark / GB10, default):
#   docker build -t police-llm .
#
# Build for a different GPU (e.g. an RTX 4090 = sm_89):
#   docker build --build-arg CUDA_ARCH=89 -t police-llm .
#
# Build CPU-only (no GPU; for local testing):
#   docker build --build-arg LLAMA_CMAKE_ARGS="" -t police-llm .

ARG CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG CUDA_ARCH=121
# Override to "" for a CPU-only build.
# GGML_CUDA_NO_VMM=on  — skip linking libcuda.so (VMM driver API); fixes Docker
#   build linker errors (cuMemCreate etc.) when no host driver is present.
# LLAMA_BUILD_*=OFF    — we only need the Python bindings, not mtmd/cli tools.
ARG LLAMA_CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} -DGGML_CUDA_NO_VMM=on -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv python3-pip \
        git build-essential cmake ninja-build curl \
    && rm -rf /var/lib/apt/lists/*

# Isolate Python deps in a venv (Ubuntu 24.04 marks the system env externally-managed).
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# ---------------------------------------------------------------------------
# CUDA driver stubs — required to LINK ggml-cuda at Docker *build* time.
#
# The host NVIDIA driver (libcuda.so.1) is only injected at container *runtime*
# via the NVIDIA Container Toolkit. During `docker build` there is no driver,
# so the linker fails with undefined cuMemCreate / cuDeviceGet / etc.
#
# NVIDIA ships a stub libcuda in the CUDA toolkit; ggml-cuda looks for
# libcuda.so.1 specifically, so we symlink libcuda.so -> libcuda.so.1.
# On aarch64 (DGX Spark) stubs may live under targets/sbsa-linux/lib/stubs.
# ---------------------------------------------------------------------------
RUN CUDA_STUBS="" \
    && for d in /usr/local/cuda/lib64/stubs /usr/local/cuda/targets/sbsa-linux/lib/stubs; do \
         if [ -f "$d/libcuda.so" ]; then CUDA_STUBS="$d"; break; fi; \
       done \
    && if [ -z "$CUDA_STUBS" ]; then \
         echo "ERROR: libcuda.so stub not found under /usr/local/cuda"; \
         find /usr/local/cuda -name 'libcuda.so' 2>/dev/null || true; \
         exit 1; \
       fi \
    && ln -sf "$CUDA_STUBS/libcuda.so" "$CUDA_STUBS/libcuda.so.1" \
    && echo "Using CUDA stubs: $CUDA_STUBS"

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/targets/sbsa-linux/lib/stubs:${LIBRARY_PATH}

# Compile llama-cpp-python against CUDA for the target arch.
ENV CMAKE_ARGS="${LLAMA_CMAKE_ARGS}" \
    FORCE_CMAKE=1
COPY requirements.txt .
RUN if [ -n "${LLAMA_CMAKE_ARGS}" ]; then \
      echo "Building llama-cpp-python with CMAKE_ARGS=${CMAKE_ARGS}"; \
    else \
      echo "Building llama-cpp-python (CPU-only, no CMAKE_ARGS)"; \
    fi \
    && pip install -r requirements.txt

COPY app ./app

# Defaults; override at runtime (docker run -e ... / compose environment).
ENV MODEL_PATH=/models/model.gguf \
    MODEL_NAME=policeai \
    N_GPU_LAYERS=-1 \
    N_CTX=8192 \
    N_BATCH=512 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# 49B model load can take several minutes on first start.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=10 \
    CMD curl -fsS http://localhost:8000/ready || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST} --port ${PORT} --timeout-keep-alive 300"]
