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
ARG LLAMA_CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"

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

# Compile llama-cpp-python against CUDA for the target arch.
ENV CMAKE_ARGS="${LLAMA_CMAKE_ARGS}" \
    FORCE_CMAKE=1
COPY requirements.txt .
RUN pip install -r requirements.txt

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

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST} --port ${PORT}"]
