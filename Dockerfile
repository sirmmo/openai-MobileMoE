# syntax=docker/dockerfile:1

# ---- default: CPU-only image ---------------------------------------------
# torch is pulled from the CPU wheel index before requirements.txt so the
# resolver does not drag in the CUDA build (several GB of NVIDIA libraries the
# image would never use). Multi-arch: the CPU index ships aarch64 wheels too.
FROM python:3.12-slim AS cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY openai_mobilemoe ./openai_mobilemoe
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --no-deps .

# Weights are pulled from HuggingFace on first start and cached here. Mount a
# volume to keep them across restarts. The MobileMoE repos are gated: accept
# the license on huggingface.co once and pass HF_TOKEN at run time.
RUN mkdir -p /cache/huggingface
VOLUME ["/cache"]

EXPOSE 8000

ENV MOBILEMOE_HOST=0.0.0.0 \
    MOBILEMOE_PORT=8000

# First start downloads ~0.7GB (S-QAT) and dequantizes; allow for that.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); \
sys.exit(0 if r.get('status')=='ok' else 1)"

ENTRYPOINT ["python", "-m", "openai_mobilemoe"]


# ---- optional: CUDA image ---------------------------------------------------
# docker build --target cuda -t openai-mobilemoe:cuda .
# docker run --gpus all -p 8000:8000 -e HF_TOKEN=... openai-mobilemoe:cuda
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime AS cuda

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY openai_mobilemoe ./openai_mobilemoe
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --no-deps .

RUN mkdir -p /cache/huggingface
VOLUME ["/cache"]

EXPOSE 8000

ENV MOBILEMOE_HOST=0.0.0.0 \
    MOBILEMOE_PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); \
sys.exit(0 if r.get('status')=='ok' else 1)"

ENTRYPOINT ["python", "-m", "openai_mobilemoe"]
