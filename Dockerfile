# syntax=docker/dockerfile:1
#
# Stages:
#   model - downloads the NER model (Davlan/xlm-roberta-base-ner-hrl) and the embedding model
#           (intfloat/multilingual-e5-base), each at a pinned revision with sha256-checked files
#   base  - Python 3.12 on Debian 13 (trixie) + Tesseract 5 with eng/spa/por/fra/deu + uv
#           + both models at /models (read-only; Hugging Face offline mode)
#   test  - project + dev dependencies; runs the full pytest suite (incl. `tesseract` tests)
# A `runtime` stage (no dev deps) is added later on top of `base`.
#
# Build and run the tests:
#   docker build --target test -t docintel-test .
#   docker run --rm docintel-test

# python:3.12-slim-trixie, pinned by digest (resolved 2026-10-27). Debian 13 ships Tesseract 5.5,
# the version the OCR defaults were benchmarked with; bump tag and digest together.
ARG PYTHON_IMAGE=python:3.12-slim-trixie@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f
# Embedding model commit (intfloat/multilingual-e5-base main, resolved 2026-10-27). Global so the
# `model` stage downloads it and `base` exposes it as EMBEDDING_MODEL_REVISION (model identity).
ARG E5_MODEL_REVISION=d128750597153bb5987e10b1c3493a34e5a4502a


# NER model, downloaded at build time so run time never touches the network. Only this stage has
# huggingface_hub; `base` receives the model files alone.
# To bump the revision: take the new commit "sha" from
#   https://huggingface.co/api/models/Davlan/xlm-roberta-base-ner-hrl
# and the "lfs.oid" (sha256) of the two LFS files from
#   https://huggingface.co/api/models/Davlan/xlm-roberta-base-ner-hrl/tree/<sha>
# The small files are plain git blobs (their "oid" there is a git sha1), so take their sha256 from
#   https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl/resolve/<sha>/<file>  (| sha256sum)
# Update the revision and every *_SHA256 ARG below, rebuild and rerun the suite (the `ner_model`
# tests use the model).
# huggingface_hub is pinned by version only: --require-hashes would need every transitive
# dependency hashed too, and the downloaded model files are hash-checked here anyway.
FROM ${PYTHON_IMAGE} AS model

ARG HF_HUB_VERSION=2.3.0
ARG NER_MODEL_REPO=Davlan/xlm-roberta-base-ner-hrl
# Commit of the Hub repository's main branch, resolved 2026-10-27.
ARG NER_MODEL_REVISION=253f557bd8249b8515114cfd7f71974fe5fa4d2f
ARG NER_MODEL_SAFETENSORS_SHA256=49530ba99295b5992caafa1f361a658206094d26c74bbe5248cd64a0c4270061
ARG NER_MODEL_SPM_SHA256=cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865
ARG NER_MODEL_CONFIG_SHA256=b39f5cdaa31793288077859feb3d4e742b2d5a1226fe3374e97fc0220c02222f
ARG NER_MODEL_TOKENIZER_CONFIG_SHA256=25784ba43a0c3aa2afaa08cd4e89795117a2c82e56f4fff50fb0b8dcff27b29f
ARG NER_MODEL_SPECIAL_TOKENS_SHA256=7638f5bbbe86ef6d604ef28ad3647dc690d6d117c81c0d63e885416be8da1150
# The model card (README.md), kept next to the model for its licence and attribution.
ARG NER_MODEL_CARD_SHA256=ded5d21f50ad65d500946602b18a0007b431971056f5542bdb7be6e6d3d03bf4

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir "huggingface_hub==${HF_HUB_VERSION}"

# Only what transformers loads: config, tokenizer files (the repository has no tokenizer.json;
# the fast tokenizer is converted from sentencepiece.bpe.model at load time) and the safetensors
# weights, plus the model card (README.md). pytorch_model.bin and training_args.bin are pickles
# and are never downloaded. Every file is checked against its pinned sha256, and a NOTICE with
# the licences and attribution is written next to them.
RUN python - <<'PY'
import hashlib
import os
import pathlib
import shutil

from huggingface_hub import snapshot_download

expected = {
    "README.md": "NER_MODEL_CARD_SHA256",
    "config.json": "NER_MODEL_CONFIG_SHA256",
    "model.safetensors": "NER_MODEL_SAFETENSORS_SHA256",
    "sentencepiece.bpe.model": "NER_MODEL_SPM_SHA256",
    "special_tokens_map.json": "NER_MODEL_SPECIAL_TOKENS_SHA256",
    "tokenizer_config.json": "NER_MODEL_TOKENIZER_CONFIG_SHA256",
}
files = sorted(expected)
target = pathlib.Path("/models/xlm-roberta-base-ner-hrl")
snapshot_download(
    os.environ["NER_MODEL_REPO"], revision=os.environ["NER_MODEL_REVISION"],
    allow_patterns=files, local_dir=target,
)
shutil.rmtree(target / ".cache", ignore_errors=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


present = sorted(p.name for p in target.iterdir())
assert present == files, f"unexpected model files: {present}"
for name, env in expected.items():
    actual = sha256(target / name)
    assert actual == os.environ[env], f"{name}: sha256 {actual} != {os.environ[env]}"
repo, revision = os.environ["NER_MODEL_REPO"], os.environ["NER_MODEL_REVISION"]
(target / "NOTICE").write_text(
    f"Model: {repo}\n"
    f"Source: https://huggingface.co/{repo} (revision {revision})\n"
    "Author: Davlan (Hugging Face), fine-tuned for named entity recognition.\n"
    "Licence: Academic Free License v3.0 (AFL-3.0), as declared in the model card README.md.\n"
    "Base model: FacebookAI/xlm-roberta-base (XLM-RoBERTa, Conneau et al., 2019), MIT License.\n"
    "The model files in this directory are redistributed unmodified.\n",
    encoding="utf-8",
)
for name in sorted(p.name for p in target.iterdir()):
    print(name, (target / name).stat().st_size)
PY

# Embedding model (intfloat/multilingual-e5-base), same procedure and bump instructions as the NER
# model above with E5_ in place of NER_ (API: .../api/models/intfloat/multilingual-e5-base).
# Only what AutoModel + AutoTokenizer load: config, safetensors weights, tokenizer.json (the fast
# tokenizer, loaded directly), sentencepiece.bpe.model and the two small tokenizer configs, plus
# the model card. pytorch_model.bin, the onnx/ and openvino/ variants and the sentence-transformers
# files (modules.json, 1_Pooling/, sentence_bert_config.json) are never downloaded: the adapter
# does its own mean pooling.
ARG E5_MODEL_REPO=intfloat/multilingual-e5-base
# The revision is the global ARG at the top of the file.
ARG E5_MODEL_REVISION
ARG E5_MODEL_SAFETENSORS_SHA256=a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7
ARG E5_MODEL_TOKENIZER_SHA256=62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626
ARG E5_MODEL_SPM_SHA256=cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865
ARG E5_MODEL_CONFIG_SHA256=9dab198f24c8c0879e481cf7822005d5ecbceedbacb390ffafa594e28d31bac4
ARG E5_MODEL_TOKENIZER_CONFIG_SHA256=efb5c0d09722e5fe59a462cd2a9976ee216d55b037597d997cd3fe833216da15
ARG E5_MODEL_SPECIAL_TOKENS_SHA256=06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f
ARG E5_MODEL_CARD_SHA256=9d129cb0b912b55c7b5378f976a9c35ae4796c80466b659ad63930d8ca090fca

RUN python - <<'PY'
import hashlib
import os
import pathlib
import shutil

from huggingface_hub import snapshot_download

expected = {
    "README.md": "E5_MODEL_CARD_SHA256",
    "config.json": "E5_MODEL_CONFIG_SHA256",
    "model.safetensors": "E5_MODEL_SAFETENSORS_SHA256",
    "sentencepiece.bpe.model": "E5_MODEL_SPM_SHA256",
    "special_tokens_map.json": "E5_MODEL_SPECIAL_TOKENS_SHA256",
    "tokenizer.json": "E5_MODEL_TOKENIZER_SHA256",
    "tokenizer_config.json": "E5_MODEL_TOKENIZER_CONFIG_SHA256",
}
files = sorted(expected)
target = pathlib.Path("/models/multilingual-e5-base")
snapshot_download(
    os.environ["E5_MODEL_REPO"], revision=os.environ["E5_MODEL_REVISION"],
    allow_patterns=files, local_dir=target,
)
shutil.rmtree(target / ".cache", ignore_errors=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


present = sorted(p.name for p in target.iterdir())
assert present == files, f"unexpected model files: {present}"
for name, env in expected.items():
    actual = sha256(target / name)
    assert actual == os.environ[env], f"{name}: sha256 {actual} != {os.environ[env]}"
repo, revision = os.environ["E5_MODEL_REPO"], os.environ["E5_MODEL_REVISION"]
(target / "NOTICE").write_text(
    f"Model: {repo}\n"
    f"Source: https://huggingface.co/{repo} (revision {revision})\n"
    "Authors: Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei\n"
    "(Microsoft), 'Multilingual E5 Text Embeddings: A Technical Report', arXiv:2402.05672.\n"
    "Licence: MIT License, as declared in the model card README.md.\n"
    "Base model: FacebookAI/xlm-roberta-base (XLM-RoBERTa, Conneau et al., 2019), MIT License.\n"
    "The model files in this directory are redistributed unmodified.\n",
    encoding="utf-8",
)
for name in sorted(p.name for p in target.iterdir()):
    print(name, (target / name).stat().st_size)
PY

# Read-only for everyone; COPY keeps these modes (owner root), so the app user cannot change it.
RUN chmod -R a=rX /models


FROM ${PYTHON_IMAGE} AS base

ARG E5_MODEL_REVISION

# uv 0.12.19 (same version as the host), pinned by digest.
COPY --from=ghcr.io/astral-sh/uv:0.12.19@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 /uv /uvx /usr/local/bin/

# Apt packages pinned to the versions resolved on 2026-10-27 so a rebuild cannot silently change
# the OCR engine. To bump: build with the pins removed (or run
#   docker run --rm --user root --entrypoint sh docintel-test -c \
#     'apt-get update -qq && apt-cache policy tesseract-ocr libtesseract5 libleptonica6 tini'
# ), copy the new candidate versions here, then rebuild and rerun the suite. If Debian drops a
# pinned version from the mirror the build fails loudly; bump the pins (or the base digest) then.
# tesseract-ocr-osd is a dependency of tesseract-ocr, pinned so the full set is fixed.
# tini: minimal init as PID 1, reaps orphaned/zombie processes and forwards signals.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr=5.5.0-1+b1 \
        libtesseract5=5.5.0-1+b1 \
        libleptonica6=1.84.1-4 \
        tesseract-ocr-osd=1:4.1.0-2 \
        tesseract-ocr-eng=1:4.1.0-2 \
        tesseract-ocr-spa=1:4.1.0-2 \
        tesseract-ocr-por=1:4.1.0-2 \
        tesseract-ocr-fra=1:4.1.0-2 \
        tesseract-ocr-deu=1:4.1.0-2 \
        tini=0.19.0-3+b8 \
    && rm -rf /var/lib/apt/lists/*

# OMP_THREAD_LIMIT is deliberately not set image-wide: it would also cap torch (NER) at one
# thread. The Tesseract runner sets OMP_THREAD_LIMIT=1 on each tesseract process instead (one
# thread per process; parallelism comes from page workers).
# UV_PROJECT_ENVIRONMENT keeps the venv outside the copied source tree.
# HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE: the model is baked in, nothing is fetched at run time.
# NER_MODEL_PATH: where the baked-in model lives (read by the `ner_model` tests and app settings).
# EMBEDDING_MODEL_PATH/EMBEDDING_MODEL_REVISION: the baked-in e5 embedder and its pinned commit.
# MALLOC_ARENA_MAX=2: glibc otherwise keeps one memory arena per worker thread; under load the
# service's RSS kept growing by 16-520 MB per round of 200 requests, while with 2 arenas it stays
# flat about 840 MB lower at the same throughput (docs/BENCHMARKS.md section 15).
ENV PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_ARENA_MAX=2 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    NER_MODEL_PATH=/models/xlm-roberta-base-ner-hrl \
    EMBEDDING_MODEL_PATH=/models/multilingual-e5-base \
    EMBEDDING_MODEL_REVISION=${E5_MODEL_REVISION}

RUN useradd --create-home --uid 10001 app

COPY --from=model /models /models


FROM base AS test

WORKDIR /app

# Dependencies first so the layer is reused while only source changes.
# The uv cache lives in a BuildKit cache mount: reused across builds, never written into a layer
# (UV_LINK_MODE=copy is required because the mount is a separate filesystem).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev --no-install-project

COPY --chown=app:app . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev \
    && chown app:app /app

USER app

# tini as PID 1 so subprocesses (e.g. tesseract) orphaned by a killed worker are reaped and
# SIGTERM/SIGINT reach pytest. Any command passed to `docker run` still runs under tini.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Plain `pytest -q`: the coverage gate is enforced by the pre-push hook on the host.
# For a coverage report here: docker run --rm docintel-test pytest -q --cov
CMD ["pytest", "-q"]
