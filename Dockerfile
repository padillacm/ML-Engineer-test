# syntax=docker/dockerfile:1
#
# Stages:
#   model - downloads the NER model (Davlan/xlm-roberta-base-ner-hrl) and the embedding model
#           (intfloat/multilingual-e5-base), each at a pinned revision with sha256-checked files
#   prompt-guard - downloads the injection detector (meta-llama/Llama-Prompt-Guard-2-86M, gated:
#           needs the `hf_token` build secret), pinned and checked the same way; built only when
#           INJECTION_MODEL=1 (the default)
#   uv    - the uv binaries only (pinned by digest); copied into `test`, mounted while `runtime`
#           installs, so the runtime image has no uv
#   pdfjs - downloads the audit viewer's pdf.js (npm pdfjs-dist, pinned with its sha256 in
#           deploy/pdfjs.lock) and keeps only the files the lock lists
#   base  - Python 3.12 on Debian 13 (trixie) + Tesseract 5 with eng/spa/por/fra/deu
#           + the models at /models (read-only; Hugging Face offline mode). Chosen by the
#           INJECTION_MODEL build argument: `base-1` adds the detector and sets
#           INJECTION_MODEL_PATH, `base-0` has neither (the service then runs with injection
#           detection off); `base` is the one selected
#   test    - project + dev dependencies + uv; runs the full pytest suite (incl. `tesseract` tests)
#   chromium - downloads the Chrome Headless Shell that Playwright pins, checked against its sha256
#   e2e     - `test` + the `e2e` dependency group (Playwright) + that headless Chromium, for
#             the browser tests of the audit viewer; the only stage with a browser
#   runtime - the service: runtime dependencies and `app/` only (no tests, no dev tools, no uv),
#             non-root, tini as PID 1, Uvicorn on port 8000 with a readiness healthcheck. Last stage, so a
#             plain `docker build .` builds it; docker-compose.yml runs it with Qdrant.
#
# Build and run the tests:
#   docker build --target test -t docintel-test .
#   docker run --rm docintel-test
# Build the service image (docker compose build does the same, reading HF_TOKEN from .env):
#   docker build --target runtime --secret id=hf_token,env=HF_TOKEN -t docintel .
# Without the detector model (no token needed; the service starts with detection off):
#   docker build --target runtime --build-arg INJECTION_MODEL=0 -t docintel .
# With the CUDA build of torch (run it with `--gpus all`; compose's `gpu` profile builds this one):
#   docker build --target runtime --build-arg TORCH_VARIANT=cuda --secret id=hf_token,env=HF_TOKEN -t docintel:cuda .

# python:3.12-slim-trixie, pinned by digest (resolved 2026-10-27). Debian 13 ships Tesseract 5.5,
# the version the OCR defaults were benchmarked with; bump tag and digest together.
ARG PYTHON_IMAGE=python:3.12-slim-trixie@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f
# Embedding model commit (intfloat/multilingual-e5-base main, resolved 2026-10-27). Global so the
# `model` stage downloads it and `base` exposes it as EMBEDDING_MODEL_REVISION (model identity).
ARG E5_MODEL_REVISION=d128750597153bb5987e10b1c3493a34e5a4502a
# 1: download the Prompt Guard 2 model (needs the `hf_token` build secret) and set
# INJECTION_MODEL_PATH in the images; 0: skip the `prompt-guard` stage and leave the variable
# unset, so the service runs with injection detection off. Any other value fails the build.
ARG INJECTION_MODEL=1
# huggingface_hub of the download stages, pinned by version only: --require-hashes would need
# every transitive dependency hashed too, and the downloaded model files are hash-checked anyway.
ARG HF_HUB_VERSION=2.3.0
# The torch build of the `test`, `e2e` and `runtime` stages: the pyproject.toml extra of the same
# name. cpu (default): the PyTorch CPU wheel, no NVIDIA library in the image (checked below).
# cuda: the CUDA 12.6 wheel, whose nvidia-* dependencies bring the CUDA runtime, so the base image
# stays python:3.12-slim (no nvidia/cuda image); it needs a GPU at run time (`--gpus all`, or the
# reservation of compose's `gpu` profile). Any other value fails the build.
ARG TORCH_VARIANT=cpu


# NER model, downloaded at build time so run time never touches the network. Only this stage has
# huggingface_hub; `base` receives the model files alone.
# To bump the revision: take the new commit "sha" from
#   https://huggingface.co/api/models/Davlan/xlm-roberta-base-ner-hrl
# and the "lfs.oid" (sha256) of the two LFS files from
#   https://huggingface.co/api/models/Davlan/xlm-roberta-base-ner-hrl/tree/<sha>
# The small files are plain git blobs (their "oid" there is a git sha1), so take their sha256 from
#   https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl/resolve/<sha>/<file>  (| sha256sum)
# Update the revision and every *_SHA256 ARG below, rebuild and rerun the suite (the `ner_model`
# tests use the model). scripts/resolve_model_hashes.py prints the ARG lines for a revision:
#   uv run python scripts/resolve_model_hashes.py Davlan/xlm-roberta-base-ner-hrl <sha> \
#       --prefix NER_MODEL README.md config.json model.safetensors sentencepiece.bpe.model \
#       special_tokens_map.json tokenizer_config.json
FROM ${PYTHON_IMAGE} AS model

ARG HF_HUB_VERSION
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

# Prompt injection detector (meta-llama/Llama-Prompt-Guard-2-86M), downloaded like the models
# above. The repository is gated: the build needs a Hugging Face read token of an account that
# accepted the Llama 4 Community License, passed as the BuildKit secret `hf_token` (an
# environment variable of this RUN step only, in no layer, never in the image):
#   docker build --secret id=hf_token,env=HF_TOKEN ...   (compose: `build.secrets`, from .env)
# The Hub masks the sha256 of the LFS files of a gated repository in its public tree listing, so
# to bump the revision run, with HF_TOKEN in the environment:
#   uv run python scripts/resolve_model_hashes.py meta-llama/Llama-Prompt-Guard-2-86M <sha> \
#       --prefix PROMPT_GUARD README.md LICENSE USE_POLICY.md config.json model.safetensors \
#       special_tokens_map.json tokenizer.json tokenizer_config.json
# and copy its ARG lines here. Only what AutoModelForSequenceClassification + AutoTokenizer load
# (config, safetensors weights, the fast tokenizer and its two small configs) plus the model
# card, the licence and the use policy; nothing else of the repository is downloaded.
# Stage referenced by `base-1` only: with INJECTION_MODEL=0 it is never built.
FROM ${PYTHON_IMAGE} AS prompt-guard

ARG HF_HUB_VERSION
ARG PROMPT_GUARD_REPO=meta-llama/Llama-Prompt-Guard-2-86M
# Commit of the Hub repository's main branch, resolved 2026-10-29.
ARG PROMPT_GUARD_REVISION=a8ded8e697ce7c355e395a0df51f94adb4a2fd27
ARG PROMPT_GUARD_CARD_SHA256=5d2a3e0c46397609a784a5034dc823da408aa8b5a414828609f32dc18ee16933
ARG PROMPT_GUARD_LICENSE_SHA256=73755cee886613ae3135140882289e0a4955d5f0e90f8b1c1ffc962aa9082917
ARG PROMPT_GUARD_USE_POLICY_SHA256=5ae40fe842b87b5773c47cfd25992a496cc09b17cbb246968762f06352d89270
ARG PROMPT_GUARD_CONFIG_SHA256=cd54ac39a1f2c3c5146bd5295b34038f8b4d9069e2f844450da014a523bb7653
ARG PROMPT_GUARD_SAFETENSORS_SHA256=e72017dbbe89c1232dcbc4a74ce0c389db5b468c42afd05850347b2a8c5f6b09
ARG PROMPT_GUARD_SPECIAL_TOKENS_SHA256=9463f61e1b109a8eb4688b829260d7c6b1e6dff04c98ff7269bb89e2b92369b9
ARG PROMPT_GUARD_TOKENIZER_SHA256=3e7e96867c2acdd575f0862c74822e05d1d15b93d9d9a4a2144b1ce83ae3339f
ARG PROMPT_GUARD_TOKENIZER_CONFIG_SHA256=fa09cf22b9d723377f95706967f789ca1a8267e1412385705e1ed51ce0559189

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir "huggingface_hub==${HF_HUB_VERSION}"

# The token is read from the environment of this step only (never printed, never saved: the
# download passes it explicitly, no `login`), so `docker history` and the image never hold it.
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN python - <<'PY'
import hashlib
import os
import pathlib
import shutil

from huggingface_hub import snapshot_download

repo, revision = os.environ["PROMPT_GUARD_REPO"], os.environ["PROMPT_GUARD_REVISION"]
token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    raise SystemExit(
        f"{repo} is gated: build with --secret id=hf_token,env=HF_TOKEN (docker compose build "
        "reads HF_TOKEN from .env), or without the detector: --build-arg INJECTION_MODEL=0"
    )
expected = {
    "LICENSE": "PROMPT_GUARD_LICENSE_SHA256",
    "README.md": "PROMPT_GUARD_CARD_SHA256",
    "USE_POLICY.md": "PROMPT_GUARD_USE_POLICY_SHA256",
    "config.json": "PROMPT_GUARD_CONFIG_SHA256",
    "model.safetensors": "PROMPT_GUARD_SAFETENSORS_SHA256",
    "special_tokens_map.json": "PROMPT_GUARD_SPECIAL_TOKENS_SHA256",
    "tokenizer.json": "PROMPT_GUARD_TOKENIZER_SHA256",
    "tokenizer_config.json": "PROMPT_GUARD_TOKENIZER_CONFIG_SHA256",
}
files = sorted(expected)
target = pathlib.Path("/models/prompt-guard")
snapshot_download(repo, revision=revision, allow_patterns=files, local_dir=target, token=token)
del token
shutil.rmtree(target / ".cache", ignore_errors=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


present = sorted(p.name for p in target.iterdir())
assert present == files, f"unexpected model files: {present}"
mismatches = []
for name, env in expected.items():
    actual, pinned = sha256(target / name), os.environ[env]
    if actual != pinned:
        mismatches.append(f"{name}: sha256 {actual} != pinned {env}={pinned}")
if mismatches:
    raise SystemExit(
        "prompt-guard: the downloaded files do not match the pinned hashes (resolve them with "
        "scripts/resolve_model_hashes.py):\n" + "\n".join(mismatches)
    )
(target / "NOTICE").write_text(
    f"Model: {repo}\n"
    f"Source: https://huggingface.co/{repo} (revision {revision})\n"
    "Author: Meta Platforms, Inc. (Llama Prompt Guard 2, 86M parameters; the model card names\n"
    "mDeBERTa-base (mdeberta-v3, Microsoft, MIT License) as its base model).\n"
    "Licence: Llama 4 Community License, as declared in the model card README.md; its text is\n"
    "LICENSE and its acceptable use policy USE_POLICY.md in this directory.\n"
    "Built with Llama\n"
    "Llama 4 is licensed under the Llama 4 Community License, Copyright \u00a9 Meta Platforms,\n"
    "Inc. All Rights Reserved.\n"
    "The model files in this directory are redistributed unmodified.\n",
    encoding="utf-8",
)
for name in sorted(p.name for p in target.iterdir()):
    print(name, (target / name).stat().st_size)
PY

# Read-only for everyone, as for the other models.
RUN chmod -R a=rX /models


# uv 0.12.19 (same version as the host), pinned by digest.
FROM ghcr.io/astral-sh/uv:0.12.19@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 AS uv


# pdf.js for the audit viewer (/audit), fetched at build time like the models: the stage runs
# scripts/fetch_pdfjs.py, the same script as a host run, against the one pin in
# deploy/pdfjs.lock. It downloads the npm tarball, fails the build unless the tarball and each
# kept file match their sha256, and writes only the files the lock lists (library, worker, the two
# JavaScript image decoders used with WebAssembly off, licences; no .wasm).
# To bump the version: see deploy/pdfjs.lock.
FROM ${PYTHON_IMAGE} AS pdfjs

COPY deploy/pdfjs.lock scripts/fetch_pdfjs.py /build/
RUN python -I /build/fetch_pdfjs.py --lock /build/pdfjs.lock --dest /pdfjs \
    && chmod -R a=rX /pdfjs


FROM ${PYTHON_IMAGE} AS base-common

ARG E5_MODEL_REVISION

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


# `base` with the injection detector: its files next to the other models (read-only, owner root,
# as copied) and INJECTION_MODEL_PATH pointing at them, so the service loads it at start-up.
FROM base-common AS base-1

COPY --from=prompt-guard /models/prompt-guard /models/prompt-guard
ENV INJECTION_MODEL_PATH=/models/prompt-guard


# `base` without it: no detector files and no INJECTION_MODEL_PATH (the setting stays unset, the
# service logs "injection detection off"). Nothing to add.
FROM base-common AS base-0


# The variant the INJECTION_MODEL build argument names (1 or 0; another value names no stage and
# the build fails). BuildKit builds only the stages the target depends on, so with 0 the
# `prompt-guard` stage never runs and no token is needed.
FROM base-${INJECTION_MODEL} AS base


FROM base AS test

ARG TORCH_VARIANT
# Validated before the first sync, so a wrong value fails fast with its own message.
RUN case "${TORCH_VARIANT}" in cpu|cuda) ;; \
      *) echo "TORCH_VARIANT must be cpu or cuda, not '${TORCH_VARIANT}'" >&2; exit 1 ;; esac
# Kept in the image, so its checks know the variant (tests/deploy/test_torch_variant.py, the
# smoke `hygiene` service); the service does not read it.
ENV TORCH_VARIANT=${TORCH_VARIANT}

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies first so the layer is reused while only source changes.
# The uv cache lives in a BuildKit cache mount: reused across builds, never written into a layer
# (UV_LINK_MODE=copy is required because the mount is a separate filesystem).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev --extra "${TORCH_VARIANT}" --no-install-project

COPY --chown=app:app . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev --extra "${TORCH_VARIANT}" \
    && chown app:app /app

# The torch variant, checked after the sync: the CPU image holds no NVIDIA library (no nvidia-*,
# cuda-* or triton distribution) and a torch without CUDA; the CUDA image a torch built with CUDA
# (tests/deploy/test_torch_variant.py and the smoke `hygiene` service check it again).
RUN python -I - <<'PY'
import importlib.metadata as md
import os

import torch

variant = os.environ["TORCH_VARIANT"]
names = sorted({(d.metadata["Name"] or "").lower() for d in md.distributions()})
gpu = [n for n in names if n.startswith(("nvidia-", "cuda-")) or n == "triton"]
if variant == "cpu" and (gpu or torch.version.cuda is not None):
    raise SystemExit(f"CPU image with CUDA: torch {torch.__version__}, packages {gpu}")
if variant == "cuda" and torch.version.cuda is None:
    raise SystemExit(f"CUDA image with a CPU torch: {torch.__version__}")
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}); NVIDIA packages: {len(gpu)}")
PY

# pdf.js, read-only and owned by root as in `runtime` (COPY keeps the stage's modes). Late in the
# stage, so a new pin rebuilds only this layer. PDFJS_DIR names it (the service's default for
# this layout is the same path).
COPY --from=pdfjs /pdfjs /app/vendor/pdfjs
ENV PDFJS_DIR=/app/vendor/pdfjs

USER app

# tini as PID 1 so subprocesses (e.g. tesseract) orphaned by a killed worker are reaped and
# SIGTERM/SIGINT reach pytest. Any command passed to `docker run` still runs under tini.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Plain `pytest -q`: the coverage gate is enforced by the pre-push hook on the host.
# For a coverage report here: docker run --rm docintel-test pytest -q --cov
CMD ["pytest", "-q"]


# The Chrome Headless Shell of the `e2e` stage, fetched and checked like the models: the build of
# Chrome for Testing that Playwright 1.64.0 pins (browsers.json of its driver: chromium-headless-
# shell revision 1248, browser version 156.0.8078.4), from the URL Playwright itself downloads it
# from (its registry's `cftUrl`), checked against a pinned sha256 and unpacked into the layout
# Playwright looks for: $PLAYWRIGHT_BROWSERS_PATH/chromium_headless_shell-<revision>/
# chrome-headless-shell-linux64/chrome-headless-shell, with the INSTALLATION_COMPLETE marker that
# `playwright install` writes. (`playwright install` checks no hash.) The sha256 was taken on
# 2026-10-29; the archive's executable has the same sha256 as the one `playwright install`
# had written (docs/BENCHMARKS.md section 20.7).
# To bump Playwright: take "revision" and "browserVersion" of chromium-headless-shell from
# playwright/driver/package/browsers.json of the new version, download the archive, put its
# sha256 here, rebuild and rerun the e2e suite.
FROM ${PYTHON_IMAGE} AS chromium

ARG CHROMIUM_HEADLESS_SHELL_VERSION=156.0.8078.4
ARG CHROMIUM_HEADLESS_SHELL_REVISION=1248
ARG CHROMIUM_HEADLESS_SHELL_SHA256=c3e50596aef52c58f8fda3c6b3411c33d34c88b177b3ed1c2243359fd3b8ba20

RUN python -I - <<'PY'
import hashlib
import io
import os
import pathlib
import stat
import urllib.request
import zipfile

version = os.environ["CHROMIUM_HEADLESS_SHELL_VERSION"]
revision = os.environ["CHROMIUM_HEADLESS_SHELL_REVISION"]
expected = os.environ["CHROMIUM_HEADLESS_SHELL_SHA256"]
url = f"https://cdn.playwright.dev/builds/cft/{version}/linux64/chrome-headless-shell-linux64.zip"
top = "chrome-headless-shell-linux64/"

with urllib.request.urlopen(url, timeout=60) as response:
    data = response.read(512 * 1024 * 1024)
actual = hashlib.sha256(data).hexdigest()
if actual != expected:
    raise SystemExit(f"chrome-headless-shell: sha256 {actual} != pinned {expected}")

dest = pathlib.Path("/ms-playwright") / f"chromium_headless_shell-{revision}"
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for member in archive.infolist():
        name = member.filename
        mode = member.external_attr >> 16
        parts = pathlib.PurePosixPath(name).parts
        if not name.startswith(top) or name.startswith("/") or ".." in parts:
            raise SystemExit(f"chrome-headless-shell: unexpected member {name!r}")
        if stat.S_ISLNK(mode):
            raise SystemExit(f"chrome-headless-shell: symbolic link {name!r}")
        target = dest.joinpath(*parts)
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(member))
        target.chmod(0o755 if mode & 0o111 else 0o644)
executable = dest / top / "chrome-headless-shell"
if not os.access(executable, os.X_OK):
    raise SystemExit("chrome-headless-shell: the executable is missing")
(dest / "INSTALLATION_COMPLETE").touch()
print(f"chrome-headless-shell {version} (revision {revision}): {len(data)} bytes, sha256 ok")
PY


# Browser end-to-end tests of the audit viewer (tests/e2e, docker-compose.e2e.yml): the test
# image plus the `e2e` dependency group (Playwright for Python, pinned in pyproject.toml and
# uv.lock) and the Chrome Headless Shell of the `chromium` stage (the build Playwright's headless
# launch uses; the full browser is not needed and adds 403 MB, docs/BENCHMARKS.md section 20.5).
# `playwright install-deps` installs only the system libraries it needs, through apt from the
# same Debian sources (signed by the Debian archive keyring) the `base` stage takes Tesseract
# from; those package versions are not pinned (KNOWN_LIMITATIONS V6). One launch of the browser
# as root checks that it starts and records Playwright's DEPENDENCIES_VALIDATED marker.
# Browsers live in /ms-playwright, readable by the app user; /e2e-artifacts is the tests'
# output directory, writable by it. Neither `test` nor `runtime` contains Playwright or a browser.
FROM test AS e2e

ARG TORCH_VARIANT
USER root

ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    E2E_ARTIFACTS=/e2e-artifacts

COPY --from=chromium /ms-playwright /ms-playwright

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev --extra "${TORCH_VARIANT}" --group e2e \
    && playwright install-deps chromium-headless-shell \
    && rm -rf /var/lib/apt/lists/* \
    && python -I -c "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); p.chromium.launch(headless=True).close(); p.stop()" \
    && test -f /ms-playwright/chromium_headless_shell-1248/DEPENDENCIES_VALIDATED \
    && chmod -R a+rX,go-w /ms-playwright \
    && install -d -o app -g app /e2e-artifacts

USER app

# Needs the stack: VIEWER_URL is the page's address (docker-compose.e2e.yml sets it).
CMD ["pytest", "-q", "-rs", "-p", "no:cacheprovider", "-m", "e2e", "tests/e2e"]


FROM base AS runtime

ARG TORCH_VARIANT
# Validated before the first sync, so a wrong value fails fast with its own message.
RUN case "${TORCH_VARIANT}" in cpu|cuda) ;; \
      *) echo "TORCH_VARIANT must be cpu or cuda, not '${TORCH_VARIANT}'" >&2; exit 1 ;; esac
# Kept in the image, so its checks know the variant (tests/deploy/test_torch_variant.py, the
# smoke `hygiene` service); the service does not read it.
ENV TORCH_VARIANT=${TORCH_VARIANT}

WORKDIR /app

# Runtime dependencies only: no `dev` extra (pytest, ruff, mypy, pre-commit, PyMuPDF), torch of
# the TORCH_VARIANT extra, no project yet, so this layer is reused while only the source
# changes. Same cache mount as `test`.
# --compile-bytecode: the app user cannot write .pyc files into the root-owned venv (and
# PYTHONDONTWRITEBYTECODE is set), so without precompiled bytecode every start would recompile
# torch and transformers in memory. uv is mounted for these two steps only, never copied.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv     --mount=from=uv,source=/uv,target=/usr/local/bin/uv     uv sync --frozen --no-dev --extra "${TORCH_VARIANT}" --no-install-project --compile-bytecode

# Only the service code. The build backend (hatchling, `packages = ["app"]`) needs nothing else:
# pyproject.toml declares no readme or licence file. Code and venv stay owned by root, so the
# service cannot modify them.
COPY app/ ./app/
RUN --mount=type=cache,target=/root/.cache/uv     --mount=from=uv,source=/uv,target=/usr/local/bin/uv     uv sync --frozen --no-dev --extra "${TORCH_VARIANT}" --compile-bytecode

# The torch variant, checked after the sync: the CPU image holds no NVIDIA library (no nvidia-*,
# cuda-* or triton distribution) and a torch without CUDA; the CUDA image a torch built with CUDA
# (tests/deploy/test_torch_variant.py and the smoke `hygiene` service check it again).
RUN python -I - <<'PY'
import importlib.metadata as md
import os

import torch

variant = os.environ["TORCH_VARIANT"]
names = sorted({(d.metadata["Name"] or "").lower() for d in md.distributions()})
gpu = [n for n in names if n.startswith(("nvidia-", "cuda-")) or n == "triton"]
if variant == "cpu" and (gpu or torch.version.cuda is not None):
    raise SystemExit(f"CPU image with CUDA: torch {torch.__version__}, packages {gpu}")
if variant == "cuda" and torch.version.cuda is None:
    raise SystemExit(f"CUDA image with a CPU torch: {torch.__version__}")
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}); NVIDIA packages: {len(gpu)}")
PY

# The audit viewer's pdf.js (stage `pdfjs`): read-only, owned by root. Late, so a new pin
# rebuilds only this layer.
COPY --from=pdfjs /pdfjs /app/vendor/pdfjs
ENV PDFJS_DIR=/app/vendor/pdfjs

# The service runs on a read-only root filesystem with /tmp on tmpfs (docker-compose.yml): every
# path a library may write to points there. TMPDIR for Python's tempfile; HF_HOME for
# transformers (it writes a cache version marker under the hub cache at import; nothing is
# downloaded, HF_HUB_OFFLINE is set) and TORCH_HOME for torch's hub cache, instead of the app
# user's home, which is read-only too.
ENV TMPDIR=/tmp \
    HF_HOME=/tmp/huggingface \
    TORCH_HOME=/tmp/torch

USER app

EXPOSE 8000

# Healthy only when readiness succeeds (vector store reachable). Python's urllib, because the slim
# image has no curl; a 503 raises HTTPError, so the probe exits non-zero. The models load in about
# 10 s, hence the start period; start-interval probes every second until the first success, so
# the service is marked healthy at most 1 s after it is ready (docs/BENCHMARKS.md 19.4). A
# successful probe's access line is logged at DEBUG, so the 15 s probes do not flood the log.
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --start-interval=1s --retries=3     CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health/ready', timeout=4)"]

# tini as PID 1: reaps orphaned tesseract processes and forwards SIGTERM to Uvicorn.
ENTRYPOINT ["/usr/bin/tini", "--"]

# One worker per container (every worker loads every model, about 2.3 GB RSS; scale by replicas).
# --http h11: httptools hides a client's disconnect while a pipelined request waits (block 5 P5).
# --timeout-graceful-shutdown 30: running requests get 30 s after SIGTERM (compose waits longer).
# --limit-concurrency 64: connections open at once (above MAX_CONCURRENT_JOBS +
# MAX_CONCURRENT_QUESTIONS + the waiting room); Uvicorn answers 503 beyond it.
# --timeout-keep-alive 5: an idle keep-alive connection is closed after 5 s, so a client holding
# connections open cannot keep the limit filled. docker-compose.yml repeats this command.
# Uvicorn's access log is replaced by the service's own request log (app/observability.py).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--http", "h11", \
     "--timeout-graceful-shutdown", "30", "--no-server-header", \
     "--limit-concurrency", "64", "--timeout-keep-alive", "5"]
