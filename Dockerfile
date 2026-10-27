# syntax=docker/dockerfile:1
#
# Stages:
#   base  - Python 3.12 on Debian 13 (trixie) + Tesseract 5 with eng/spa/por/fra/deu + uv
#   test  - project + dev dependencies; runs the full pytest suite (incl. `tesseract` tests)
# A `runtime` stage (no dev deps) is added later on top of `base`.
#
# Build and run the tests:
#   docker build --target test -t docintel-test .
#   docker run --rm docintel-test

# python:3.12-slim-trixie, pinned by digest (resolved 2026-10-27). Debian 13 ships Tesseract 5.5,
# the version the OCR defaults were benchmarked with; bump tag and digest together.
FROM python:3.12-slim-trixie@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f AS base

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

# OMP_THREAD_LIMIT=1: one Tesseract thread per process; parallelism comes from page workers.
# UV_PROJECT_ENVIRONMENT keeps the venv outside the copied source tree.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    OMP_THREAD_LIMIT=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN useradd --create-home --uid 10001 app


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
