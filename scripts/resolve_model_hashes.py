"""Print the Dockerfile ``ARG`` lines that pin a Hugging Face model's files by sha256.

    uv run python scripts/resolve_model_hashes.py meta-llama/Llama-Prompt-Guard-2-86M <sha> \\
        --prefix PROMPT_GUARD README.md LICENSE USE_POLICY.md config.json model.safetensors \\
        special_tokens_map.json tokenizer.json tokenizer_config.json

Downloads each named file of the repository at the given revision (a full commit sha, as the
Dockerfile pins it) into a temporary directory, hashes it and prints one ``ARG`` line per file,
named ``<PREFIX>_<FILE>_SHA256`` as the Dockerfile's ``model`` and ``prompt-guard`` stages expect
(``README.md`` is ``CARD``, ``sentencepiece.bpe.model`` is ``SPM``; any other name is its stem,
upper-cased). The Hub's public tree listing masks the sha256 of the LFS files of a gated
repository, which is why the files are downloaded rather than looked up.

A gated repository needs a read token of an account that accepted its licence: the token is
read from the ``HF_TOKEN`` environment variable only (never a command-line argument, which
would land in the shell history and the process list), is never printed, and is sent to
huggingface.co alone. With ``huggingface_hub`` installed (a dependency of the project) the
download goes through it; otherwise the standard library fetches the ``resolve`` URL directly.

Exit status: 0 when every file was hashed, 1 when a download failed or the revision is not a
commit sha.
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

# No telemetry, and no warning about the temporary cache's symlinks on Windows (the files are
# hashed and thrown away).
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    from huggingface_hub import hf_hub_download as _hub_download

    HUB_DOWNLOAD: Callable[..., str] | None = _hub_download
except ImportError:  # pragma: no cover - the project installs it; a bare interpreter may not
    HUB_DOWNLOAD = None

HUB = "https://huggingface.co"
AUTH_VARIABLE = "HF_TOKEN"  # the token's environment variable (the value is never here)
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DOWNLOAD_TIMEOUT_S = 60.0
READ_CHUNK_BYTES = 1 << 20

ARG_NAMES = {
    "README.md": "CARD",
    "sentencepiece.bpe.model": "SPM",
    "special_tokens_map.json": "SPECIAL_TOKENS",
}
"""File names whose ARG name is not the upper-cased stem."""


class ResolveError(Exception):
    """A download failed or an argument is not what the Dockerfile pins."""


def arg_name(prefix: str, file: str) -> str:
    """``PROMPT_GUARD_TOKENIZER_CONFIG_SHA256`` for ``tokenizer_config.json``."""
    stem = ARG_NAMES.get(file) or re.sub(r"[^A-Z0-9]+", "_", Path(file).stem.upper()).strip("_")
    return f"{prefix}_{stem}_SHA256"


def default_prefix(repo: str) -> str:
    """``LLAMA_PROMPT_GUARD_2_86M`` for ``meta-llama/Llama-Prompt-Guard-2-86M``."""
    return re.sub(r"[^A-Z0-9]+", "_", repo.split("/")[1].upper()).strip("_")


def sha256_of(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download_with_hub(repo: str, revision: str, file: str, into: Path, token: str | None) -> Path:
    if HUB_DOWNLOAD is None:
        raise ResolveError("huggingface_hub is not installed")
    try:
        return Path(
            HUB_DOWNLOAD(repo, file, revision=revision, token=token, cache_dir=into / "hub")
        )
    except Exception as error:  # noqa: BLE001 - every Hub error ends the same way, named
        raise ResolveError(f"{file}: {type(error).__name__}: {error}") from None


def download_with_urllib(
    repo: str, revision: str, file: str, into: Path, token: str | None
) -> Path:
    url = f"{HUB}/{repo}/resolve/{revision}/{file}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    target = into / file
    try:
        request = urllib.request.Request(url, headers=headers)  # noqa: S310 - https, the Hub
        with (
            urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response,  # noqa: S310
            target.open("wb") as out,
        ):
            shutil.copyfileobj(response, out, READ_CHUNK_BYTES)
    except urllib.error.HTTPError as error:
        hint = f" (set {AUTH_VARIABLE}; a gated repository needs an accepted licence)"
        raise ResolveError(
            f"{file}: HTTP {error.code}{hint if error.code in (401, 403) else ''}"
        ) from None
    except (urllib.error.URLError, OSError) as error:
        raise ResolveError(f"{file}: {error}") from None
    return target


Downloader = Callable[[str, str, str, Path, str | None], Path]


def resolve(
    repo: str,
    revision: str,
    files: Sequence[str],
    prefix: str,
    token: str | None,
    download: Downloader,
) -> list[tuple[str, int, str]]:
    """``(ARG line, size in bytes, file)`` for each file, in the given order."""
    lines = []
    with tempfile.TemporaryDirectory(prefix="model-hashes-") as scratch:
        into = Path(scratch)
        for file in files:
            path = download(repo, revision, file, into, token)
            lines.append(
                (f"ARG {arg_name(prefix, file)}={sha256_of(path)}", path.stat().st_size, file)
            )
    return lines


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"print the sha256 ARG lines of a model's files; the token comes from "
        f"{AUTH_VARIABLE} in the environment",
    )
    parser.add_argument("repo", help="repository id, e.g. meta-llama/Llama-Prompt-Guard-2-86M")
    parser.add_argument("revision", help="the commit sha to pin (40 hex digits)")
    parser.add_argument("files", nargs="+", help="file names to hash, e.g. config.json")
    parser.add_argument(
        "--prefix", help="ARG name prefix, e.g. PROMPT_GUARD (default: from the repository name)"
    )
    return parser.parse_args(argv)


def check_args(args: argparse.Namespace) -> None:
    if not REPO_ID.match(args.repo):
        raise ResolveError(f"not a repository id: {args.repo!r}")
    if not COMMIT_SHA.match(args.revision):
        raise ResolveError(f"not a commit sha (40 hex digits): {args.revision!r}")
    for file in args.files:
        if not FILE_NAME.match(file):
            raise ResolveError(f"not a plain file name: {file!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    token = os.environ.get(AUTH_VARIABLE, "").strip() or None
    try:
        check_args(args)
        if token is None:
            print(f"{AUTH_VARIABLE} is unset: downloading anonymously", file=sys.stderr)
        download = download_with_hub if HUB_DOWNLOAD is not None else download_with_urllib
        prefix = args.prefix or default_prefix(args.repo)
        lines = resolve(args.repo, args.revision, args.files, prefix, token, download)
    except ResolveError as error:
        print(f"resolve_model_hashes: {error}", file=sys.stderr)
        return 1
    print(f"# {args.repo} at revision {args.revision} ({HUB}/{args.repo}/tree/{args.revision})")
    for line, size, file in lines:
        print(f"# {file} ({size} bytes)")
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
