"""Fetch the pinned pdf.js files of the audit viewer (standard library only).

    uv run python scripts/fetch_pdfjs.py            # into vendor/pdfjs (git- and docker-ignored)

Downloads the npm tarball named by ``deploy/pdfjs.lock``, checks its sha256, and writes only
the files of ``FILES`` from it (the library, its worker, the two JavaScript image decoders pdf.js
imports when WebAssembly is off, and the licences), each checked against its own sha256, flat
into the destination (by default ``vendor/pdfjs`` at the project root,
where the service looks when ``PDFJS_DIR`` is unset). Nothing is written unless every check
passes, and a destination that already holds the pinned files is left alone. The Dockerfile's
``pdfjs`` stage runs this script with ``--lock`` and ``--dest``, so the host and the images get
the same bytes from the same pin. ``--tarball`` reads a tarball already on disk instead of
downloading it.

Exit status: 0 when the files are in place, 1 when a check fails or the download fails.
"""

import argparse
import hashlib
import http.client
import io
import os
import re
import sys
import tarfile
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "deploy" / "pdfjs.lock"
DEFAULT_DEST = ROOT / "vendor" / "pdfjs"

FILES: Mapping[str, tuple[str, str]] = {
    "pdf.min.mjs": ("package/build/pdf.min.mjs", "PDFJS_LIB_SHA256"),
    "pdf.worker.min.mjs": ("package/build/pdf.worker.min.mjs", "PDFJS_WORKER_SHA256"),
    "LICENSE": ("package/LICENSE", "PDFJS_LICENSE_SHA256"),
    "jbig2_nowasm_fallback.js": (
        "package/wasm/jbig2_nowasm_fallback.js",
        "PDFJS_JBIG2_FALLBACK_SHA256",
    ),
    "openjpeg_nowasm_fallback.js": (
        "package/wasm/openjpeg_nowasm_fallback.js",
        "PDFJS_OPENJPEG_FALLBACK_SHA256",
    ),
    "LICENSE_JBIG2": ("package/wasm/LICENSE_JBIG2", "PDFJS_LICENSE_JBIG2_SHA256"),
    "LICENSE_OPENJPEG": ("package/wasm/LICENSE_OPENJPEG", "PDFJS_LICENSE_OPENJPEG_SHA256"),
    "LICENSE_PDFJS_JBIG2": ("package/wasm/LICENSE_PDFJS_JBIG2", "PDFJS_LICENSE_PDFJS_JBIG2_SHA256"),
    "LICENSE_PDFJS_OPENJPEG": (
        "package/wasm/LICENSE_PDFJS_OPENJPEG",
        "PDFJS_LICENSE_PDFJS_OPENJPEG_SHA256",
    ),
}
"""Each file kept -> its member in the npm tarball and the lock entry of its sha256."""

REGISTRY_PREFIX = "https://registry.npmjs.org/pdfjs-dist/-/pdfjs-dist-"
MAX_TARBALL_BYTES = 64 * 1024 * 1024
"""Largest tarball read (6.4.299 is 8.5 MB): a wrong URL cannot fill the memory."""
MAX_FILE_BYTES = 16 * 1024 * 1024
"""Largest file taken from the tarball (the worker of 6.4.299 is 1.3 MB)."""
DOWNLOAD_TIMEOUT_S = 60.0
"""Longest wait for the connection and for each read of the body."""
DOWNLOAD_DEADLINE_S = 300.0
"""Longest whole download: a server that keeps sending a byte now and then is cut here (checked
after each read, so the worst case is this plus one ``DOWNLOAD_TIMEOUT_S``)."""
READ_CHUNK_BYTES = 256 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}")
_VERSION = re.compile(r"\d+\.\d+\.\d+")
_ASSIGNMENT = re.compile(r"([A-Z][A-Z0-9_]*)=(.*)")
_SAFE_VALUE = re.compile(r"[A-Za-z0-9._:/-]+")
"""Characters a lock value may have: the lock is also sourced by ``sh`` (the hygiene check), so
no space, quote, ``$``, backquote, ``;`` or other shell syntax."""


class PinError(Exception):
    """The lock, the tarball or a file does not match the pin."""


@dataclass(frozen=True)
class Pin:
    """The pinned release: version, tarball and the sha256 of the tarball and of each file."""

    version: str
    tarball_url: str
    tarball_sha256: str
    file_sha256: Mapping[str, str]


def lock_entries() -> list[str]:
    """The names a lock must set, and the only ones it may set."""
    return [
        "PDFJS_VERSION",
        "PDFJS_TARBALL_URL",
        "PDFJS_TARBALL_SHA256",
        *(entry for _, entry in FILES.values()),
    ]


def read_lock(path: Path) -> Pin:
    """The pin in ``path`` (``NAME=value`` lines, ``#`` comments); ``PinError`` when malformed:
    a line that is not ``NAME=value``, a value with characters outside ``_SAFE_VALUE``, an
    unknown or repeated name, a missing one, or a value of the wrong form."""
    names = lock_entries()
    values = _assignments(path, frozenset(names))
    missing = [name for name in names if name not in values]
    if missing:
        raise PinError(f"{path.name}: missing {', '.join(missing)}")
    version, url = values["PDFJS_VERSION"], values["PDFJS_TARBALL_URL"]
    if _VERSION.fullmatch(version) is None:
        raise PinError(f"{path.name}: PDFJS_VERSION is not a release version")
    if url != f"{REGISTRY_PREFIX}{version}.tgz":
        raise PinError(f"{path.name}: PDFJS_TARBALL_URL is not the npm tarball of {version}")
    hashes = [name for name in names if name.endswith("_SHA256")]
    malformed = [name for name in hashes if _SHA256.fullmatch(values[name]) is None]
    if malformed:
        raise PinError(f"{path.name}: not a lower-case sha256: {', '.join(malformed)}")
    return Pin(
        version=version,
        tarball_url=url,
        tarball_sha256=values["PDFJS_TARBALL_SHA256"],
        file_sha256={name: values[entry] for name, (_, entry) in FILES.items()},
    )


def _assignments(path: Path, known: frozenset[str]) -> dict[str, str]:
    """The ``NAME=value`` lines of ``path``, each name known and set once, each value safe."""
    values: dict[str, str] = {}
    lines: dict[str, int] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGNMENT.fullmatch(line)
        if match is None:
            raise PinError(f"{path.name}:{number}: not a NAME=value line")
        name, value = match[1], match[2]
        if name not in known:
            raise PinError(f"{path.name}:{number}: unknown entry {name}")
        if name in lines:
            raise PinError(f"{path.name}:{number}: {name} is already set on line {lines[name]}")
        if _SAFE_VALUE.fullmatch(value) is None:
            raise PinError(
                f"{path.name}:{number}: the value of {name} is empty or has a character "
                "other than letters, digits and . _ : / -"
            )
        values[name], lines[name] = value, number
    return values


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    """The body at ``url`` (an npm registry tarball): at most ``MAX_TARBALL_BYTES``, within
    ``DOWNLOAD_DEADLINE_S``; a broken HTTP exchange is a ``PinError``."""
    if not url.startswith(REGISTRY_PREFIX):
        raise PinError("the tarball URL is not on the npm registry")
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_S
    chunks: list[bytes] = []
    size = 0
    try:
        # S310: the scheme and host are the registry's, checked above.
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:  # noqa: S310
            while chunk := response.read1(READ_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_TARBALL_BYTES:
                    raise PinError(f"the tarball is over {MAX_TARBALL_BYTES} bytes")
                if time.monotonic() > deadline:
                    raise PinError(f"the download took over {DOWNLOAD_DEADLINE_S:.0f} s")
                chunks.append(chunk)
    except http.client.HTTPException as error:
        raise PinError(f"the download failed ({type(error).__name__})") from None
    return b"".join(chunks)


def extract(tarball: bytes, pin: Pin) -> dict[str, bytes]:
    """The kept files of ``tarball``, each checked; ``PinError`` on any mismatch."""
    actual = sha256(tarball)
    if actual != pin.tarball_sha256:
        raise PinError(f"tarball sha256 {actual} != pinned {pin.tarball_sha256}")
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as archive:
        for name, (member_name, _) in FILES.items():
            try:
                member = archive.getmember(member_name)
            except KeyError:
                raise PinError(f"{member_name} is not in the tarball") from None
            if not member.isfile() or member.size > MAX_FILE_BYTES:
                raise PinError(f"{member_name} is not a regular file of at most {MAX_FILE_BYTES}")
            stream = archive.extractfile(member)
            if stream is None:  # pragma: no cover - isfile() above guarantees a stream
                raise PinError(f"{member_name} cannot be read")
            files[name] = stream.read()
    for name, data in files.items():
        actual = sha256(data)
        if actual != pin.file_sha256[name]:
            raise PinError(f"{name}: sha256 {actual} != pinned {pin.file_sha256[name]}")
    return files


def is_current(dest: Path, pin: Pin) -> bool:
    """Whether ``dest`` already holds every kept file with its pinned sha256."""
    return all(
        (dest / name).is_file() and sha256((dest / name).read_bytes()) == expected
        for name, expected in pin.file_sha256.items()
    )


def install(files: Mapping[str, bytes], dest: Path) -> None:
    """Write ``files`` into ``dest``, each through a temporary file and an atomic rename."""
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        partial = dest / f".{name}.partial"
        partial.write_bytes(data)
        os.replace(partial, dest / name)


def fetch(pin: Pin, dest: Path, tarball: Path | None = None) -> str:
    """Put the pinned files into ``dest``; returns what was done, for the report line."""
    if is_current(dest, pin):
        return f"pdf.js {pin.version} already in {dest}"
    data = tarball.read_bytes() if tarball is not None else download(pin.tarball_url)
    files = extract(data, pin)
    install(files, dest)
    sizes = ", ".join(f"{name} {len(content)} bytes" for name, content in files.items())
    return f"pdf.js {pin.version} written to {dest}: {sizes}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the pinned pdf.js files.")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK, help="the pin file")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="target directory")
    parser.add_argument("--tarball", type=Path, help="a tarball on disk instead of downloading")
    args = parser.parse_args(argv)
    try:
        report = fetch(read_lock(args.lock), args.dest, args.tarball)
    except (PinError, OSError, tarfile.TarError) as error:
        print(f"fetch_pdfjs: {error}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
