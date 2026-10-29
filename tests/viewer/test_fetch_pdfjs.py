"""``scripts/fetch_pdfjs.py``: only the pinned bytes are written (design D2).

The tarballs here are built in the test, in the npm layout (``package/...``), with a lock that
pins them; nothing is downloaded (``urlopen`` is replaced where the download path is tested).
"""

import hashlib
import http.client
import io
import tarfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import fetch_pdfjs
from scripts.fetch_pdfjs import FILES, MAX_TARBALL_BYTES, Pin, PinError, extract, read_lock

VERSION = "9.8.7"
URL = f"https://registry.npmjs.org/pdfjs-dist/-/pdfjs-dist-{VERSION}.tgz"
CONTENT = {
    **{member: f"kept {name}\n".encode() for name, (member, _) in FILES.items()},
    "package/build/pdf.mjs": b"not kept\n",
    "package/wasm/openjpeg.wasm": b"\x00asm not kept\n",
    "package/package.json": b"{}\n",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tarball(
    members: Mapping[str, bytes] = CONTENT, links: Mapping[str, str] | None = None
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            archive.addfile(info)
    return buffer.getvalue()


def _lock_text(tarball: bytes, **changes: str) -> str:
    values = {
        "PDFJS_VERSION": VERSION,
        "PDFJS_TARBALL_URL": URL,
        "PDFJS_TARBALL_SHA256": _sha(tarball),
        **{entry: _sha(CONTENT[member]) for member, entry in FILES.values()},
        **changes,
    }
    lines = ["# a test pin", ""] + [f"{name}={value}" for name, value in values.items()]
    return "\n".join(lines) + "\n"


def _write(tmp_path: Path, tarball: bytes, **changes: str) -> tuple[Path, Path]:
    lock = tmp_path / "pdfjs.lock"
    lock.write_bytes(_lock_text(tarball, **changes).encode())
    archive = tmp_path / "pdfjs.tgz"
    archive.write_bytes(tarball)
    return lock, archive


def _run(lock: Path, dest: Path, archive: Path) -> int:
    return fetch_pdfjs.main(["--lock", str(lock), "--dest", str(dest), "--tarball", str(archive)])


def test_only_the_pinned_files_are_written_flat(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lock, archive = _write(tmp_path, _tarball())
    dest = tmp_path / "out" / "pdfjs"

    assert _run(lock, dest, archive) == 0

    assert sorted(path.name for path in dest.iterdir()) == sorted(FILES)
    for name, (member, _) in FILES.items():
        assert (dest / name).read_bytes() == CONTENT[member]
    assert f"pdf.js {VERSION} written to" in capsys.readouterr().out


def test_a_destination_with_the_pinned_files_is_left_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lock, archive = _write(tmp_path, _tarball())
    dest = tmp_path / "pdfjs"
    assert _run(lock, dest, archive) == 0
    archive.unlink()  # not read again

    assert _run(lock, dest, archive) == 0
    assert "already in" in capsys.readouterr().out


def test_a_changed_file_in_the_destination_is_replaced(tmp_path: Path) -> None:
    lock, archive = _write(tmp_path, _tarball())
    dest = tmp_path / "pdfjs"
    assert _run(lock, dest, archive) == 0
    (dest / "pdf.min.mjs").write_bytes(b"tampered")

    assert _run(lock, dest, archive) == 0
    assert (dest / "pdf.min.mjs").read_bytes() == CONTENT["package/build/pdf.min.mjs"]


def test_a_tampered_tarball_fails_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lock, archive = _write(tmp_path, _tarball())
    archive.write_bytes(_tarball({**CONTENT, "package/LICENSE": b"changed\n"}))
    dest = tmp_path / "pdfjs"

    assert _run(lock, dest, archive) == 1

    assert not dest.exists()
    assert "tarball sha256" in capsys.readouterr().err


@pytest.mark.parametrize(("name", "entry"), [(n, e) for n, (_, e) in FILES.items()])
def test_a_file_that_differs_from_its_pin_fails_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str, entry: str
) -> None:
    lock, archive = _write(tmp_path, _tarball(), **{entry: "0" * 64})
    dest = tmp_path / "pdfjs"

    assert _run(lock, dest, archive) == 1

    assert not dest.exists()
    assert f"{name}: sha256" in capsys.readouterr().err


def test_a_missing_member_fails(tmp_path: Path) -> None:
    members = {k: v for k, v in CONTENT.items() if k != "package/build/pdf.worker.min.mjs"}
    tarball = _tarball(members)

    with pytest.raises(PinError, match=r"pdf\.worker\.min\.mjs is not in the tarball"):
        extract(tarball, read_lock(_write(tmp_path, tarball)[0]))


def test_a_member_that_is_a_link_fails(tmp_path: Path) -> None:
    members = {k: v for k, v in CONTENT.items() if k != "package/LICENSE"}
    tarball = _tarball(members, links={"package/LICENSE": "/etc/passwd"})

    with pytest.raises(PinError, match="not a regular file"):
        extract(tarball, read_lock(_write(tmp_path, tarball)[0]))


def test_an_oversized_member_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tarball = _tarball()
    monkeypatch.setattr(fetch_pdfjs, "MAX_FILE_BYTES", 4)

    with pytest.raises(PinError, match="not a regular file of at most 4"):
        extract(tarball, read_lock(_write(tmp_path, tarball)[0]))


def test_a_corrupt_tarball_with_the_pinned_hash_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lock, archive = _write(tmp_path, b"not a gzip file")

    assert _run(lock, tmp_path / "pdfjs", archive) == 1
    assert capsys.readouterr().err.startswith("fetch_pdfjs: ")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"PDFJS_VERSION": "latest"}, "not a release version"),
        ({"PDFJS_TARBALL_URL": "https://example.com/pdfjs-dist-9.8.7.tgz"}, "npm tarball"),
        ({"PDFJS_TARBALL_URL": URL.replace(VERSION, "1.0.0")}, "npm tarball of 9.8.7"),
        ({"PDFJS_TARBALL_SHA256": "ABC"}, "not a lower-case sha256: PDFJS_TARBALL_SHA256"),
        ({"PDFJS_LIB_SHA256": "A" * 64}, "not a lower-case sha256: PDFJS_LIB_SHA256"),
        ({"PDFJS_WORKER_SHA256": ""}, "the value of PDFJS_WORKER_SHA256 is empty"),
        ({"PDFJS_VERSION": "9.8.7;id"}, "the value of PDFJS_VERSION .* character"),
        ({"PDFJS_VERSION": "$(id)"}, "the value of PDFJS_VERSION .* character"),
        ({"PDFJS_TARBALL_URL": "`id`"}, "the value of PDFJS_TARBALL_URL .* character"),
        ({"PDFJS_LICENSE_SHA256": "'abc'"}, "the value of PDFJS_LICENSE_SHA256 .* character"),
        ({"PDFJS_VERSION": "9.8.7 extra"}, "the value of PDFJS_VERSION .* character"),
        ({"PDFJS_EXTRA": "1"}, "unknown entry PDFJS_EXTRA"),
        ({"PATH": "/usr/bin"}, "unknown entry PATH"),
    ],
)
def test_a_malformed_lock_fails_naming_the_entry(
    tmp_path: Path, changes: dict[str, str], message: str
) -> None:
    lock, _ = _write(tmp_path, _tarball(), **changes)

    with pytest.raises(PinError, match=message):
        read_lock(lock)


def test_a_line_that_is_not_an_assignment_fails(tmp_path: Path) -> None:
    lock = tmp_path / "pdfjs.lock"
    lock.write_bytes(b"# pin\nPDFJS_VERSION = 1.2.3\n")

    with pytest.raises(PinError, match=r"pdfjs\.lock:2: not a NAME=value line"):
        read_lock(lock)


def test_a_name_set_twice_fails_naming_both_lines(tmp_path: Path) -> None:
    lock, _ = _write(tmp_path, _tarball())
    text = lock.read_text(encoding="utf-8")
    lock.write_bytes((text + f"PDFJS_TARBALL_URL={URL}\n").encode())
    first = text.splitlines().index(f"PDFJS_TARBALL_URL={URL}") + 1

    with pytest.raises(PinError, match=f"PDFJS_TARBALL_URL is already set on line {first}"):
        read_lock(lock)


def test_a_missing_lock_fails_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert fetch_pdfjs.main(["--lock", str(tmp_path / "none"), "--dest", str(tmp_path)]) == 1
    assert "fetch_pdfjs:" in capsys.readouterr().err


class _Response:
    """A response whose body arrives in ``chunk``-byte reads; ``error`` is raised by the read
    after the first."""

    def __init__(self, data: bytes, chunk: int = 4096, error: Exception | None = None) -> None:
        self.data = data
        self.chunk = chunk
        self.error = error
        self.reads = 0

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read1(self, limit: int) -> bytes:
        self.reads += 1
        if self.error is not None and self.reads > 1:
            raise self.error
        size = min(limit, self.chunk)
        part, self.data = self.data[:size], self.data[size:]
        return part


def _serve(monkeypatch: pytest.MonkeyPatch, response: _Response) -> list[str]:
    requested: list[str] = []

    def urlopen(url: str, timeout: float) -> _Response:
        requested.append(url)
        assert timeout == fetch_pdfjs.DOWNLOAD_TIMEOUT_S
        return response

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return requested


def test_the_tarball_is_downloaded_from_the_pinned_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tarball = _tarball()
    lock, _ = _write(tmp_path, tarball)
    response = _Response(tarball, chunk=100)
    requested = _serve(monkeypatch, response)

    assert fetch_pdfjs.main(["--lock", str(lock), "--dest", str(tmp_path / "pdfjs")]) == 0
    assert requested == [URL]
    assert response.reads > 2  # read in pieces, then the empty read at the end


def test_an_oversized_download_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, _Response(b"x" * (MAX_TARBALL_BYTES + 10), chunk=1024 * 1024))

    with pytest.raises(PinError, match="over"):
        fetch_pdfjs.download(URL)


def test_a_download_over_the_deadline_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server dripping bytes is cut by the overall deadline, not only by the per-read one."""
    clock = iter([0.0, 100.0, 200.0, 301.0, 400.0])
    monkeypatch.setattr(fetch_pdfjs, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    _serve(monkeypatch, _Response(b"x" * 100, chunk=1))

    with pytest.raises(PinError, match="the download took over 300 s"):
        fetch_pdfjs.download(URL)


@pytest.mark.parametrize(
    "error",
    [http.client.IncompleteRead(b"partial", 100), http.client.RemoteDisconnected("closed")],
)
def test_a_broken_http_exchange_fails_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    lock, _ = _write(tmp_path, _tarball())
    _serve(monkeypatch, _Response(b"x" * 10_000, error=error))

    assert fetch_pdfjs.main(["--lock", str(lock), "--dest", str(tmp_path / "pdfjs")]) == 1

    message = capsys.readouterr().err
    assert message == f"fetch_pdfjs: the download failed ({type(error).__name__})\n"
    assert not (tmp_path / "pdfjs").exists()


def test_only_the_registry_is_downloaded_from() -> None:
    with pytest.raises(PinError, match="not on the npm registry"):
        fetch_pdfjs.download("file:///etc/passwd")


def test_the_real_lock_is_well_formed() -> None:
    pin = read_lock(fetch_pdfjs.DEFAULT_LOCK)

    assert isinstance(pin, Pin)
    assert pin.tarball_url == f"{fetch_pdfjs.REGISTRY_PREFIX}{pin.version}.tgz"
    assert set(pin.file_sha256) == set(FILES)
