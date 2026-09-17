"""Content-addressed offloading of oversized event payloads (issue #254).

Event payloads above a configurable size (gateway evidence bodies, reasoning
summaries, OTel attribute bundles) would bloat both the live log and the
archive, and every replay pays to re-read them. Durable-execution platforms
solve this with a payload codec: store the bytes out of band and keep a digest
plus reference inline (Temporal's large-payload codec is the reference
pattern). This module is that codec for CONTINUUM.

How it fits the integrity model
-------------------------------
An event's identity is the hash of its content, and the content includes the
payload. Offloading therefore keeps the hash over the *recorded* payload and
stores only a reference in the row::

    {"__offloaded": <sha256>, "keys": [<original top-level keys>], "bytes": <n>}

The blob file ``<db>.blobs/<sha256>`` holds the canonical JSON bytes, so the
digest is the file's name and its content hash at once. Reading an event
rehydrates the original payload before returning it, which means the chain,
``verify``, export/import and forking all behave exactly as if the payload had
been stored inline. A caller never has to know offloading is in play, and a
projection never folds a reference in place of the fact it stands for.

Reading is fail-closed
----------------------
``CorruptedRecord`` is raised, naming the digest, when a blob is missing or no
longer hashes to its recorded digest. Returning a substitute (an empty payload,
or the reference itself) would hand a caller state that cannot be shown to be
the recorded state, which is precisely what ``CorruptedRecord`` exists to
refuse. The chain audit would eventually catch an altered blob as a
``TAMPERED_CONTENT`` violation, but a plain ``read_events`` should not wait for
an audit to refuse untrusted data.

Compaction and deletion
-----------------------
Blobs are content-addressed and immutable, so compaction moves nothing: the
reference travels verbatim into ``events_archive`` with its row, and the blob is
found by digest from either table. Blob lifetime is deliberately operator-owned;
nothing here deletes one, because the same payload may be referenced from a
run the operator still wants to inspect.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from continuum.security.hashing import hash_content, to_json
from continuum.storage.base import CorruptedRecord

__all__ = [
    "OFFLOAD_KEY",
    "ENV_VAR",
    "DEFAULT_OFFLOAD_THRESHOLD",
    "offload_threshold",
    "payload_digest",
    "is_offload_marker",
    "BlobStore",
    "FileBlobStore",
]

#: Inline reference key. Chosen to be implausible as a real payload field name
#: while remaining valid JSON, so a reference is distinguishable from data
#: without a side table.
OFFLOAD_KEY = "__offloaded"

#: The two fields that ride along with a reference: the original top-level keys
#: (so a reader can describe the payload without opening the blob) and the byte
#: count moved out of the row.
KEYS_KEY = "keys"
BYTES_KEY = "bytes"

_MARKER_KEYS = frozenset({OFFLOAD_KEY, KEYS_KEY, BYTES_KEY})

#: Configuration key for the threshold in bytes. Default 0 disables offloading.
ENV_VAR = "CONTINUUM_PAYLOAD_OFFLOAD_BYTES"

#: 0 means every payload is stored inline, exactly as before the feature existed.
DEFAULT_OFFLOAD_THRESHOLD = 0


def offload_threshold(explicit: int | None = None, *, env: Mapping[str, str] | None = None) -> int:
    """Resolve the payload offload threshold in bytes, never negative.

    Order: an explicit argument, then ``CONTINUUM_PAYLOAD_OFFLOAD_BYTES``, then
    the default. Zero or a negative value disables offloading. A set but
    unparseable value raises instead of silently disabling a durability feature
    the operator believes is on: a typo'd ``CONTINUUM_PAYLOAD_OFFLOAD_BYTES=1MB``
    must not quietly turn off payload offloading.
    """
    if explicit is not None:
        raw: str | None = str(explicit)
    else:
        raw = (env if env is not None else os.environ).get(ENV_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_OFFLOAD_THRESHOLD
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{ENV_VAR}={raw!r} is not an integer number of bytes; set it to 0 "
            f"(or unset it) to store payloads inline"
        ) from exc
    return max(0, value)


def payload_digest(payload: Mapping[str, Any]) -> str:
    """SHA-256 of the payload's canonical JSON, which is also its blob's name."""
    return hash_content(to_json(payload).encode("utf-8"))


def is_offload_marker(payload: Any) -> bool:
    """True when ``payload`` is an offload reference rather than recorded data.

    Strict about shape: exactly the three reference fields, with a string
    digest, an integer byte count and a list of string keys. A payload that
    merely contains an ``__offloaded`` key alongside anything else is data and
    is returned untouched, so the check cannot misread a coincidental field
    name as a reference.
    """
    return (
        isinstance(payload, Mapping)
        and frozenset(payload) == _MARKER_KEYS
        and isinstance(payload[OFFLOAD_KEY], str)
        and isinstance(payload[BYTES_KEY], int)
        and not isinstance(payload[BYTES_KEY], bool)
        and isinstance(payload[KEYS_KEY], list)
        and all(isinstance(key, str) for key in payload[KEYS_KEY])
    )


class BlobStore:
    """Out-of-band storage for event payloads too large to keep in a row."""

    def maybe_offload(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return the inline reference for ``payload``, writing the blob first.

        ``None`` means store the payload inline: offloading is disabled, or the
        payload is within the threshold. The caller keeps the original payload
        for hashing; only the stored column holds the reference.

        A payload that is already a reference is never re-offloaded. That case
        reaches a store through import or fork, and the blob it points at is the
        one that must be read back, not a blob of the reference itself.
        """
        raise NotImplementedError

    def rehydrate(self, payload: Any) -> dict[str, Any]:
        """Return the recorded payload behind a reference.

        Anything that is not a reference is returned unchanged, so a caller can
        route every payload through this without branching. Raises
        ``CorruptedRecord`` naming the digest when the blob is gone or no longer
        matches it.
        """
        raise NotImplementedError

    def audit(self, payload: Any) -> tuple[bool, str]:
        """Non-raising integrity check of one reference (issue #254 deep verify).

        ``(True, "")`` for a payload stored inline or a reference whose blob
        exists and hashes to its recorded digest; ``(False, detail)`` otherwise,
        with the digest in the detail so an operator can find the file.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release any scratch space this store owns. Idempotent."""
        raise NotImplementedError


class FileBlobStore(BlobStore):
    """Blobs as files named by digest, alongside the database they serve.

    For a file database the root is ``<database>.blobs``. An in-memory database
    has no directory to sit next to, so it gets a scratch directory created on
    first use and removed on :meth:`close`; without that fallback, enabling
    offloading on an in-memory store would silently store every payload inline.
    """

    def __init__(self, root: Path | None, threshold: int) -> None:
        self._root = root
        self._threshold = max(0, threshold)
        self._scratch: Path | None = None

    @classmethod
    def for_database(cls, path: str, threshold: int) -> FileBlobStore:
        """Root at ``<path>.blobs``, or scratch space for an in-memory database."""
        if path in ("", ":memory:"):
            return cls(None, threshold)
        return cls(Path(f"{path}.blobs"), threshold)

    @property
    def enabled(self) -> bool:
        """Whether payloads above the threshold are moved out of the rows."""
        return self._threshold > 0

    @property
    def root(self) -> Path | None:
        """The blob directory, or None when nothing has been offloaded yet."""
        return self._root

    def _blob_root(self) -> Path:
        if self._root is not None:
            return self._root
        if self._scratch is None:
            # Removed on close, so an unclosed in-memory store leaks a temp dir
            # rather than leaking payloads into the working directory.
            self._scratch = Path(tempfile.mkdtemp(prefix="continuum-blobs-"))
        return self._scratch

    def _blob_path(self, digest: str) -> Path:
        return self._blob_root() / digest

    def _write_blob(self, digest: str, body: bytes) -> None:
        """Persist ``body`` under ``digest`` atomically, or leave the copy alone.

        An existing file with the same name already holds the same bytes, since
        the name is the content hash, so it is left untouched. Writing through a
        temporary name and renaming means a reader never observes a half-written
        blob, and the fsync matches what the SQLite engine already pays per
        append: a payload that survives in the row but not on disk would
        reintroduce the exact data loss offloading exists to avoid.
        """
        root = self._blob_root()
        root.mkdir(parents=True, exist_ok=True)
        final = root / digest
        if final.exists():
            return
        staging = root / f".{digest}.tmp"
        with open(staging, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, final)

    def _read_blob(self, digest: str) -> bytes:
        """Read and digest-verify one blob, refusing anything untrustworthy."""
        path = self._blob_path(digest)
        try:
            body = path.read_bytes()
        except FileNotFoundError as exc:
            raise CorruptedRecord(
                f"payload blob {digest} is missing: the event references a blob "
                f"that is not in {self._blob_root()}"
            ) from exc
        except OSError as exc:
            raise CorruptedRecord(f"payload blob {digest} could not be read: {exc}") from exc
        actual = hash_content(body)
        if actual != digest:
            raise CorruptedRecord(
                f"payload blob {digest} was altered: its content now hashes to {actual}"
            )
        return body

    def maybe_offload(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        if not self.enabled or is_offload_marker(payload):
            return None
        body = to_json(payload).encode("utf-8")
        if len(body) <= self._threshold:
            return None
        digest = hash_content(body)
        self._write_blob(digest, body)
        return {
            OFFLOAD_KEY: digest,
            KEYS_KEY: sorted(payload),
            BYTES_KEY: len(body),
        }

    def rehydrate(self, payload: Any) -> dict[str, Any]:
        if not is_offload_marker(payload):
            return dict(payload) if isinstance(payload, Mapping) else payload
        digest = payload[OFFLOAD_KEY]
        body = self._read_blob(digest)
        try:
            restored: dict[str, Any] = json.loads(body)
        except json.JSONDecodeError as exc:
            raise CorruptedRecord(
                f"payload blob {digest} is not the canonical JSON it was written as: {exc}"
            ) from exc
        return restored

    def audit(self, payload: Any) -> tuple[bool, str]:
        if not is_offload_marker(payload):
            return True, ""
        digest = payload[OFFLOAD_KEY]
        try:
            self._read_blob(digest)
        except CorruptedRecord as exc:
            return False, str(exc)
        return True, ""

    def close(self) -> None:
        """Remove the scratch directory, if this store owns one. Idempotent."""
        if self._scratch is None:
            return
        shutil.rmtree(self._scratch, ignore_errors=True)
        self._scratch = None
