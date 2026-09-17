"""Large-payload offloading (issue #254).

An event's identity is the hash of its content, and the content includes the
payload, so offloading must keep the hash over the *recorded* payload while the
row holds only a reference. These tests hold that invariant against every path
that reads events back: plain reads, the chain audit, import/export, the action
index, compaction, and state projection.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from continuum.events import Event, EventType
from continuum.models import Run, RunStatus
from continuum.state.semantic import project
from continuum.storage import CorruptedRecord
from continuum.storage.blobs import (
    DEFAULT_OFFLOAD_THRESHOLD,
    ENV_VAR,
    is_offload_marker,
    offload_threshold,
    payload_digest,
)
from continuum.storage.sqlite import SQLiteStorage

# A payload whose JSON body clears a small threshold, and one that does not.
BIG = {"evidence_id": "ev_big", "summary": "x" * 400, "kind": "log"}
SMALL = {"evidence_id": "ev_small", "summary": "tiny", "kind": "log"}
THRESHOLD = 200


@pytest.fixture
def file_db(tmp_path: Path) -> tuple[Path, SQLiteStorage]:
    """A file-backed store with offloading switched on."""
    path = tmp_path / "offload.db"
    store = SQLiteStorage(path, payload_offload_bytes=THRESHOLD)
    yield path, store
    store.close()


def seed(store: SQLiteStorage, *payloads: dict, run_id: str = "run_1") -> None:
    """Create a run and append one EVIDENCE_ADDED event per payload."""
    store.create_run_started(Run(run_id=run_id, goal="g", status=RunStatus.RUNNING))
    for payload in payloads:
        store.append_event(run_id, EventType.EVIDENCE_ADDED, payload)


def stored_payload(path: Path, sequence: int) -> dict:
    """Read the raw payload column, bypassing rehydration."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT payload FROM events WHERE run_id = ? AND sequence = ?",
            ("run_1", sequence),
        ).fetchone()
        return json.loads(row["payload"])
    finally:
        conn.close()


# --- marker shape and configuration ---------------------------------------- #


def test_threshold_resolution_prefers_explicit_then_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "1024")
    assert offload_threshold(4096) == 4096
    assert offload_threshold(None) == 1024
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert offload_threshold(None) == DEFAULT_OFFLOAD_THRESHOLD
    assert offload_threshold(-5) == 0


def test_unparseable_threshold_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "1MB")
    with pytest.raises(ValueError, match="is not an integer number of bytes"):
        offload_threshold(None)


def test_marker_is_strict_about_shape() -> None:
    assert is_offload_marker({"__offloaded": "d", "keys": ["a"], "bytes": 9})
    # Extra field, coincidental key, wrong types: all data, not a reference.
    assert not is_offload_marker({"__offloaded": "d", "keys": ["a"], "bytes": 9, "extra": 1})
    assert not is_offload_marker({"__offloaded": "d"})
    assert not is_offload_marker({"__offloaded": "d", "keys": ["a"], "bytes": True})
    assert not is_offload_marker({"keys": ["a"], "bytes": 9})


# --- round trip ------------------------------------------------------------ #


def test_small_payload_stays_inline(file_db: tuple[Path, SQLiteStorage]) -> None:
    path, store = file_db
    seed(store, SMALL)
    assert stored_payload(path, 2) == SMALL
    assert not (path.parent / f"{path.name}.blobs").exists()


def test_oversized_payload_round_trips_identically(file_db: tuple[Path, SQLiteStorage]) -> None:
    path, store = file_db
    seed(store, BIG)
    event = store.read_events("run_1")[1]
    assert event.payload == BIG
    # The row holds the reference, the digest still covers the real payload.
    assert is_offload_marker(stored_payload(path, 2))
    assert event.hash == event.digest()
    twin = Event(
        event_id=event.event_id,
        run_id=event.run_id,
        sequence=event.sequence,
        type=event.type,
        timestamp=event.timestamp,
        payload=BIG,
        causer_event_id=event.causer_event_id,
        source=event.source,
        prev_hash=event.prev_hash,
    )
    assert event.digest() == twin.digest()


def test_blob_is_content_addressed_and_deduplicated(file_db: tuple[Path, SQLiteStorage]) -> None:
    path, store = file_db
    seed(store, BIG, BIG, SMALL)
    blobs = sorted(p.name for p in (path.parent / f"{path.name}.blobs").iterdir())
    # Two identical payloads share one blob; the name is the content hash.
    assert blobs == [payload_digest(BIG)]
    assert store.read_events("run_1")[1].payload == BIG
    assert store.read_events("run_1")[2].payload == BIG
    assert store.read_events("run_1")[3].payload == SMALL


def test_memory_database_offloads_to_scratch() -> None:
    store = SQLiteStorage(":memory:", payload_offload_bytes=THRESHOLD)
    scratch = None
    try:
        seed(store, BIG)
        assert store.read_events("run_1")[1].payload == BIG
        # An in-memory database has no directory to sit beside, so offloading
        # would silently no-op without scratch space to write into.
        scratch = store._blobs._blob_root()
        assert scratch.exists()
    finally:
        store.close()
        # Scratch space is released with the store, not leaked into the cwd.
        assert scratch is None or not scratch.exists()


# --- fail-closed reads ----------------------------------------------------- #


def test_missing_blob_is_refused_not_faked(file_db: tuple[Path, SQLiteStorage]) -> None:
    path, store = file_db
    seed(store, BIG)
    blob = next((path.parent / f"{path.name}.blobs").iterdir())
    blob.unlink()
    with pytest.raises(CorruptedRecord, match=blob.name[:16]):
        store.read_events("run_1")


def test_altered_blob_is_refused(file_db: tuple[Path, SQLiteStorage]) -> None:
    path, store = file_db
    seed(store, BIG)
    blob = next((path.parent / f"{path.name}.blobs").iterdir())
    blob.write_bytes(b"tampered")
    with pytest.raises(CorruptedRecord, match="was altered"):
        store.read_events("run_1")


def test_shallow_verify_reports_an_unreadable_blob(file_db: tuple[Path, SQLiteStorage]) -> None:
    path, store = file_db
    seed(store, BIG)
    blob = next((path.parent / f"{path.name}.blobs").iterdir())
    blob.unlink()
    report = store.verify_events("run_1")
    assert not report.ok
    # Shallow mode never claims to have walked the blob store: the failure is
    # the unreadable row, reported because reading is refused, not audited.
    assert [v.kind for v in report.violations] == ["UNREADABLE_RECORD"]
    assert report.blobs_checked == 0


def test_deep_verify_walks_the_blob_store(file_db: tuple[Path, SQLiteStorage]) -> None:
    path, store = file_db
    seed(store, BIG, SMALL)
    report = store.verify_events("run_1", deep=True)
    assert report.ok
    # The healthy blob was examined, not assumed.
    assert report.blobs_checked == 1

    blob = next((path.parent / f"{path.name}.blobs").iterdir())
    blob.write_bytes(b"tampered")
    deep = store.verify_events("run_1", deep=True)
    assert not deep.ok
    assert deep.blobs_checked == 1
    assert deep.violations[0].kind == "BLOB_UNAVAILABLE"
    assert blob.name in deep.violations[0].detail


def test_deep_audit_survives_archived_blobs(file_db: tuple[Path, SQLiteStorage]) -> None:
    path, store = file_db
    seed(store, BIG, SMALL)
    store.compact_run("run_1")
    report = store.verify_events("run_1", deep=True)
    assert report.ok
    # The archived payload's blob is audited from events_archive.
    assert report.blobs_checked >= 1


# --- paths that must not see the reference --------------------------------- #


def test_sealed_appends_rehydrate_on_replay(file_db: tuple[Path, SQLiteStorage]) -> None:
    """Import, export and fork all route through append_sealed (issue #259)."""
    _, store = file_db
    seed(store, BIG)
    sealed = store.read_events("run_1")[1]
    other = SQLiteStorage(":memory:")
    try:
        other.create_run(Run(run_id="run_1", goal="g"))
        other.extend_events(store.read_all_events("run_1"))
        replayed = other.read_events("run_1")[1]
        assert replayed.payload == BIG
        assert replayed.hash == sealed.hash
        assert other.verify_events("run_1").ok
    finally:
        other.close()


def test_projection_folds_the_recorded_payload(file_db: tuple[Path, SQLiteStorage]) -> None:
    _, store = file_db
    seed(store, BIG, {**BIG, "evidence_id": "ev_mid"}, SMALL)
    state = project("run_1", store.read_all_events("run_1"))
    assert [e.evidence_id for e in state.evidence] == ["ev_big", "ev_mid", "ev_small"]


def test_action_index_folds_offloaded_payloads(file_db: tuple[Path, SQLiteStorage]) -> None:
    _, store = file_db
    store.create_run_started(Run(run_id="run_1", goal="g", status=RunStatus.RUNNING))
    big_action = {
        "key": "pay/INV-001",
        "action": {
            "run_id": "run_1",
            "action_id": "a1",
            "action_type": "payment",
            "status": "completed",
            "arguments": {"invoice": "INV-001", "note": "x" * 400},
        },
    }
    store.append_event("run_1", EventType.ACTION_RECORDED, big_action)
    assert store.action_index_drift() == 0
    assert store.foreign_action("pay/INV-001", exclude_run="other") is not None


def test_offloading_disabled_by_default(tmp_path: Path) -> None:
    path = tmp_path / "inline.db"
    store = SQLiteStorage(path)
    try:
        seed(store, BIG)
        assert not is_offload_marker(stored_payload(path, 2))
        assert store.supports_blob_offload is True  # capable, but configured off
    finally:
        store.close()


def test_env_var_enables_offloading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(THRESHOLD))
    path = tmp_path / "env.db"
    store = SQLiteStorage(path)
    try:
        seed(store, BIG)
        assert store.read_events("run_1")[1].payload == BIG
        assert is_offload_marker(stored_payload(path, 2))
    finally:
        store.close()


def test_postgres_refuses_offload_configuration() -> None:
    """Postgres stores payloads inline and says so rather than ignoring the knob."""
    from continuum.storage.postgres import PostgresStorage

    with pytest.raises(ValueError, match="does not support payload offloading"):
        PostgresStorage("postgresql://localhost/continuum", payload_offload_bytes=THRESHOLD)
    assert PostgresStorage.supports_blob_offload is False
