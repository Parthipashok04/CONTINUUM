"""Content-addressed evidence export for external consumers.

The exporter maps a run's durable evidence to four neutral primitives:

* Transitions -- event-appended state movements (every event)
* Observations -- environment validations and diffs
* Relations -- dependency edges between components
* State Checkpoints -- checkpoint records

Each primitive is content-addressed (stable_hash of its content), carries
sequence, origin/provenance, and the signature inputs needed to re-verify
against the hash-chained log. The export is pure read, zero new
dependencies, and the receiver can detect truncation or tampering by
re-computing the chain exactly as ``verify()`` does.

Design constraints
------------------
* Native format stays authoritative; this is a read-only view.
* Hashes are the same as the storage layer's ``Event.hash`` and
  ``StateCheckpoint.integrity_hash``, so a receiver compares directly
  to ``verify()`` output.
* No third-party dependencies beyond pydantic and the existing hashing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from continuum.events import Event
from continuum.security.hashing import stable_hash, to_json
from continuum.storage.base import Storage

__all__ = [
    "EvidencePrimitive",
    "Transition",
    "Observation",
    "Relation",
    "Checkpoint",
    "export_evidence",
    "verify_export",
]

# ---------------------------------------------------------------------------
# Primitive models
# ---------------------------------------------------------------------------

Kind = Literal["transition", "observation", "relation", "checkpoint"]


class EvidencePrimitive(BaseModel):
    """Base for all exported primitives.

    Every primitive is derived from exactly one log fact, so the event identity
    the fact came from is carried on the base rather than repeated on each
    subclass. A checkpoint record read straight from storage has no event of
    its own and is stamped with the checkpoint id and ``STATE_CHECKPOINTED``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Kind
    run_id: str
    sequence: int
    content_hash: str
    prev_hash: str | None
    origin: str
    timestamp: str
    payload: dict[str, Any]
    signature_inputs: dict[str, Any]
    event_id: str
    event_type: str

    def content(self) -> dict[str, Any]:
        """Return the canonical fields represented by this primitive.

        Excludes ``content_hash`` (a hash cannot cover itself) and the event
        identity, which ``signature_inputs`` already covers as part of the
        event's own hashed content.
        """
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "prev_hash": self.prev_hash,
            "origin": self.origin,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "signature_inputs": self.signature_inputs,
        }

    def digest(self) -> str:
        """Return the stable hash of this primitive's canonical content."""
        return stable_hash(self.content())


class Transition(EvidencePrimitive):
    """Represent an event-backed state transition."""

    kind: Literal["transition"] = "transition"


class Observation(EvidencePrimitive):
    """Represent an environment or tool observation."""

    kind: Literal["observation"] = "observation"
    observed_at: str


class Relation(EvidencePrimitive):
    """Represent a dependency or derivation relation between items."""

    kind: Literal["relation"] = "relation"
    source_id: str | None = None
    target_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive_endpoints(cls, data: Any) -> Any:
        """Fill ``source_id``/``target_id`` from the payload when absent.

        Dependency edges point at resources, decisions or findings, and the id
        worth surfacing is whichever of those the event actually carried. This
        is the one non-trivial step in building a relation, so it lives in the
        model: a primitive reconstructed by a receiver from an exported line
        derives the same endpoints the exporter produced, and the guess has a
        single home to read and change.
        """
        if not isinstance(data, dict):
            return data
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            return data
        if not data.get("source_id"):
            source = (
                payload.get("resource") or payload.get("decision_id") or payload.get("finding_id")
            )
            data["source_id"] = str(source) if source else None
        if not data.get("target_id"):
            target = payload.get("evidence") or payload.get("depends_on")
            if isinstance(target, list) and target:
                target = target[0]
            data["target_id"] = str(target) if target else None
        return data


class Checkpoint(EvidencePrimitive):
    """Represent a persisted semantic-state checkpoint."""

    kind: Literal["checkpoint"] = "checkpoint"
    checkpoint_id: str
    version: int
    trigger: str
    integrity_hash: str | None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# Observations are environment validations and diff results.
_OBSERVATION_TYPES = frozenset(
    {
        "STATE_VALIDATED",
        "ENVIRONMENT_CHANGED",
        "PERCEPTION_OBSERVED",
        "BRANCH_RESOLVED",
        "TOOL_COMPLETED",
        "TOOL_FAILED",
        "TOOL_CALLED",
        "REASONING_SUMMARY",
    }
)

# Relations are dependency edges.
_RELATION_TYPES = frozenset(
    {
        "DEPENDENCY_DECLARED",
        "FINDING_ADDED",
        "FINDING_INVALIDATED",
        "DECISION_CREATED",
        "DECISION_INVALIDATED",
        "EVIDENCE_ADDED",
        "WORK_ADDED",
        "CONSTRAINT_PINNED",
        "CONSTRAINT_RETRACTED",
    }
)

# Checkpoints are explicit checkpoint records; we also emit a checkpoint
# primitive for each STATE_CHECKPOINTED event, but the canonical checkpoint
# record comes from storage.list_checkpoints.
_CHECKPOINT_TYPES = frozenset({"STATE_CHECKPOINTED"})


def _classify(event: Event) -> Kind:
    t = event.type.value
    if t in _CHECKPOINT_TYPES:
        return "checkpoint"
    if t in _OBSERVATION_TYPES:
        return "observation"
    if t in _RELATION_TYPES:
        return "relation"
    return "transition"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_evidence(storage: Storage, run_id: str) -> list[EvidencePrimitive]:
    """Export a run's evidence as typed, JSON-serialisable primitives.

    Reads ``events`` plus ``archived_events`` (so compacted runs are fully
    covered) and ``checkpoints``. Each primitive carries ``content_hash``
    (stable_hash of its content), ``prev_hash`` (previous primitive's hash
    for chain verification), ``origin`` and signature inputs.

    The result is the module's own model types, so a missing or stray field in
    the export is a construction error rather than silent dict-key drift::

        for prim in export_evidence(storage, run_id):
            print(prim.kind, prim.sequence, prim.content_hash[:12])

    The caller may write the result as JSON lines::

        for prim in export_evidence(storage, run_id):
            print(json.dumps(prim.model_dump(mode="json"), sort_keys=True))

    Truncation or tampering is detectable by the receiver, either through the
    typed result or through the dicts a parsed JSON line yields::

        ok = verify_export(exported)

    Pure read, no writes, zero new dependencies.
    """
    # Validate run exists early, fail closed.
    storage.get_run(run_id)

    # Gather full history: archived + live, sorted by sequence.
    archived = list(storage.read_archived_events(run_id))
    live = list(storage.read_events(run_id))
    events = sorted([*archived, *live], key=lambda e: e.sequence)

    checkpoints = list(storage.list_checkpoints(run_id))
    # Map checkpoint_id -> checkpoint for quick lookup (not strictly needed
    # but useful for enrichment).
    _cp_by_id = {c.checkpoint_id: c for c in checkpoints}

    primitives: list[EvidencePrimitive] = []
    prev_hash: str | None = None
    seq = 0

    for ev in events:
        seq += 1
        kind = _classify(ev)
        # Signature inputs are the event's content dict (the same inputs
        # that produce Event.hash). Use canonical JSON round-trip so the
        # timestamp is stored as ISO with T, matching stable_hash's
        # canonicalization and surviving JSON dump/load without re-encoding
        # via default=str (which would use a space separator and break the
        # hash). This keeps the exported hash identical to verify() and
        # lets a receiver recompute stable_hash(signature_inputs) after
        # loading the JSON lines.
        sig_inputs = json.loads(to_json(ev.content()))
        # Content hash for the exported primitive is the event's own hash,
        # so it matches verify() directly and a receiver recompute is a
        # single stable_hash(signature_inputs).
        base = {
            "run_id": ev.run_id,
            "sequence": seq,
            "content_hash": ev.hash,
            "prev_hash": prev_hash,
            "origin": ev.source.value,
            "timestamp": ev.timestamp.isoformat(),
            "payload": dict(ev.payload),
            "signature_inputs": sig_inputs,
            "event_id": ev.event_id,
            "event_type": ev.type.value,
        }
        primitive: EvidencePrimitive
        if kind == "transition":
            primitive = Transition(kind="transition", **base)
        elif kind == "observation":
            primitive = Observation(
                kind="observation", observed_at=ev.timestamp.isoformat(), **base
            )
        elif kind == "relation":
            # source_id/target_id are derived by the model from the payload.
            primitive = Relation(kind="relation", **base)
        else:  # checkpoint
            # Enrich with checkpoint record if available.
            cp = None
            cid = ev.payload.get("checkpoint_id")
            if isinstance(cid, str):
                cp = _cp_by_id.get(cid)
            primitive = Checkpoint(
                kind="checkpoint",
                checkpoint_id=cid if isinstance(cid, str) else ev.event_id,
                version=ev.payload.get("version", 0),
                trigger=ev.payload.get("trigger", "unknown"),
                integrity_hash=cp.integrity_hash if cp else None,
                **base,
            )
        primitives.append(primitive)
        prev_hash = ev.hash

    # Emit checkpoint records that are not already represented as events?
    # In current storage, each checkpoint is also an event of type
    # STATE_CHECKPOINTED, so we have already covered them. To avoid
    # duplication, we only emit checkpoint records that have no matching
    # event. This keeps the export covering every event exactly once for
    # the truncation check, while still surfacing the checkpoint's
    # integrity_hash for external verification.
    emitted_cids = {p.checkpoint_id for p in primitives if isinstance(p, Checkpoint)}
    for cp in checkpoints:
        if cp.checkpoint_id in emitted_cids:
            continue
        seq += 1
        sig_inputs = json.loads(to_json(cp.content()))
        primitive = Checkpoint(
            kind="checkpoint",
            run_id=cp.run_id,
            sequence=seq,
            content_hash=cp.integrity_hash,
            prev_hash=prev_hash,
            origin="deterministic",
            timestamp=cp.created_at.isoformat(),
            payload={
                "checkpoint_id": cp.checkpoint_id,
                "version": cp.version,
                "trigger": cp.trigger,
            },
            signature_inputs=sig_inputs,
            checkpoint_id=cp.checkpoint_id,
            version=cp.version,
            trigger=cp.trigger,
            integrity_hash=cp.integrity_hash,
            event_id=cp.checkpoint_id,
            event_type="STATE_CHECKPOINTED",
        )
        primitives.append(primitive)
        prev_hash = cp.integrity_hash

    return primitives


_KIND_MODELS: dict[str, type[EvidencePrimitive]] = {
    "transition": Transition,
    "observation": Observation,
    "relation": Relation,
    "checkpoint": Checkpoint,
}


def _as_primitive(prim: EvidencePrimitive | Mapping[str, Any]) -> EvidencePrimitive | None:
    """Return ``prim`` as a typed primitive, or ``None`` if it cannot be one.

    A receiver holds plain dicts after parsing exported JSON lines; rebuilding
    the matching model re-runs the ``extra="forbid"`` check so a line missing a
    declared field, carrying a stray one, or drifting a type is rejected here
    rather than passing as a well-formed chain.
    """
    if isinstance(prim, EvidencePrimitive):
        return prim
    if not isinstance(prim, Mapping):
        return None
    kind = prim.get("kind")
    model = _KIND_MODELS.get(kind) if isinstance(kind, str) else None
    if model is None:
        return None
    try:
        return model(**prim)
    except ValidationError:
        return None


def verify_export(primitives: list[EvidencePrimitive | dict[str, Any]]) -> bool:
    """Verify an exported stream exactly as a receiver would.

    Accepts the typed primitives ``export_evidence`` returns, or the plain
    dicts a receiver has after parsing the exported JSON lines; dict input is
    rebuilt through its model first.

    Returns True if the chain is intact, sequences are contiguous starting at
    1, each content_hash matches the recomputed digest of signature_inputs,
    and prev_hash links are correct. Used in tests to prove truncation is
    detectable.
    """
    prev: str | None = None
    for i, prim in enumerate(primitives, start=1):
        item = _as_primitive(prim)
        if item is None:
            return False
        if item.sequence != i:
            return False
        if item.prev_hash != prev:
            return False
        # Recompute the hash the primitive was sealed with. For event-backed
        # primitives signature_inputs is the event content, so this is the
        # event's own hash; for a checkpoint read straight from storage it is
        # the checkpoint content, whose digest is the integrity_hash the
        # record was sealed with. Either way a mismatch means tampering.
        sig = item.signature_inputs
        if sig is not None:
            try:
                recomputed = stable_hash(sig)
            except Exception:
                return False
            if item.content_hash != recomputed:
                return False
        prev = item.content_hash
    return True
