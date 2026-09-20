"""Read the fixed owner-authenticated runtime snapshot for governed claims.

The producer is external to Hermes.  It atomically replaces one private file
after collecting the five authoritative owner inputs.  This consumer accepts
no caller-selected path and treats file ownership, private mode, link count,
fresh per-source evidence, and content digests as one fail-closed boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Callable

from hermes_cli.kanban_runtime_adapter import TRUSTED_RUNTIME_SCHEMA
from hermes_cli.kanban_runtime_policy import RuntimeSnapshotError

OWNER_SNAPSHOT_PATH = "/var/run/kyanite/hermes-kanban-owner-runtime.json"
OWNER_SNAPSHOT_SCHEMA = "hermes-kanban-owner-snapshot/v1"
OWNER_SNAPSHOT_PROVIDER = "hermes-kanban-owner-snapshot-reader/v1"
MAX_FILE_BYTES = 1_000_000
MAX_SOURCE_AGE_SECONDS = 30
_HEX_DIGEST_LENGTH = 64
_TOP_LEVEL_FIELDS = {
    "schema",
    "identity",
    "observed_at",
    "lease_expires_at",
    "sources",
    "runtime",
    "snapshot_digest",
}
_IDENTITY_FIELDS = {"provider", "task_id", "lease_id"}
_SOURCE_FIELDS = {"owner", "source_id", "observed_at", "record_digest"}
_SOURCE_BINDINGS = {
    "predicate": ("predicate_results",),
    "engine_policy": ("engine_lane",),
    "role_registry": (
        "role_by_seat",
        "capabilities_by_seat",
        "approval_mode_by_seat",
        "verifier_identity",
    ),
    "founder_registry": ("founder_channel_ids",),
    "buzz_request": ("request_context",),
}


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeSnapshotError(
            "malformed_owner_snapshot", f"{name} must be an object"
        )
    missing = fields - set(value)
    extra = set(value) - fields
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected: " + ", ".join(sorted(extra)))
        raise RuntimeSnapshotError(
            "malformed_owner_snapshot",
            f"{name} fields are invalid ({'; '.join(details)})",
        )
    return value


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _private_parent(path: str) -> None:
    try:
        info = Path(path).parent.lstat()
    except OSError as exc:
        raise RuntimeSnapshotError(
            "missing_owner_snapshot",
            "owner snapshot directory is unavailable",
        ) from exc
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeSnapshotError(
            "unsafe_owner_snapshot",
            "owner snapshot parent must be a directory",
        )
    if info.st_uid not in {0, os.geteuid()} or stat.S_IMODE(info.st_mode) & 0o022:
        raise RuntimeSnapshotError(
            "unsafe_owner_snapshot",
            "owner snapshot parent must be process/root owned and not group/world writable",
        )


def _read_private_json(path: str) -> dict[str, Any]:
    if not os.path.isabs(path):
        raise RuntimeSnapshotError(
            "unsafe_owner_snapshot",
            "owner snapshot path must be absolute",
        )
    _private_parent(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeSnapshotError(
            "missing_owner_snapshot",
            "owner snapshot is unavailable",
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeSnapshotError(
                "unsafe_owner_snapshot",
                "owner snapshot must be a regular file",
            )
        if info.st_uid not in {0, os.geteuid()}:
            raise RuntimeSnapshotError(
                "unsafe_owner_snapshot",
                "owner snapshot has an untrusted owner",
            )
        if stat.S_IMODE(info.st_mode) not in {0o400, 0o600}:
            raise RuntimeSnapshotError(
                "unsafe_owner_snapshot",
                "owner snapshot mode must be 0400 or 0600",
            )
        if info.st_nlink != 1:
            raise RuntimeSnapshotError(
                "unsafe_owner_snapshot",
                "owner snapshot must have one hard link",
            )
        if info.st_size <= 0 or info.st_size > MAX_FILE_BYTES:
            raise RuntimeSnapshotError(
                "unsafe_owner_snapshot",
                "owner snapshot size is outside the accepted bound",
            )
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            value = json.load(stream)
    except RuntimeSnapshotError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeSnapshotError(
            "malformed_owner_snapshot",
            "owner snapshot is not valid UTF-8 JSON",
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise RuntimeSnapshotError(
            "malformed_owner_snapshot",
            "owner snapshot must be an object",
        )
    return value


def _integer_epoch(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeSnapshotError(
            "malformed_owner_snapshot",
            f"{name} must be an integer epoch",
        )
    return value


def read_owner_snapshot(
    task: dict[str, Any],
    *,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Return the existing trusted-runtime envelope for one exact task.

    The keyword-only clock is solely a deterministic test seam.  The file path
    remains fixed and cannot be selected by a dispatch caller.
    """
    snapshot = _exact_object(
        _read_private_json(OWNER_SNAPSHOT_PATH),
        _TOP_LEVEL_FIELDS,
        "owner snapshot",
    )
    if snapshot["schema"] != OWNER_SNAPSHOT_SCHEMA:
        raise RuntimeSnapshotError(
            "malformed_owner_snapshot",
            "owner snapshot schema is unsupported",
        )
    identity = _exact_object(
        snapshot["identity"], _IDENTITY_FIELDS, "owner snapshot identity"
    )
    if identity["provider"] != OWNER_SNAPSHOT_PROVIDER or not _nonempty(
        identity["lease_id"]
    ):
        raise RuntimeSnapshotError(
            "malformed_owner_snapshot",
            "owner snapshot provider or lease is invalid",
        )
    if identity["task_id"] != task.get("task_id"):
        raise RuntimeSnapshotError(
            "owner_snapshot_identity_mismatch",
            "owner snapshot belongs to another task",
        )

    now = int(clock())
    observed_at = _integer_epoch(snapshot["observed_at"], "observed_at")
    lease_expires_at = _integer_epoch(snapshot["lease_expires_at"], "lease_expires_at")
    runtime = snapshot["runtime"]
    if not isinstance(runtime, dict):
        raise RuntimeSnapshotError(
            "malformed_owner_snapshot",
            "owner snapshot runtime must be an object",
        )
    sources = _exact_object(
        snapshot["sources"], set(_SOURCE_BINDINGS), "owner snapshot sources"
    )
    for source_name, runtime_fields in _SOURCE_BINDINGS.items():
        source = _exact_object(
            sources[source_name],
            _SOURCE_FIELDS,
            f"owner snapshot source {source_name}",
        )
        if not _nonempty(source["owner"]) or not _nonempty(source["source_id"]):
            raise RuntimeSnapshotError(
                "malformed_owner_snapshot",
                f"{source_name} provenance is incomplete",
            )
        source_observed = _integer_epoch(
            source["observed_at"], f"{source_name}.observed_at"
        )
        if (
            source_observed > observed_at
            or now - source_observed > MAX_SOURCE_AGE_SECONDS
        ):
            raise RuntimeSnapshotError(
                "stale_owner_source",
                f"{source_name} owner input is stale or future-dated",
            )
        digest = source["record_digest"]
        if (
            not isinstance(digest, str)
            or len(digest) != _HEX_DIGEST_LENGTH
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise RuntimeSnapshotError(
                "malformed_owner_snapshot",
                f"{source_name} digest is invalid",
            )
        bound_record = {field: runtime.get(field) for field in runtime_fields}
        if any(field not in runtime for field in runtime_fields) or digest != _digest(
            bound_record
        ):
            raise RuntimeSnapshotError(
                "owner_source_digest_mismatch",
                f"{source_name} digest does not bind its runtime fields",
            )

    expected_digest = _digest(
        {key: snapshot[key] for key in _TOP_LEVEL_FIELDS - {"snapshot_digest"}}
    )
    if snapshot["snapshot_digest"] != expected_digest:
        raise RuntimeSnapshotError(
            "owner_snapshot_digest_mismatch",
            "owner snapshot digest does not match its content",
        )
    return {
        "schema": TRUSTED_RUNTIME_SCHEMA,
        "identity": dict(identity),
        "observed_at": observed_at,
        "lease_expires_at": lease_expires_at,
        "runtime": runtime,
    }
