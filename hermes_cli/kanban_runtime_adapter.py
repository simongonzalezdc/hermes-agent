"""Trusted runtime adapter for governed Kanban claims.

The owner supplies a fresh, authenticated policy snapshot reader.  This
adapter validates its task identity and lease, then derives mutable queue
facts from the claim transaction's existing SQLite connection.  It owns no
queue, writer, file path, or process-global fallback.
"""

from __future__ import annotations

import copy
import sqlite3
import time
from typing import Any, Callable, Mapping

from hermes_cli.kanban_runtime_policy import RuntimeProvider, RuntimeSnapshotError
from hermes_cli.kanban_runtime_policy import _lifecycle_contracts


TRUSTED_RUNTIME_SCHEMA = "hermes-kanban-trusted-runtime/v1"
MAX_SNAPSHOT_AGE_SECONDS = 30
MAX_LEASE_DURATION_SECONDS = 60
TrustedRuntimeReader = Callable[[dict[str, Any]], Mapping[str, Any]]

_ENVELOPE_FIELDS = {
    "schema", "identity", "observed_at", "lease_expires_at", "runtime",
}
_IDENTITY_FIELDS = {"provider", "task_id", "lease_id"}
_POLICY_RUNTIME_FIELDS = {
    "role_by_seat", "predicate_results", "engine_lane",
    "capabilities_by_seat", "approval_mode_by_seat", "founder_channel_ids",
    "request_context", "verifier_identity",
}
_QUEUE_DERIVED_FIELDS = {
    "dispatcher_tasks", "unverified_queue_depth", "in_flight_by_seat",
    "active_write_sets",
}


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _exact_object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeSnapshotError("malformed_trusted_runtime", f"{name} must be an object")
    missing = fields - set(value)
    extra = set(value) - fields
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected: " + ", ".join(sorted(extra)))
        raise RuntimeSnapshotError(
            "malformed_trusted_runtime",
            f"{name} fields are invalid ({'; '.join(details)})",
        )
    return value


def _validate_envelope(
    task: dict[str, Any], value: Any, *, now: int,
) -> dict[str, Any]:
    envelope = _exact_object(value, _ENVELOPE_FIELDS, "trusted runtime snapshot")
    if envelope["schema"] != TRUSTED_RUNTIME_SCHEMA:
        raise RuntimeSnapshotError(
            "malformed_trusted_runtime", "trusted runtime schema is unsupported",
        )
    identity = _exact_object(
        envelope["identity"], _IDENTITY_FIELDS, "trusted runtime identity",
    )
    if not _nonempty(identity["provider"]) or not _nonempty(identity["lease_id"]):
        raise RuntimeSnapshotError(
            "malformed_trusted_runtime", "trusted runtime provider and lease_id are required",
        )
    if identity["task_id"] != task.get("task_id"):
        raise RuntimeSnapshotError(
            "trusted_runtime_identity_mismatch",
            "trusted runtime task identity does not match the claim",
        )
    observed_at = envelope["observed_at"]
    expires_at = envelope["lease_expires_at"]
    if (
        not isinstance(observed_at, int) or isinstance(observed_at, bool)
        or not isinstance(expires_at, int) or isinstance(expires_at, bool)
    ):
        raise RuntimeSnapshotError(
            "malformed_trusted_runtime", "trusted runtime lease timestamps must be integer epochs",
        )
    if (
        observed_at > now
        or now - observed_at > MAX_SNAPSHOT_AGE_SECONDS
        or expires_at <= now
        or expires_at <= observed_at
        or expires_at - observed_at > MAX_LEASE_DURATION_SECONDS
    ):
        raise RuntimeSnapshotError(
            "stale_trusted_runtime",
            "trusted runtime snapshot lease is stale, future-dated, or overlong",
        )
    runtime = envelope["runtime"]
    if not isinstance(runtime, dict):
        raise RuntimeSnapshotError(
            "malformed_trusted_runtime", "trusted runtime policy must be an object",
        )
    supplied_derived = set(runtime) & _QUEUE_DERIVED_FIELDS
    if supplied_derived:
        raise RuntimeSnapshotError(
            "trusted_runtime_derived_field",
            "trusted reader may not supply queue-derived fields: "
            + ", ".join(sorted(supplied_derived)),
        )
    return copy.deepcopy(_exact_object(runtime, _POLICY_RUNTIME_FIELDS, "trusted runtime policy"))


def _queue_runtime(conn: sqlite3.Connection, task: dict[str, Any]) -> dict[str, Any]:
    counts = {
        row["assignee"]: int(row["count"])
        for row in conn.execute(
            "SELECT assignee, COUNT(*) AS count FROM tasks "
            "WHERE status = 'running' AND assignee IS NOT NULL GROUP BY assignee"
        )
    }
    counts.setdefault(task["seat"], 0)

    active_write_sets: list[dict[str, Any]] = []
    unverified_depth = 0
    rows = conn.execute(
        "SELECT id, status, admission_policy, creation_admission_policy FROM tasks "
        "WHERE status IN ('running', 'review') "
        "AND (admission_policy IS NOT NULL OR creation_admission_policy IS NOT NULL)"
    ).fetchall()
    for row in rows:
        active, _, lifecycle_error = _lifecycle_contracts(
            row["admission_policy"], row["creation_admission_policy"],
        )
        if lifecycle_error or active is None:
            raise RuntimeSnapshotError(
                "invalid_queue_runtime",
                f"governed task {row['id']} has invalid lifecycle policy: {lifecycle_error}",
            )
        if row["status"] == "review" or active["role"] == "verifier":
            unverified_depth += 1
        if row["status"] == "running":
            active_write_sets.append({
                "task_id": row["id"],
                "workspace": active["workspace"],
                "write_set": list(active["write_set"]),
            })

    return {
        "dispatcher_tasks": [copy.deepcopy(task)],
        "unverified_queue_depth": unverified_depth,
        "in_flight_by_seat": counts,
        "active_write_sets": active_write_sets,
    }


def build_trusted_runtime_provider(
    conn: sqlite3.Connection,
    reader: TrustedRuntimeReader,
    *,
    clock: Callable[[], float] = time.time,
) -> RuntimeProvider:
    """Adapt one owner-supplied reader to the claim policy provider contract."""
    if not callable(reader):
        raise TypeError("trusted runtime reader must be callable")

    def _provider(task: dict[str, Any]) -> Mapping[str, Any]:
        policy_runtime = _validate_envelope(
            task, reader(copy.deepcopy(task)), now=int(clock()),
        )
        return {**policy_runtime, **_queue_runtime(conn, task)}

    return _provider
