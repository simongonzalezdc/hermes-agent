"""Fail-closed policy checks for governed Kanban claims and delegation receipts.

The Kanban database remains the queue and single writer.  A governed task
stores its active dispatcher binding in ``tasks.admission_policy`` and its
immutable producer binding in ``tasks.creation_admission_policy``.  The
dispatcher supplies a fresh trusted runtime snapshot while ``claim_task``
holds the existing ``BEGIN IMMEDIATE`` transaction.  Ungoverned tasks keep
their historic claim behaviour.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlparse


MAX_UNVERIFIED = 5
ALLOWED_NIP29_ROLES = {"owner", "admin"}
ROLE_FLOORS = {
    "founder": {"floor": 0, "hermes_mode": "off"},
    "verifier": {"floor": 1, "hermes_mode": "smart"},
    "seat": {"floor": 2, "hermes_mode": "manual"},
}
CONTRACT_FIELDS = {
    "seat", "role", "queue_lane", "target_channel", "predicate", "workspace",
    "write_set", "capabilities", "model_override", "provider_override",
    "parent_task_id", "verifier_identity",
}
RECEIPT_FIELDS = {
    "parent_task_id", "child_task_id", "workspace", "write_set", "verifier",
    "terminal_disposition",
}
VERIFIER_FIELDS = {"identity", "result", "evidence"}
TERMINAL_DISPOSITIONS = {"passed", "failed", "blocked", "cancelled"}
RuntimeProvider = Callable[[dict[str, Any]], Mapping[str, Any]]
_runtime_provider: Optional[RuntimeProvider] = None


def register_runtime_provider(provider: RuntimeProvider) -> None:
    """Register the process-wide trusted snapshot adapter exactly once.

    A standalone plugin can call this during startup.  Replacing a live
    provider would change admission authority underneath the dispatcher, so a
    second distinct registration is refused.
    """
    global _runtime_provider
    if not callable(provider):
        raise TypeError("runtime policy provider must be callable")
    if _runtime_provider is not None and _runtime_provider is not provider:
        raise RuntimeError("runtime policy provider is already registered")
    _runtime_provider = provider


def get_runtime_provider() -> Optional[RuntimeProvider]:
    return _runtime_provider


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _deny(code: str, detail: str, *, approval: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    result: dict[str, Any] = {"decision": "deny", "code": code, "detail": detail}
    if approval is not None:
        result["approval"] = approval
    return result


def _validate_predicate(predicate: Any) -> Optional[str]:
    if not isinstance(predicate, dict) or set(predicate) != {"kind", "value"}:
        return "predicate must contain exactly kind and value"
    kind, value = predicate.get("kind"), predicate.get("value")
    if kind not in {"command", "path", "url"} or not _nonempty(value):
        return "predicate requires kind command|path|url and a non-empty value"
    if kind == "url":
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "URL predicate must be an absolute HTTP(S) URL"
    return None


def _normalized_scope(workspace: Any, write_set: Any) -> tuple[Path, tuple[Path, ...]] | str:
    if not _nonempty(workspace):
        return "workspace must be a non-empty absolute path"
    root = Path(workspace).expanduser()
    if not root.is_absolute() or ".." in root.parts:
        return "workspace must be a normalized absolute path"
    root = Path(os.path.normpath(str(root)))
    if not isinstance(write_set, list) or not write_set:
        return "write_set must be a non-empty array"
    normalized: list[Path] = []
    for item in write_set:
        if not _nonempty(item):
            return "write_set entries must be non-empty paths"
        candidate = Path(item).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        if ".." in candidate.parts:
            return "write_set entries may not traverse parents"
        candidate = Path(os.path.normpath(str(candidate)))
        try:
            candidate.relative_to(root)
        except ValueError:
            return "write_set entries must remain inside workspace"
        normalized.append(candidate)
    if len({str(path) for path in normalized}) != len(normalized):
        return "write_set entries must be unique"
    return root, tuple(normalized)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def validate_contract(value: Any) -> dict[str, Any]:
    """Return a JSON-safe dispatcher contract or raise ``ValueError``."""
    if not isinstance(value, dict):
        raise ValueError("admission_policy must be an object")
    extra = set(value) - CONTRACT_FIELDS
    missing = CONTRACT_FIELDS - set(value)
    if extra or missing:
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(sorted(missing)))
        if extra:
            detail.append("unexpected: " + ", ".join(sorted(extra)))
        raise ValueError("invalid admission_policy fields (" + "; ".join(detail) + ")")
    for field in ("seat", "role", "queue_lane", "target_channel", "workspace"):
        if not _nonempty(value.get(field)):
            raise ValueError(f"admission_policy.{field} must be a non-empty string")
    if value["role"] not in ROLE_FLOORS:
        raise ValueError("admission_policy.role is unknown")
    if value["queue_lane"] not in {"ready", "review"}:
        raise ValueError("admission_policy.queue_lane must be ready or review")
    predicate_error = _validate_predicate(value.get("predicate"))
    if predicate_error:
        raise ValueError(predicate_error)
    scope = _normalized_scope(value.get("workspace"), value.get("write_set"))
    if isinstance(scope, str):
        raise ValueError(scope)
    capabilities = value.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or not all(_nonempty(item) for item in capabilities)
        or len(capabilities) != len(set(capabilities))
    ):
        raise ValueError("admission_policy.capabilities must be a unique non-empty string array")
    for field in ("model_override", "provider_override"):
        if value.get(field) is not None and not _nonempty(value.get(field)):
            raise ValueError(f"admission_policy.{field} must be null or a non-empty string")
    if value.get("provider_override") is not None and value.get("model_override") is None:
        raise ValueError("admission_policy.provider_override requires model_override")
    for field in ("parent_task_id", "verifier_identity"):
        if value.get(field) is not None and not _nonempty(value.get(field)):
            raise ValueError(f"admission_policy.{field} must be null or a non-empty string")
    if value.get("parent_task_id") is not None and value.get("verifier_identity") is None:
        raise ValueError("delegated admission_policy requires verifier_identity")
    return json.loads(json.dumps(value, sort_keys=True))


def derive_review_contract(value: Any, reviewer: str) -> dict[str, Any]:
    """Derive the immutable verifier-lane binding for a review handoff."""
    contract = validate_contract(value)
    if not _nonempty(reviewer):
        raise ValueError("governed review requires an explicit reviewer")
    reviewer = reviewer.strip()
    expected = contract.get("verifier_identity")
    if expected is not None and expected != reviewer:
        raise ValueError("reviewer does not match the delegated verifier identity")
    derived = {
        **contract,
        "seat": reviewer,
        "role": "verifier",
        "queue_lane": "review",
        "verifier_identity": reviewer,
    }
    return validate_contract(derived)


def _decoded_capabilities(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def canonical_binding_error(contract: Any, canonical: Any) -> Optional[str]:
    """Compare a policy with the row fields the dispatcher actually consumes."""
    try:
        contract = validate_contract(contract)
    except ValueError as exc:
        return str(exc)
    if canonical is None:
        return "canonical task row is missing"
    expected = {
        "seat": canonical["assignee"],
        "model_override": canonical["model_override"],
        "provider_override": canonical["provider_override"],
        "capabilities": _decoded_capabilities(canonical["capabilities"]),
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            return f"admission_policy.{field} does not match canonical task.{field}"
    scope = _normalized_scope(canonical["workspace_path"], contract["write_set"])
    if isinstance(scope, str):
        return f"canonical workspace_path is invalid: {scope}"
    policy_scope = _normalized_scope(contract["workspace"], contract["write_set"])
    if isinstance(policy_scope, str) or scope[0] != policy_scope[0]:
        return "admission_policy.workspace does not match canonical task.workspace_path"
    return None


def _dispatcher_binding_error(task: dict[str, Any], runtime: Mapping[str, Any]) -> Optional[str]:
    records = runtime.get("dispatcher_tasks")
    if not isinstance(records, list) or not records or not all(isinstance(item, dict) for item in records):
        return "trusted dispatcher_tasks snapshot is required"
    ids = [item.get("task_id") for item in records]
    if not all(_nonempty(item) for item in ids) or len(ids) != len(set(ids)):
        return "dispatcher task IDs must be non-empty and unique"
    matches = [item for item in records if item["task_id"] == task["task_id"]]
    if len(matches) != 1:
        return "task_id was not issued exactly once by the dispatcher"
    if any(matches[0].get(field) != task.get(field) for field in CONTRACT_FIELDS):
        return "task does not match its trusted dispatcher binding"
    return None


def evaluate(task: Any, runtime: Any) -> dict[str, Any]:
    """Evaluate a governed task against a trusted, current runtime snapshot."""
    if not isinstance(task, dict) or not isinstance(runtime, Mapping):
        return _deny("invalid_input", "task and runtime must be objects")
    task = dict(task)
    task_id = task.pop("task_id", None)
    if not _nonempty(task_id):
        return _deny("missing_dispatcher_task_id", "dispatcher-issued task_id is required")
    try:
        task = validate_contract(task)
    except ValueError as exc:
        return _deny("invalid_admission_policy", str(exc))
    task["task_id"] = task_id
    binding_error = _dispatcher_binding_error(task, runtime)
    if binding_error:
        return _deny("invalid_dispatcher_binding", binding_error)
    role = task.get("role")
    roles = runtime.get("role_by_seat")
    if not isinstance(roles, dict) or roles.get(task.get("seat")) != role:
        return _deny("role_binding_mismatch", "seat role does not match the runtime registry")
    floor = ROLE_FLOORS.get(role)
    if floor is None:
        return _deny("unknown_role", "role has no configured approval floor")
    approval = {"role": role, **floor}
    queue_lane = task.get("queue_lane")
    if role == "verifier" and queue_lane != "review":
        return _deny("verifier_lane_required", "verifier work must use review", approval=approval)
    if role != "verifier" and queue_lane == "review":
        return _deny("review_lane_verifier_only", "review requires verifier role", approval=approval)
    predicate_error = _validate_predicate(task.get("predicate"))
    if predicate_error:
        return _deny("invalid_predicate", predicate_error, approval=approval)
    predicate_results = runtime.get("predicate_results")
    predicate_result = predicate_results.get(task["task_id"]) if isinstance(predicate_results, dict) else None
    if (
        not isinstance(predicate_result, dict)
        or predicate_result.get("satisfied") is not True
        or not _nonempty(predicate_result.get("evidence"))
    ):
        return _deny("predicate_unsatisfied", "current predicate evidence is required", approval=approval)
    scope = _normalized_scope(task.get("workspace"), task.get("write_set"))
    if isinstance(scope, str):
        return _deny("invalid_write_scope", scope, approval=approval)
    _, requested_paths = scope

    lane = runtime.get("engine_lane")
    if not isinstance(lane, dict) or not _nonempty(lane.get("id")) or not _nonempty(lane.get("policy_id")):
        return _deny("missing_engine_policy", "effective ENGINE-POLICY lane and identity are required", approval=approval)
    if lane.get("class") not in {"gateway", "subscription", "frontier"}:
        return _deny("unknown_engine_lane_class", "engine lane class is not recognized", approval=approval)
    if lane.get("class") == "frontier" and role != "founder":
        return _deny("frontier_lane_founder_only", "frontier decisions are founder-owned", approval=approval)
    if lane.get("exhausted") is not False:
        return _deny("engine_lane_exhausted", "lane is exhausted or unknown", approval=approval)
    if lane.get("fallback_mode") != "none":
        return _deny("engine_fallback_forbidden", "implicit fallback is forbidden", approval=approval)

    depth = runtime.get("unverified_queue_depth")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        return _deny("unknown_queue_depth", "unverified queue depth must be known", approval=approval)
    if depth > MAX_UNVERIFIED and role != "verifier":
        return _deny("unverified_queue_backpressure", "unverified queue depth exceeds the producer limit", approval=approval)
    in_flight = runtime.get("in_flight_by_seat")
    seat_count = in_flight.get(task["seat"]) if isinstance(in_flight, dict) else None
    if not isinstance(seat_count, int) or isinstance(seat_count, bool) or seat_count < 0:
        return _deny("unknown_in_flight", "trusted in-flight counts are required", approval=approval)
    if seat_count >= 1:
        return _deny("seat_already_in_flight", "seat already has an in-flight task", approval=approval)

    allowed = runtime.get("capabilities_by_seat")
    granted = allowed.get(task["seat"]) if isinstance(allowed, dict) else None
    if not isinstance(granted, list) or not all(_nonempty(item) for item in granted):
        return _deny("unknown_capabilities", "trusted seat capabilities are required", approval=approval)
    missing_capabilities = sorted(set(task["capabilities"]) - set(granted))
    if missing_capabilities:
        return _deny("capability_denied", "seat lacks: " + ", ".join(missing_capabilities), approval=approval)
    approval_modes = runtime.get("approval_mode_by_seat")
    if not isinstance(approval_modes, dict) or approval_modes.get(task["seat"]) != approval["hermes_mode"]:
        return _deny("approval_floor_mismatch", "effective Hermes approval mode does not match the role floor", approval=approval)

    active = runtime.get("active_write_sets")
    if not isinstance(active, list):
        return _deny("unknown_active_write_sets", "trusted active write sets are required", approval=approval)
    for record in active:
        if not isinstance(record, dict):
            return _deny("invalid_active_write_set", "active write-set record is invalid", approval=approval)
        active_scope = _normalized_scope(record.get("workspace"), record.get("write_set"))
        if isinstance(active_scope, str):
            return _deny("invalid_active_write_set", active_scope, approval=approval)
        if any(_paths_overlap(a, b) for a in requested_paths for b in active_scope[1]):
            return _deny("write_set_overlap", "write_set overlaps an active task", approval=approval)

    founder_channels = runtime.get("founder_channel_ids")
    request_context = runtime.get("request_context")
    if (
        not isinstance(founder_channels, list) or not founder_channels
        or not all(_nonempty(item) for item in founder_channels)
        or len(founder_channels) != len(set(founder_channels))
    ):
        return _deny("unknown_founder_channels", "trusted founder channels are required", approval=approval)
    if not isinstance(request_context, dict) or request_context.get("target_channel") != task["target_channel"]:
        return _deny("untrusted_channel_context", "trusted request context is required", approval=approval)
    if task["target_channel"] in founder_channels:
        if role != "verifier" or request_context.get("authenticated_identity") != runtime.get("verifier_identity"):
            return _deny("founder_channel_verifier_only", "configured verifier identity is required", approval=approval)
        if request_context.get("nip29_role") not in ALLOWED_NIP29_ROLES:
            return _deny("founder_channel_role_denied", "verifier requires owner or admin membership", approval=approval)

    return {
        "decision": "eligible", "code": "eligible_for_atomic_claim",
        "approval": approval, "engine_lane": lane["id"],
        "normalized_write_set": [str(path) for path in requested_paths],
    }


def assess_claim(
    conn: sqlite3.Connection,
    task_id: str,
    source_status: str,
    runtime_provider: Optional[RuntimeProvider],
) -> Optional[dict[str, Any]]:
    """Return ``None`` for ungoverned tasks, otherwise an allow/deny assessment.

    Caller holds the Kanban write transaction.  Local running-task checks repeat
    the externally supplied snapshot under SQLite's single-writer lock, closing
    the stale-snapshot race for one-seat and overlapping-write-set claims.
    """
    row = conn.execute(
        "SELECT admission_policy, creation_admission_policy, assignee, "
        "workspace_path, capabilities, model_override, provider_override "
        "FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if row is None:
        return None
    raw = row["admission_policy"]
    if raw is None and row["creation_admission_policy"] is None:
        return None
    if raw is None:
        return _deny("invalid_admission_policy", "governed task is missing its active admission_policy")
    try:
        task = json.loads(raw)
        task = validate_contract(task)
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        return _deny("invalid_admission_policy", str(exc))
    binding_error = canonical_binding_error(task, row)
    if binding_error:
        return _deny("canonical_task_mismatch", binding_error)
    task["task_id"] = task_id
    if task["queue_lane"] != source_status:
        return _deny("queue_lane_mismatch", "stored queue lane does not match the claim transition")
    runtime_provider = runtime_provider or get_runtime_provider()
    if runtime_provider is None:
        return _deny("missing_trusted_runtime", "governed claims require a trusted runtime provider")
    try:
        runtime = runtime_provider(dict(task))
    except Exception as exc:
        return _deny("runtime_provider_failed", f"trusted runtime provider failed: {type(exc).__name__}")
    assessment = evaluate(task, runtime)
    if assessment.get("decision") != "eligible":
        return assessment

    scope = _normalized_scope(task["workspace"], task["write_set"])
    if isinstance(scope, str):
        return _deny("invalid_write_scope", scope)
    for active in conn.execute(
        "SELECT id, admission_policy FROM tasks "
        "WHERE status = 'running' AND id != ? "
        "AND (admission_policy IS NOT NULL OR creation_admission_policy IS NOT NULL)",
        (task_id,),
    ):
        try:
            other = validate_contract(json.loads(active["admission_policy"]))
        except (TypeError, json.JSONDecodeError, ValueError):
            return _deny("invalid_active_admission_policy", f"running task {active['id']} has invalid policy")
        if other["seat"] == task["seat"]:
            return _deny("seat_already_in_flight", "seat already has a canonical running task")
        other_scope = _normalized_scope(other["workspace"], other["write_set"])
        if isinstance(other_scope, str):
            return _deny("invalid_active_write_set", f"running task {active['id']} has invalid scope")
        if any(_paths_overlap(a, b) for a in scope[1] for b in other_scope[1]):
            return _deny("write_set_overlap", f"write_set overlaps running task {active['id']}")
    return {**assessment, "decision": "allow", "code": "admitted"}


def validate_receipt(receipt: Any) -> list[str]:
    if not isinstance(receipt, dict):
        return ["delegation_receipt must be an object"]
    errors: list[str] = []
    extra = set(receipt) - RECEIPT_FIELDS
    missing = RECEIPT_FIELDS - set(receipt)
    if extra:
        errors.append("unexpected receipt fields: " + ", ".join(sorted(extra)))
    if missing:
        errors.append("missing receipt fields: " + ", ".join(sorted(missing)))
    for key in ("parent_task_id", "child_task_id", "workspace"):
        if not _nonempty(receipt.get(key)):
            errors.append(f"{key} must be a non-empty string")
    if receipt.get("parent_task_id") == receipt.get("child_task_id"):
        errors.append("parent_task_id and child_task_id must differ")
    scope = _normalized_scope(receipt.get("workspace"), receipt.get("write_set"))
    if isinstance(scope, str):
        errors.append(scope)
    verifier = receipt.get("verifier")
    if not isinstance(verifier, dict):
        errors.append("verifier must be an object")
    else:
        if set(verifier) != VERIFIER_FIELDS:
            errors.append("verifier must contain exactly identity, result and evidence")
        if not _nonempty(verifier.get("identity")):
            errors.append("verifier.identity must be non-empty")
        if verifier.get("result") not in {"pass", "fail"}:
            errors.append("verifier.result must be pass or fail")
        evidence = verifier.get("evidence")
        if not isinstance(evidence, list) or not evidence or not all(_nonempty(item) for item in evidence):
            errors.append("verifier.evidence must be a non-empty string array")
    disposition = receipt.get("terminal_disposition")
    if disposition not in TERMINAL_DISPOSITIONS:
        errors.append("terminal_disposition is invalid")
    if isinstance(verifier, dict):
        if verifier.get("result") == "pass" and disposition != "passed":
            errors.append("a passing verifier requires terminal_disposition=passed")
        if verifier.get("result") == "fail" and disposition == "passed":
            errors.append("a failed verifier cannot pass")
    return errors


def validate_completion_receipt(
    conn: sqlite3.Connection, task_id: str, metadata: Any,
) -> list[str]:
    """Validate a governed child receipt against its immutable task binding."""
    row = conn.execute(
        "SELECT admission_policy, creation_admission_policy, creator_task_id, current_run_id "
        "FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if row is None:
        return []
    raw = row["admission_policy"]
    if raw is None and row["creation_admission_policy"] is None:
        return []
    if raw is None:
        return ["governed task is missing its active admission_policy"]
    try:
        contract = validate_contract(json.loads(raw))
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        return [f"invalid admission_policy: {exc}"]
    parent_id = contract.get("parent_task_id")
    if parent_id is None:
        return []
    if row["creator_task_id"] != parent_id:
        return ["admission_policy.parent_task_id does not match canonical delegation lineage"]
    if conn.execute("SELECT 1 FROM tasks WHERE id = ?", (parent_id,)).fetchone() is None:
        return ["canonical parent task does not exist"]
    verifier_identity = contract.get("verifier_identity")
    if not _nonempty(verifier_identity):
        return ["delegated task requires a verifier identity"]
    run = conn.execute(
        "SELECT profile, status FROM task_runs WHERE id = ? AND task_id = ?",
        (row["current_run_id"], task_id),
    ).fetchone()
    if (
        contract.get("role") != "verifier"
        or contract.get("queue_lane") != "review"
        or run is None
        or run["profile"] != verifier_identity
        or run["status"] != "running"
    ):
        return ["completion is not owned by the bound canonical verifier review run"]
    receipt = metadata.get("delegation_receipt") if isinstance(metadata, dict) else None
    errors = validate_receipt(receipt)
    if errors:
        return errors
    if not isinstance(receipt, dict):
        return errors
    expected = {
        "parent_task_id": parent_id,
        "child_task_id": task_id,
        "workspace": contract["workspace"],
        "write_set": contract["write_set"],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            errors.append(f"delegation_receipt.{key} does not match the task binding")
    verifier = receipt.get("verifier")
    if isinstance(verifier, dict):
        if verifier.get("identity") != verifier_identity:
            errors.append("delegation_receipt.verifier.identity does not match the task binding")
        if verifier.get("result") != "pass" or receipt.get("terminal_disposition") != "passed":
            errors.append("verification_failed")
    return errors


def prepare_policy_rollback(conn: sqlite3.Connection) -> list[str]:
    """Fence governed cards before running code that predates policy support.

    The persistent claim trigger blocks old claim SQL, but an already-running
    governed worker must be drained first.  Once drained, every potentially
    runnable governed row is parked for explicit operator recovery.
    """
    from hermes_cli.kanban_db import write_txn

    fenced: list[str] = []
    with write_txn(conn):
        running = [row["id"] for row in conn.execute(
            "SELECT id FROM tasks WHERE "
            "(admission_policy IS NOT NULL OR creation_admission_policy IS NOT NULL) "
            "AND status = 'running' ORDER BY id"
        )]
        if running:
            raise RuntimeError("drain governed running tasks before rollback: " + ", ".join(running))
        rows = conn.execute(
            "SELECT id FROM tasks WHERE "
            "(admission_policy IS NOT NULL OR creation_admission_policy IS NOT NULL) "
            "AND status IN ('triage', 'todo', 'scheduled', 'ready', 'review') ORDER BY id"
        ).fetchall()
        for row in rows:
            task_id = row["id"]
            conn.execute(
                "UPDATE tasks SET status = 'blocked', block_kind = 'needs_input', "
                "claim_lock = NULL, claim_expires = NULL WHERE id = ?", (task_id,),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'rollback_fenced', ?, strftime('%s','now'))",
                (task_id, json.dumps({"reason": "governed_runtime_rollback"}, sort_keys=True)),
            )
            fenced.append(task_id)
    return fenced
