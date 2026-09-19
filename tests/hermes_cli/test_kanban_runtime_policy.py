"""Governed Kanban claims run policy and receipt checks in canonical transactions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sqlite3
import threading
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_runtime_policy as policy
from hermes_cli.kanban_runtime_adapter import TRUSTED_RUNTIME_SCHEMA


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _contract(
    workspace: Path,
    *,
    seat: str = "builder",
    role: str = "seat",
    queue_lane: str = "ready",
    write_set: list[str] | None = None,
    capabilities: list[str] | None = None,
    parent_task_id: str | None = None,
    verifier_identity: str | None = None,
    model_override: str | None = "model-a",
    provider_override: str | None = "provider-a",
) -> dict:
    return {
        "seat": seat,
        "role": role,
        "queue_lane": queue_lane,
        "target_channel": "engineering",
        "predicate": {"kind": "command", "value": "pytest -q"},
        "workspace": str(workspace),
        "write_set": write_set or ["src"],
        "capabilities": capabilities or ["terminal"],
        "model_override": model_override,
        "provider_override": provider_override,
        "parent_task_id": parent_task_id,
        "verifier_identity": verifier_identity or ("verifier-1" if parent_task_id else None),
    }


def _create_governed(conn, contract: dict, *, title: str = "governed", **kwargs) -> str:
    return kb.create_task(
        conn,
        title=title,
        assignee=contract["seat"],
        workspace_kind="dir",
        workspace_path=contract["workspace"],
        capabilities=contract["capabilities"],
        model_override=contract["model_override"],
        provider_override=contract["provider_override"],
        creator_task_id=contract["parent_task_id"],
        admission_policy=contract,
        **kwargs,
    )


def _runtime(task: dict, *, capabilities: list[str] | None = None) -> dict:
    binding = {key: task[key] for key in task if key != "task_id"}
    binding["task_id"] = task["task_id"]
    return {
        "dispatcher_tasks": [binding],
        "role_by_seat": {task["seat"]: task["role"]},
        "predicate_results": {
            task["task_id"]: {"satisfied": True, "evidence": "exit=0 at trusted runner"},
        },
        "engine_lane": {
            "id": "free-primary", "policy_id": "engine-policy-v1",
            "class": "gateway", "exhausted": False, "fallback_mode": "none",
        },
        "unverified_queue_depth": 0,
        "in_flight_by_seat": {task["seat"]: 0},
        "active_write_sets": [],
        "capabilities_by_seat": {task["seat"]: capabilities or ["terminal"]},
        "approval_mode_by_seat": {
            task["seat"]: policy.ROLE_FLOORS[task["role"]]["hermes_mode"],
        },
        "founder_channel_ids": ["founder"],
        "request_context": {
            "target_channel": task["target_channel"],
            "authenticated_identity": task["seat"],
            "nip29_role": "member",
        },
        "verifier_identity": "verifier-1",
    }


def _trusted_snapshot(task: dict, *, observed_at: int | None = None) -> dict:
    now = int(time.time()) if observed_at is None else observed_at
    runtime = _runtime(task)
    for field in (
        "dispatcher_tasks", "unverified_queue_depth", "in_flight_by_seat",
        "active_write_sets",
    ):
        runtime.pop(field)
    return {
        "schema": TRUSTED_RUNTIME_SCHEMA,
        "identity": {
            "provider": "owner-runtime-reader/v1",
            "task_id": task["task_id"],
            "lease_id": f"lease:{task['task_id']}",
        },
        "observed_at": now,
        "lease_expires_at": now + 30,
        "runtime": runtime,
    }


def _latest_event(conn, task_id: str, kind: str):
    return [event for event in kb.list_events(conn, task_id) if event.kind == kind][-1]


def test_ungoverned_claim_keeps_existing_behavior(conn):
    task_id = kb.create_task(conn, title="legacy", assignee="builder")
    assert kb.claim_task(conn, task_id) is not None


def test_governed_claim_requires_runtime_and_records_admission(conn, tmp_path):
    contract = _contract(tmp_path)
    task_id = _create_governed(conn, contract)

    assert kb.claim_task(conn, task_id) is None
    assert _latest_event(conn, task_id, "claim_rejected").payload["code"] == "missing_trusted_runtime"

    claimed = kb.claim_task(conn, task_id, runtime_policy_provider=_runtime)
    assert claimed is not None
    event = _latest_event(conn, task_id, "claimed")
    assert event.payload["runtime_policy"] == "admitted"
    assert event.payload["approval"] == {"role": "seat", "floor": 2, "hermes_mode": "manual"}


@pytest.mark.parametrize("active_policy", [None, ""])
def test_creation_policy_marker_prevents_claim_downgrade(conn, tmp_path, active_policy):
    task_id = _create_governed(conn, _contract(tmp_path))
    conn.execute(
        "UPDATE tasks SET admission_policy = ? WHERE id = ?", (active_policy, task_id),
    )

    assert kb.claim_task(conn, task_id, runtime_policy_provider=_runtime) is None
    rejected = _latest_event(conn, task_id, "claim_rejected")
    assert rejected.payload["code"] == "invalid_admission_policy"
    assert kb.get_task(conn, task_id).status == "ready"


def test_registered_adapter_reaches_the_standard_dispatch_claim_path(conn, tmp_path):
    task_id = _create_governed(conn, _contract(tmp_path), title="registered adapter")
    assert policy.get_runtime_provider() is None
    with policy.runtime_provider_scope(_runtime):
        assert policy.get_runtime_provider() is _runtime
        assert kb.claim_task(conn, task_id) is not None
    assert policy.get_runtime_provider() is None


def test_dispatch_registers_fresh_snapshot_only_for_its_tick(conn, tmp_path, monkeypatch):
    task_id = _create_governed(conn, _contract(tmp_path), title="fresh runtime")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    preceding_provider = lambda task: _runtime(task)

    with policy.runtime_provider_scope(preceding_provider):
        result = kbd.dispatch_once(
            conn,
            spawn_fn=lambda _task, _workspace: None,
            runtime_policy_provider=_trusted_snapshot,
        )
        assert policy.get_runtime_provider() is preceding_provider

    assert [item[0] for item in result.spawned] == [task_id]
    assert kb.get_task(conn, task_id).status == "running"
    assert policy.get_runtime_provider() is None

    later = _create_governed(
        conn, _contract(tmp_path / "later", seat="later-builder"), title="later",
    )
    assert kb.claim_task(conn, later) is None
    assert _latest_event(conn, later, "claim_rejected").payload["code"] == "missing_trusted_runtime"


def test_runtime_provider_scopes_do_not_cross_concurrent_dispatch_contexts():
    barrier = threading.Barrier(2)
    providers = [lambda task: {"provider": "one"}, lambda task: {"provider": "two"}]

    def probe(provider):
        assert policy.get_runtime_provider() is None
        with policy.runtime_provider_scope(provider):
            barrier.wait(timeout=5)
            current = policy.get_runtime_provider()
            barrier.wait(timeout=5)
        return current is provider and policy.get_runtime_provider() is None

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert all(executor.map(probe, providers))
    assert policy.get_runtime_provider() is None


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda snapshot, _task: snapshot.update(
                observed_at=int(time.time()) - 31,
                lease_expires_at=int(time.time()) + 1,
            ),
            "stale_trusted_runtime",
        ),
        (
            lambda snapshot, _task: snapshot["runtime"].pop("engine_lane"),
            "malformed_trusted_runtime",
        ),
        (
            lambda snapshot, _task: snapshot["identity"].update(task_id="other-task"),
            "trusted_runtime_identity_mismatch",
        ),
        (
            lambda snapshot, _task: snapshot["runtime"].update(in_flight_by_seat={}),
            "trusted_runtime_derived_field",
        ),
    ],
)
def test_dispatch_refuses_untrusted_snapshot_state(
    conn, tmp_path, monkeypatch, mutate, expected_code,
):
    task_id = _create_governed(conn, _contract(tmp_path), title=expected_code)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)

    def reader(task):
        snapshot = _trusted_snapshot(task)
        mutate(snapshot, task)
        return snapshot

    result = kbd.dispatch_once(
        conn,
        spawn_fn=lambda _task, _workspace: pytest.fail("denied task must not spawn"),
        runtime_policy_provider=reader,
    )

    assert result.spawned == []
    assert kb.get_task(conn, task_id).status == "ready"
    assert _latest_event(conn, task_id, "claim_rejected").payload["code"] == expected_code
    assert policy.get_runtime_provider() is None


def test_review_claim_enforces_the_verifier_floor(conn, tmp_path):
    contract = _contract(tmp_path)
    task_id = _create_governed(conn, contract, title="verify")
    assert kb.request_review(conn, task_id, summary="ready", reviewer="verifier-1") is True
    derived = kb.get_task(conn, task_id)
    assert derived.assignee == "verifier-1"
    assert derived.admission_policy["seat"] == "verifier-1"
    assert derived.admission_policy["role"] == "verifier"
    assert derived.admission_policy["queue_lane"] == "review"

    claimed = kb.claim_review_task(conn, task_id, runtime_policy_provider=_runtime)
    assert claimed is not None
    event = _latest_event(conn, task_id, "claimed")
    assert event.payload["approval"] == {
        "role": "verifier", "floor": 1, "hermes_mode": "smart",
    }

    ok, implementer = kb.request_changes(
        conn, task_id, reason="fix it", expected_run_id=claimed.current_run_id,
    )
    assert (ok, implementer) == (True, "builder")
    restored = kb.get_task(conn, task_id)
    assert restored.assignee == "builder"
    assert restored.admission_policy == contract


def test_capability_governance_fails_closed(conn, tmp_path):
    contract = _contract(tmp_path, capabilities=["terminal", "browser"])
    task_id = _create_governed(conn, contract, title="needs browser")

    provider = lambda task: _runtime(task, capabilities=["terminal"])
    assert kb.claim_task(conn, task_id, runtime_policy_provider=provider) is None
    assert _latest_event(conn, task_id, "claim_rejected").payload["code"] == "capability_denied"


def test_predicate_and_approval_floor_require_current_runtime_evidence(conn, tmp_path):
    task_id = _create_governed(conn, _contract(tmp_path), title="guarded")

    def unsatisfied(task):
        runtime = _runtime(task)
        runtime["predicate_results"][task["task_id"]] = {"satisfied": False, "evidence": "exit=1"}
        return runtime

    assert kb.claim_task(conn, task_id, runtime_policy_provider=unsatisfied) is None
    assert _latest_event(conn, task_id, "claim_rejected").payload["code"] == "predicate_unsatisfied"

    def wrong_floor(task):
        runtime = _runtime(task)
        runtime["approval_mode_by_seat"][task["seat"]] = "off"
        return runtime

    assert kb.claim_task(conn, task_id, runtime_policy_provider=wrong_floor) is None
    assert _latest_event(conn, task_id, "claim_rejected").payload["code"] == "approval_floor_mismatch"


def test_evaluator_rejects_malformed_contract_without_raising(tmp_path):
    task = {**_contract(tmp_path), "task_id": "task-1"}
    runtime = _runtime(task)
    malformed = dict(task)
    malformed.pop("capabilities")
    runtime["dispatcher_tasks"][0].pop("capabilities")

    result = policy.evaluate(malformed, runtime)
    assert result["decision"] == "deny"
    assert result["code"] == "invalid_admission_policy"


def test_evaluator_requires_explicit_in_flight_count_for_target_seat(tmp_path):
    task = {**_contract(tmp_path), "task_id": "task-1"}
    runtime = _runtime(task)
    runtime["in_flight_by_seat"] = {}

    result = policy.evaluate(task, runtime)
    assert result["decision"] == "deny"
    assert result["code"] == "unknown_in_flight"


def test_atomic_claim_rechecks_seat_and_write_set_in_canonical_db(conn, tmp_path):
    first_contract = _contract(tmp_path, seat="builder-a", write_set=["src"])
    second_contract = _contract(tmp_path, seat="builder-b", write_set=["src/module.py"])
    first = _create_governed(conn, first_contract, title="first")
    second = _create_governed(conn, second_contract, title="second")

    assert kb.claim_task(conn, first, runtime_policy_provider=_runtime) is not None
    assert kb.claim_task(conn, second, runtime_policy_provider=_runtime) is None
    assert _latest_event(conn, second, "claim_rejected").payload["code"] == "write_set_overlap"

    third_contract = _contract(tmp_path, seat="builder-a", write_set=["docs"])
    third = _create_governed(conn, third_contract, title="third")
    assert kb.claim_task(conn, third, runtime_policy_provider=_runtime) is None
    assert _latest_event(conn, third, "claim_rejected").payload["code"] == "seat_already_in_flight"


def test_governed_child_completion_requires_bound_delegation_receipt(conn, tmp_path):
    parent = kb.create_task(conn, title="parent")
    contract = _contract(tmp_path, parent_task_id=parent)
    task_id = _create_governed(conn, contract, title="delegated child")
    claimed = kb.claim_task(conn, task_id, runtime_policy_provider=_runtime)
    assert claimed is not None
    assert kb.complete_task(conn, task_id, expected_run_id=claimed.current_run_id) is False
    rejected = _latest_event(conn, task_id, "completion_rejected")
    assert rejected.payload["reason"] == "delegation_receipt"

    receipt = {
        "parent_task_id": parent,
        "child_task_id": task_id,
        "workspace": str(tmp_path),
        "write_set": ["src"],
        "verifier": {
            "identity": "verifier-1", "result": "pass", "evidence": ["tests passed"],
        },
        "terminal_disposition": "passed",
    }
    # A delegated child must complete from the bound verifier's canonical
    # review run, not from its producer run.
    assert kb.request_review(
        conn, task_id, reviewer="verifier-1", expected_run_id=claimed.current_run_id,
    ) is True
    reviewed = kb.claim_review_task(conn, task_id, runtime_policy_provider=_runtime)
    assert reviewed is not None
    assert kb.complete_task(
        conn,
        task_id,
        expected_run_id=reviewed.current_run_id,
        metadata={"delegation_receipt": receipt},
    ) is True


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("assignee", "other-seat"),
        ("workspace_path", "/tmp/other-workspace"),
        ("capabilities", json.dumps(["browser"])),
        ("model_override", "other-model"),
        ("provider_override", "other-provider"),
    ],
)
def test_claim_rejects_policy_drift_from_canonical_dispatch_fields(
    conn, tmp_path, column, value,
):
    contract = _contract(tmp_path)
    task_id = _create_governed(conn, contract, title=f"drift {column}")
    conn.execute(f"UPDATE tasks SET {column} = ? WHERE id = ?", (value, task_id))

    assert kb.claim_task(conn, task_id, runtime_policy_provider=_runtime) is None
    assert _latest_event(conn, task_id, "claim_rejected").payload["code"] == "canonical_task_mismatch"


def test_dispatch_uses_canonical_governed_capabilities(conn, tmp_path, monkeypatch):
    contract = _contract(tmp_path, capabilities=["terminal"])
    task_id = _create_governed(conn, contract, title="bounded tools")
    task = kb.get_task(conn, task_id)
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda _home: ["browser", "web"])

    argv = kbd._worker_argv(task, "builder", None)
    toolsets = argv[argv.index("--toolsets") + 1]
    assert toolsets == "terminal"
    assert argv[argv.index("-m") + 1] == "model-a"
    assert argv[argv.index("--provider") + 1] == "provider-a"


def test_idempotent_governed_create_never_aliases_policy_mismatch(conn, tmp_path):
    key = "same-request"
    legacy = kb.create_task(conn, title="legacy", idempotency_key=key)
    with pytest.raises(ValueError, match="governed and legacy"):
        _create_governed(conn, _contract(tmp_path), idempotency_key=key)
    assert kb.get_task(conn, legacy).admission_policy is None

    governed_key = "governed-request"
    contract = _contract(tmp_path)
    governed = _create_governed(conn, contract, idempotency_key=governed_key)
    assert _create_governed(conn, contract, idempotency_key=governed_key) == governed
    changed = {**contract, "target_channel": "other-channel"}
    with pytest.raises(ValueError, match="admission_policy mismatch"):
        _create_governed(conn, changed, idempotency_key=governed_key)
    with pytest.raises(ValueError, match="governed and legacy"):
        kb.create_task(conn, title="legacy retry", idempotency_key=governed_key)


def test_exact_governed_create_retry_stays_idempotent_during_review(conn, tmp_path):
    key = "review-retry"
    contract = _contract(tmp_path)
    task_id = _create_governed(conn, contract, idempotency_key=key)

    assert kb.request_review(conn, task_id, reviewer="verifier-1") is True
    reviewing = kb.get_task(conn, task_id)
    assert reviewing.status == "review"
    assert reviewing.admission_policy["role"] == "verifier"
    assert reviewing.creation_admission_policy == contract

    assert _create_governed(conn, contract, idempotency_key=key) == task_id
    assert conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?", (key,),
    ).fetchone()[0] == 1


def test_exact_governed_create_retry_stays_idempotent_after_done(conn, tmp_path):
    key = "done-retry"
    contract = _contract(tmp_path)
    task_id = _create_governed(conn, contract, idempotency_key=key)
    assert kb.request_review(conn, task_id, reviewer="verifier-1") is True
    reviewed = kb.claim_review_task(conn, task_id, runtime_policy_provider=_runtime)
    assert reviewed is not None
    assert kb.complete_task(
        conn, task_id, expected_run_id=reviewed.current_run_id,
    ) is True

    completed = kb.get_task(conn, task_id)
    assert completed.status == "done"
    assert completed.admission_policy["role"] == "verifier"
    assert completed.creation_admission_policy == contract
    assert _create_governed(conn, contract, idempotency_key=key) == task_id
    assert conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?", (key,),
    ).fetchone()[0] == 1


def test_migration_recovers_creation_policy_from_review_event(conn, tmp_path):
    key = "migrated-review-retry"
    contract = _contract(tmp_path)
    task_id = _create_governed(conn, contract, idempotency_key=key)
    assert kb.request_review(conn, task_id, reviewer="verifier-1") is True

    # Simulate a governed row written before creation_admission_policy existed.
    conn.execute("DROP TRIGGER trg_creation_admission_policy_immutable")
    conn.execute(
        "UPDATE tasks SET creation_admission_policy = NULL WHERE id = ?", (task_id,),
    )
    kbc._migrate_add_optional_columns(conn)

    migrated = kb.get_task(conn, task_id)
    assert migrated.admission_policy["role"] == "verifier"
    assert migrated.creation_admission_policy == contract
    assert _create_governed(conn, contract, idempotency_key=key) == task_id


def test_creation_policy_is_immutable_after_insert(conn, tmp_path):
    task_id = _create_governed(conn, _contract(tmp_path))

    with pytest.raises(sqlite3.IntegrityError, match="creation admission policy is immutable"):
        conn.execute(
            "UPDATE tasks SET creation_admission_policy = NULL WHERE id = ?", (task_id,),
        )


def test_delegation_policy_binds_to_canonical_creator_and_verifier(conn, tmp_path):
    parent = kb.create_task(conn, title="parent")
    other_parent = kb.create_task(conn, title="other parent")
    contract = _contract(tmp_path, parent_task_id=parent)
    with pytest.raises(ValueError, match="must match creator_task_id"):
        kb.create_task(
            conn,
            title="forged lineage",
            assignee=contract["seat"],
            workspace_kind="dir",
            workspace_path=contract["workspace"],
            capabilities=contract["capabilities"],
            model_override=contract["model_override"],
            provider_override=contract["provider_override"],
            creator_task_id=other_parent,
            admission_policy=contract,
        )

    task_id = _create_governed(conn, contract, title="bound child")
    claimed = kb.claim_task(conn, task_id, runtime_policy_provider=_runtime)
    assert claimed is not None
    ok, reason = kb.request_review(
        conn, task_id, reviewer="wrong-verifier",
        expected_run_id=claimed.current_run_id, with_reason=True,
    )
    assert ok is False
    assert "delegated verifier identity" in reason


def test_failed_verification_cannot_complete_or_release_dependents(conn, tmp_path):
    parent = kb.create_task(conn, title="delegator")
    contract = _contract(tmp_path, parent_task_id=parent)
    child = _create_governed(conn, contract, title="child")
    dependent = kb.create_task(conn, title="dependent", parents=[child])
    producer = kb.claim_task(conn, child, runtime_policy_provider=_runtime)
    assert producer is not None
    assert kb.request_review(
        conn, child, reviewer="verifier-1", expected_run_id=producer.current_run_id,
    ) is True
    verifier = kb.claim_review_task(conn, child, runtime_policy_provider=_runtime)
    assert verifier is not None
    failed_receipt = {
        "parent_task_id": parent,
        "child_task_id": child,
        "workspace": str(tmp_path),
        "write_set": ["src"],
        "verifier": {"identity": "verifier-1", "result": "fail", "evidence": ["red"]},
        "terminal_disposition": "failed",
    }

    assert kb.complete_task(
        conn,
        child,
        expected_run_id=verifier.current_run_id,
        metadata={"delegation_receipt": failed_receipt},
    ) is False
    assert kb.get_task(conn, child).status == "running"
    assert kb.get_task(conn, dependent).status == "todo"
    assert _latest_event(conn, child, "completion_rejected").payload["reason"] == "verification_failed"


def test_valid_active_policy_cannot_strip_immutable_delegation_obligation(conn, tmp_path):
    parent = kb.create_task(conn, title="parent")
    creation_policy = _contract(tmp_path, parent_task_id=parent)
    child = _create_governed(conn, creation_policy, title="delegated child")
    stripped_policy = {
        **creation_policy,
        "parent_task_id": None,
        "verifier_identity": None,
    }
    conn.execute(
        "UPDATE tasks SET admission_policy = ? WHERE id = ?",
        (json.dumps(stripped_policy), child),
    )

    assert kb.complete_task(conn, child) is False
    rejected = _latest_event(conn, child, "completion_rejected")
    assert rejected.payload["reason"] == "delegation_receipt"
    assert any("immutable producer contract" in error for error in rejected.payload["errors"])
    assert kb.get_task(conn, child).status == "ready"


@pytest.mark.parametrize("active_policy", [None, ""])
def test_creation_policy_marker_prevents_receipt_bypass(
    conn, tmp_path, active_policy,
):
    parent = kb.create_task(conn, title="parent")
    child = _create_governed(
        conn, _contract(tmp_path, parent_task_id=parent), title="child",
    )
    conn.execute(
        "UPDATE tasks SET admission_policy = ? WHERE id = ?", (active_policy, child),
    )

    assert kb.complete_task(conn, child) is False
    rejected = _latest_event(conn, child, "completion_rejected")
    assert rejected.payload["reason"] == "delegation_receipt"
    assert kb.get_task(conn, child).status == "ready"


def test_valid_active_write_set_mutation_cannot_hide_running_overlap(conn, tmp_path):
    first_contract = _contract(tmp_path, seat="builder-a", write_set=["src"])
    first = _create_governed(conn, first_contract, title="first")
    assert kb.claim_task(conn, first, runtime_policy_provider=_runtime) is not None
    conn.execute(
        "UPDATE tasks SET admission_policy = ? WHERE id = ?",
        (json.dumps({**first_contract, "write_set": ["docs"]}), first),
    )

    second_contract = _contract(
        tmp_path, seat="builder-b", write_set=["src/module.py"],
    )
    second = _create_governed(conn, second_contract, title="second")
    assert kb.claim_task(conn, second, runtime_policy_provider=_runtime) is None
    rejected = _latest_event(conn, second, "claim_rejected")
    assert rejected.payload["code"] == "invalid_active_admission_policy"
    assert kb.get_task(conn, second).status == "ready"


def test_rollback_requires_drain_and_persistent_trigger_fences_old_claim_sql(conn, tmp_path):
    ready = _create_governed(conn, _contract(tmp_path), title="ready")
    with pytest.raises(sqlite3.IntegrityError, match="fresh admission lease"):
        conn.execute(
            "UPDATE tasks SET status = 'running', claim_lock = 'old-code' "
            "WHERE id = ? AND status = 'ready'", (ready,),
        )

    running_contract = _contract(tmp_path / "other", seat="other-builder")
    running = _create_governed(conn, running_contract, title="running")
    claimed = kb.claim_task(conn, running, runtime_policy_provider=_runtime)
    assert claimed is not None
    with pytest.raises(RuntimeError, match="drain governed running tasks"):
        policy.prepare_policy_rollback(conn)

    assert kb.block_task(
        conn, running, reason="drained for rollback", kind="needs_input",
        expected_run_id=claimed.current_run_id,
    ) is True
    assert policy.prepare_policy_rollback(conn) == [ready]
    assert kb.get_task(conn, ready).status == "blocked"
    assert _latest_event(conn, ready, "rollback_fenced").payload == {
        "reason": "governed_runtime_rollback",
    }


def test_creation_policy_marker_survives_active_policy_loss_for_rollback(conn, tmp_path):
    task_id = _create_governed(conn, _contract(tmp_path), title="marker only")
    conn.execute(
        "UPDATE tasks SET admission_policy = NULL WHERE id = ?", (task_id,),
    )

    with pytest.raises(sqlite3.IntegrityError, match="fresh admission lease"):
        conn.execute(
            "UPDATE tasks SET status = 'running', claim_lock = 'old-code' "
            "WHERE id = ? AND status = 'ready'", (task_id,),
        )

    assert policy.prepare_policy_rollback(conn) == [task_id]
    assert kb.get_task(conn, task_id).status == "blocked"
