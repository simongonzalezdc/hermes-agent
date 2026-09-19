"""Governed Kanban claims run policy and receipt checks in canonical transactions."""

from __future__ import annotations

from pathlib import Path
import json
import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_runtime_policy as policy


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


def test_registered_adapter_reaches_the_standard_dispatch_claim_path(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "_runtime_provider", None)
    policy.register_runtime_provider(_runtime)
    policy.register_runtime_provider(_runtime)
    with pytest.raises(RuntimeError, match="already registered"):
        policy.register_runtime_provider(lambda task: _runtime(task))
    task_id = _create_governed(conn, _contract(tmp_path), title="registered adapter")
    assert kb.claim_task(conn, task_id) is not None


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
