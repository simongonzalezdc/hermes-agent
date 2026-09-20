from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_runtime_owner_snapshot as owner
from hermes_cli.kanban_runtime_policy import RuntimeSnapshotError


def _runtime() -> dict:
    return {
        "role_by_seat": {"builder": "seat"},
        "predicate_results": {"task-1": {"satisfied": True, "evidence": "receipt"}},
        "engine_lane": {
            "id": "primary",
            "policy_id": "policy-v1",
            "class": "gateway",
            "exhausted": False,
            "fallback_mode": "none",
        },
        "capabilities_by_seat": {"builder": ["terminal"]},
        "approval_mode_by_seat": {"builder": "manual"},
        "founder_channel_ids": ["founder-channel"],
        "request_context": {
            "target_channel": "engineering",
            "authenticated_identity": "builder",
            "nip29_role": "member",
        },
        "verifier_identity": "verifier",
    }


def _snapshot(*, source_observed_at: int = 1_000) -> dict:
    runtime = _runtime()
    sources = {}
    for name, fields in owner._SOURCE_BINDINGS.items():
        sources[name] = {
            "owner": f"{name}-owner",
            "source_id": f"{name}-record",
            "observed_at": source_observed_at,
            "record_digest": owner._digest({field: runtime[field] for field in fields}),
        }
    value = {
        "schema": owner.OWNER_SNAPSHOT_SCHEMA,
        "identity": {
            "provider": owner.OWNER_SNAPSHOT_PROVIDER,
            "task_id": "task-1",
            "lease_id": "lease-1",
        },
        "observed_at": 1_000,
        "lease_expires_at": 1_030,
        "sources": sources,
        "runtime": runtime,
    }
    value["snapshot_digest"] = owner._digest(value)
    return value


def _write(monkeypatch, tmp_path, value: dict):
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    path = parent / "snapshot.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setattr(owner, "OWNER_SNAPSHOT_PATH", str(path))
    return path


def test_reader_accepts_fresh_content_bound_private_snapshot(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _snapshot())

    result = owner.read_owner_snapshot({"task_id": "task-1"}, clock=lambda: 1_005)

    assert result["schema"] == "hermes-kanban-trusted-runtime/v1"
    assert result["identity"]["provider"] == owner.OWNER_SNAPSHOT_PROVIDER
    assert result["runtime"] == _runtime()


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda value: value["sources"]["predicate"].update(observed_at=900),
            "stale_owner_source",
        ),
        (
            lambda value: value["runtime"]["predicate_results"].update(
                {"task-1": {"satisfied": False}}
            ),
            "owner_source_digest_mismatch",
        ),
    ],
)
def test_reader_refuses_stale_or_source_digest_drift(
    monkeypatch, tmp_path, mutation, code
):
    value = _snapshot()
    mutation(value)
    value["snapshot_digest"] = owner._digest(
        {key: value[key] for key in owner._TOP_LEVEL_FIELDS - {"snapshot_digest"}}
    )
    _write(monkeypatch, tmp_path, value)

    with pytest.raises(RuntimeSnapshotError) as exc:
        owner.read_owner_snapshot({"task_id": "task-1"}, clock=lambda: 1_005)
    assert exc.value.code == code


def test_reader_refuses_unsafe_mode_before_parsing(monkeypatch, tmp_path):
    path = _write(monkeypatch, tmp_path, _snapshot())
    path.chmod(0o644)

    with pytest.raises(RuntimeSnapshotError) as exc:
        owner.read_owner_snapshot({"task_id": "task-1"}, clock=lambda: 1_005)
    assert exc.value.code == "unsafe_owner_snapshot"


def test_reader_refuses_missing_snapshot(monkeypatch, tmp_path):
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    monkeypatch.setattr(owner, "OWNER_SNAPSHOT_PATH", str(parent / "missing.json"))

    with pytest.raises(RuntimeSnapshotError) as exc:
        owner.read_owner_snapshot({"task_id": "task-1"}, clock=lambda: 1_005)
    assert exc.value.code == "missing_owner_snapshot"


def test_reader_refuses_wrong_task(monkeypatch, tmp_path):
    value = _snapshot()
    value["identity"]["task_id"] = "other-task"
    value["snapshot_digest"] = owner._digest(
        {key: value[key] for key in owner._TOP_LEVEL_FIELDS - {"snapshot_digest"}}
    )
    _write(monkeypatch, tmp_path, value)

    with pytest.raises(RuntimeSnapshotError) as exc:
        owner.read_owner_snapshot({"task_id": "task-1"}, clock=lambda: 1_005)
    assert exc.value.code == "owner_snapshot_identity_mismatch"


def test_reader_refuses_snapshot_digest_drift(monkeypatch, tmp_path):
    value = _snapshot()
    value["snapshot_digest"] = "0" * 64
    _write(monkeypatch, tmp_path, value)

    with pytest.raises(RuntimeSnapshotError) as exc:
        owner.read_owner_snapshot({"task_id": "task-1"}, clock=lambda: 1_005)
    assert exc.value.code == "owner_snapshot_digest_mismatch"
