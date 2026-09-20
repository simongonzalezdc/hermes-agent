"""Auto-decompose tick under multiplex.

Regression for #107955 / #57837: the tick runs off-turn in a fresh Context, so
``get_secret`` fails closed unless the tick installs the launch profile's scope.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from pathlib import Path

from agent import secret_scope as ss
from gateway import kanban_watchers_dispatcher as kwd
from gateway.kanban_watchers_common import _to_thread_process_service


def _dispatcher():
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    return kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)


def test_auto_decompose_tick_reads_launch_profile_secrets_under_multiplex(monkeypatch, tmp_path):
    import hermes_cli

    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=launch-profile-key\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])

    seen = {}

    def fake_decompose(task_id, author=None):
        seen["value"] = ss.get_secret("ANTHROPIC_API_KEY")
        return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)

    fake = SimpleNamespace(list_triage_ids=lambda: ["t1"], decompose_task=fake_decompose)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake, raising=False)

    ss.set_multiplex_active(True)
    try:
        # Same hop the gateway uses: fresh Context, no inherited per-turn scope.
        decomposed = asyncio.run(_to_thread_process_service(_dispatcher().auto_decompose_tick, 5))
    finally:
        ss.set_multiplex_active(False)

    assert decomposed == 1
    assert seen["value"] == "launch-profile-key"
    assert ss.current_secret_scope() is None


def test_embedded_dispatcher_passes_fixed_owner_snapshot_reader(monkeypatch, tmp_path):
    captured = {}

    class Connection:
        def close(self):
            captured["closed"] = True

    fake_connect = SimpleNamespace(connect=lambda **kwargs: Connection())

    def dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return "tick"

    fake_dispatch = SimpleNamespace(dispatch_once=dispatch_once)
    kb = SimpleNamespace(
        DEFAULT_BOARD="default", kanban_db_path=lambda slug: Path(tmp_path / f"{slug}.db"),
    )
    dispatcher = kwd._KanbanDispatcher(
        kb, kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None),
    )
    monkeypatch.setattr(kwd, "_kbc", lambda: fake_connect)
    monkeypatch.setattr(kwd, "_kbd", lambda: fake_dispatch)

    assert dispatcher.tick_once_for_board("default") == "tick"
    from hermes_cli.kanban_runtime_owner_snapshot import read_owner_snapshot
    assert captured["runtime_policy_provider"] is read_owner_snapshot
    assert captured["board"] == "default"
    assert captured["closed"] is True
