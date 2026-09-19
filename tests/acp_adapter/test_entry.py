"""Tests for acp_adapter.entry startup wiring."""

import os
import sys

import acp
import pytest

from acp_adapter import entry


def _write_approval_config(home, mode):
    home.mkdir()
    (home / "config.yaml").write_text(f"approvals:\n  mode: {mode}\n", encoding="utf-8")


def test_check_binds_explicit_hermes_home_before_config_load(tmp_path, monkeypatch):
    home = tmp_path / "seat-a"
    home.mkdir()
    seen = {}

    monkeypatch.setattr(
        entry,
        "_run_check",
        lambda: seen.update(home=os.environ.get("HERMES_HOME"), check=True),
    )

    entry.main(["--check", "--hermes-home", str(home)])

    assert seen == {"home": str(home.resolve()), "check": True}


@pytest.mark.parametrize("mode", ["manual", "smart", "off"])
def test_check_accepts_explicit_matching_approval_mode(tmp_path, monkeypatch, mode):
    home = tmp_path / mode
    _write_approval_config(home, mode)
    checked = []

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(entry, "_run_check", lambda: checked.append(True))

    entry.main([
        "--check", "--hermes-home", str(home), "--require-approval-mode", mode,
    ])

    assert checked == [True]


def test_required_approval_mode_refuses_mismatch(tmp_path, monkeypatch, capsys):
    home = tmp_path / "seat-a"
    _write_approval_config(home, "smart")

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(entry, "_run_check", lambda: pytest.fail("check must not run"))

    with pytest.raises(SystemExit, match="2"):
        entry.main([
            "--check", "--hermes-home", str(home), "--require-approval-mode", "manual",
        ])

    assert "required manual, effective smart" in capsys.readouterr().err


def test_required_approval_mode_refuses_implicit_default(tmp_path, monkeypatch, capsys):
    home = tmp_path / "seat-a"
    home.mkdir()
    (home / "config.yaml").write_text("agent: {}\n", encoding="utf-8")

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)

    with pytest.raises(SystemExit, match="2"):
        entry.main([
            "--check", "--hermes-home", str(home), "--require-approval-mode", "smart",
        ])

    assert "must explicitly set approvals.mode" in capsys.readouterr().err


def test_required_approval_mode_requires_explicit_home():
    with pytest.raises(SystemExit, match="2"):
        entry.main(["--check", "--require-approval-mode", "manual"])


def test_explicit_hermes_home_must_be_absolute(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="2"):
        entry.main(["--check", "--hermes-home", "seat-a"])

    assert "must be an absolute path" in capsys.readouterr().err


def test_main_enables_unstable_protocol(monkeypatch):
    calls = {}

    async def fake_run_agent(agent, **kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert calls["kwargs"]["use_unstable_protocol"] is True


def test_main_skips_configured_mcp_discovery_when_requested(monkeypatch):
    discovery_calls = []

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(
        "tools.mcp_tool_discovery.discover_mcp_tools",
        lambda: discovery_calls.append(True),
    )
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert discovery_calls == []










def test_main_setup_offers_browser_install_when_tty(monkeypatch):
    """When stdin is a TTY and the user answers yes, model setup is followed
    by a browser-tools bootstrap call."""
    monkeypatch.setattr("hermes_cli.main.main", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "y")

    bootstrap_calls = []
    monkeypatch.setattr(
        entry,
        "_run_setup_browser",
        lambda assume_yes=False: bootstrap_calls.append(assume_yes) or 0,
    )

    entry.main(["--setup"])

    assert bootstrap_calls == [False]










def test_main_setup_browser_propagates_browser_failure(monkeypatch):
    """If browser install fails, exit code is 1."""
    def fake_ensure(dep, interactive=True):
        return dep != "browser"  # browser fails

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1
