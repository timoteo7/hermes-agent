"""Regression tests for the execute_code GitHub PR owner boundary.

The execute_code runtime can call subprocess/gh directly, so the terminal bash floor is
not sufficient to enforce the operator's repository-owner policy.
"""

import pytest

from tools import approval


@pytest.fixture(autouse=True)
def _clear_owner_allowlist(monkeypatch):
    monkeypatch.delenv("HERMES_PR_OPEN_ALLOW_OWNERS", raising=False)
    monkeypatch.delenv("FUSION_PR_OPEN_ALLOW_OWNERS", raising=False)


@pytest.mark.parametrize(
    "code",
    [
        'import subprocess; subprocess.run(["gh", "pr", "create", "--repo", "Runfusion/Fusion"])',
        'import subprocess; subprocess.run(["gh", "api", "repos/Runfusion/Fusion/pulls", "-f", "title=x"])',
        'import requests; requests.post("https://api.github.com/repos/Runfusion/Fusion/pulls")',
    ],
)
def test_execute_code_denies_pr_open_for_non_allowlisted_owner(code):
    result = approval.check_execute_code_guard(code, "local")

    assert result["approved"] is False
    assert result["pattern_key"] == "execute_code_pr_open"
    assert "allow-listed repository owners" in result["message"]


def test_execute_code_allows_pr_open_for_configured_owner(monkeypatch):
    monkeypatch.setenv("HERMES_PR_OPEN_ALLOW_OWNERS", "timoteo7")

    result = approval.check_execute_code_guard(
        'import subprocess; subprocess.run(["gh", "pr", "create", "--repo", "timoteo7/Fusion"])',
        "local",
    )

    assert result["approved"] is True


def test_execute_code_accepts_fusion_allowlist_fallback(monkeypatch):
    monkeypatch.setenv("FUSION_PR_OPEN_ALLOW_OWNERS", "timoteo7")

    result = approval.check_execute_code_guard(
        'import subprocess; subprocess.run(["gh", "pr", "create", "--repo", "timoteo7/Fusion"])',
        "local",
    )

    assert result["approved"] is True


def test_execute_code_does_not_block_read_only_github_api():
    result = approval.check_execute_code_guard(
        'import requests; requests.get("https://api.github.com/repos/Runfusion/Fusion/issues")',
        "local",
    )

    assert result["approved"] is True
