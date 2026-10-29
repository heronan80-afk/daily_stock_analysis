"""Regression tests for ai-berkshire bridge CLI error handling.

The bridge used to archive any non-empty stdout from a failed Claude CLI call
as a successful research report. On 2026-10-28 the 429 quota payload
("API Error: Request rejected (429) ...") became a 484-byte
``深南电路-dsa-signal-20261028.md`` while the run still logged Success: 4.
"""

from types import SimpleNamespace

from scripts import ai_berkshire_bridge as bridge


def _proc(returncode: int, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_detects_quota_error_payload():
    assert bridge._looks_like_api_error(
        "API Error: Request rejected (429) · You have exceeded the 5-hour usage quota."
    )


def test_does_not_flag_a_report_that_happens_to_contain_429():
    assert not bridge._looks_like_api_error("# 深度研究：中际旭创\n\n| 股价 | 429.00 |")


def test_invoke_fails_on_nonzero_exit_with_quota_error(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "shutil_which", lambda _cmd: "/usr/bin/claude")
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: _proc(
            1,
            "API Error: Request rejected (429) · You have exceeded the 5-hour usage quota.",
        ),
    )

    ok, message = bridge.invoke_ai_berkshire_research(
        stock_code="002916",
        stock_name="深南电路",
        question="研究一下",
        ai_berkshire_path=str(tmp_path),
    )

    assert not ok
    assert "429" in message


def test_invoke_fails_on_zero_exit_with_quota_error(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "shutil_which", lambda _cmd: "/usr/bin/claude")
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: _proc(0, "API Error: Request rejected (429)"),
    )

    ok, message = bridge.invoke_ai_berkshire_research(
        stock_code="002916",
        stock_name="深南电路",
        question="研究一下",
        ai_berkshire_path=str(tmp_path),
    )

    assert not ok
    assert "API error" in message


def test_invoke_keeps_genuine_partial_report_on_nonzero_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "shutil_which", lambda _cmd: "/usr/bin/claude")
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: _proc(1, "# 深度研究：深南电路\n\n正文（被截断）"),
    )

    ok, report = bridge.invoke_ai_berkshire_research(
        stock_code="002916",
        stock_name="深南电路",
        question="研究一下",
        ai_berkshire_path=str(tmp_path),
    )

    assert ok
    assert report.startswith("# 深度研究")
