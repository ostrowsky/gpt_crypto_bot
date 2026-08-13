"""Fail-closed evidence and staged-change harness for gpt_crypto_bot."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
RUNTIME_PREFIXES = (".runtime/", "files/.runtime/")
RUNTIME_NAMES = {
    "files/positions.json",
    "files/bot_events.jsonl",
    "files/agent_events.jsonl",
    "files/ml_candidate_ranker.json",
    "files/ml_candidate_ranker_report.json",
    "files/ml_candidate_ranker_shadow_report.json",
}
MATERIAL_PREFIXES = (
    "files/", "skills/bot-progress-report/", "skills/signal-quality-evaluator/",
)
MATERIAL_NAMES = {"AGENTS.md", "SCOUT_OPTIMIZATION_SPEC.md"}
BEHAVIOR_NAMES = {"files/config.py", "files/bot.py", "files/monitor.py", "files/strategy.py"}


@dataclass(frozen=True)
class Finding:
    check_id: str
    invariant: str
    severity: str
    message: str
    evidence: str = ""
    remediation: str = ""

    @property
    def blocking(self) -> bool:
        return self.severity == "error"


class Audit:
    def __init__(self, profile: str) -> None:
        self.profile = profile
        self.findings: list[Finding] = []
        self.checks_run: list[str] = []

    def checked(self, check_id: str) -> None:
        if check_id not in self.checks_run:
            self.checks_run.append(check_id)

    def add(
        self,
        check_id: str,
        invariant: str,
        severity: str,
        message: str,
        evidence: str = "",
        remediation: str = "",
    ) -> None:
        self.checked(check_id)
        self.findings.append(Finding(check_id, invariant, severity, message, evidence, remediation))

    @property
    def blocking(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.blocking]

    def payload(self) -> dict:
        return {
            "schema_version": 1,
            "profile": self.profile,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "fail" if self.blocking else "pass",
            "checks_run": self.checks_run,
            "blocking_count": len(self.blocking),
            "warning_count": sum(f.severity == "warning" for f in self.findings),
            "findings": [asdict(f) for f in self.findings],
        }


def _run(cmd: Sequence[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd), cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _git_changed(root: Path, staged: bool) -> list[str]:
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--cached")
    cmd.extend(["--name-only", "--diff-filter=ACMR"])
    result = _run(cmd, root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def _is_material(path: str) -> bool:
    return path in MATERIAL_NAMES or path.startswith(MATERIAL_PREFIXES)


def _is_runtime(path: str) -> bool:
    return path in RUNTIME_NAMES or path.startswith(RUNTIME_PREFIXES) or path.startswith("catboost_info/")


def audit_enforcement(audit: Audit, root: Path = ROOT) -> None:
    check = "TH12_ENFORCEMENT"
    audit.checked(check)
    required = (
        root / "AGENTS.md",
        root / "docs" / "specs" / "truth-harness.md",
        root / "files" / "truth_harness.py",
        root / "skills" / "crypto-bot-truth-harness" / "SKILL.md",
        root / ".githooks" / "pre-commit",
    )
    for path in required:
        if not path.exists():
            audit.add(check, "TH-12", "error", "Truth Harness enforcement file is missing", str(path))
    agents = _read(root / "AGENTS.md")
    if "truth_harness.py full" not in agents or "truth_harness.py change --staged" not in agents:
        audit.add(check, "TH-12", "error", "AGENTS.md does not require both harness profiles")
    hook = _read(root / ".githooks" / "pre-commit")
    if "truth_harness.py\" change --staged" not in hook:
        audit.add(check, "TH-12", "error", "Tracked pre-commit hook does not run staged Harness")
    hooks_path = _run(["git", "config", "--get", "core.hooksPath"], root).stdout.strip()
    if hooks_path != ".githooks":
        audit.add(
            check,
            "TH-12",
            "warning",
            "Working copy does not enforce tracked hooks",
            f"core.hooksPath={hooks_path or 'unset'}",
            "Run git config core.hooksPath .githooks where local Git config is writable.",
        )


def audit_md_config(audit: Audit, root: Path = ROOT) -> None:
    check = "TH09_MD_CONFIG"
    audit.checked(check)
    index = _read(root / "docs" / "FEATURE_SPEC_INDEX.md")
    for spec in ("docs/specs/truth-harness.md", "docs/specs/full-watchlist-rotating-monitor.md"):
        if spec not in index:
            audit.add(check, "TH-09/TH-12", "error", "Feature spec is not registered", spec)
    config = _read(root / "files" / "config.py")
    bot = _read(root / "files" / "bot.py")
    monitor = _read(root / "files" / "monitor.py")
    expected = {
        "MONITOR_FULL_WATCHLIST": config,
        "MAX_POLL_PER_CYCLE": config,
        "_build_full_watchlist_reports": bot + monitor,
        "_select_poll_coins": monitor,
    }
    for marker, source in expected.items():
        if marker not in source:
            audit.add(check, "TH-09", "error", "Full-watchlist spec and implementation disagree", f"missing={marker}")


def audit_progress_report(audit: Audit, root: Path = ROOT) -> None:
    source = _read(root / "skills" / "bot-progress-report" / "scripts" / "build_progress_report.py")
    audit.checked("TH02_PROGRESS_VERDICT")
    legacy_shortcut = 'watchlist_top_bought_rate_pct"] >= 50' in source and 'toward_goal": "improving"' in source
    if legacy_shortcut:
        audit.add("TH02_PROGRESS_VERDICT", "TH-02/TH-05", "error", "Absolute-rate shortcut can claim improvement without comparison")
    result = _run([sys.executable, str(root / "skills" / "bot-progress-report" / "scripts" / "build_progress_report.py"), "--days", "14"], root)
    if result.returncode != 0:
        audit.add("TH10_PROGRESS_EXECUTION", "TH-10", "error", "Progress report failed", (result.stderr or result.stdout)[-2000:])
        return
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        audit.add("TH10_PROGRESS_EXECUTION", "TH-10", "error", "Progress report emitted invalid JSON", str(exc))
        return
    audit.checked("TH01_RATIO_CONTEXT")
    for section, metric in (("scout", "early_capture"), ("scout", "watchlist_top_bought"), ("quality", "miss_rate")):
        value = (payload.get(section) or {}).get(metric) or {}
        if not {"numerator", "denominator", "rate_pct"}.issubset(value):
            audit.add("TH01_RATIO_CONTEXT", "TH-01", "error", "Published rate lacks reconstructable evidence", f"{section}.{metric}")
    audit.checked("TH05_COVERAGE")
    days = (payload.get("coverage") or {}).get("days", [])
    by_day = {row.get("day"): row for row in days}
    invalid = [
        day for day in payload.get("objective_eligible_days", [])
        if not (by_day.get(day, {}).get("has_critic") and by_day.get(day, {}).get("has_goal"))
    ]
    if invalid:
        audit.add("TH05_COVERAGE", "TH-05", "error", "Unknown objective denominators enter capture aggregation", ", ".join(invalid))
    invalid_quality = [
        day for day in payload.get("quality_eligible_days", [])
        if by_day.get(day, {}).get("status") != "complete"
    ]
    if invalid_quality:
        audit.add("TH05_COVERAGE", "TH-05", "error", "Partial quality days enter quality aggregation", ", ".join(invalid_quality))
    if (payload.get("verdict") or {}).get("toward_goal") == "improving" and (payload.get("comparison") or {}).get("status") != "comparable":
        audit.add("TH05_COVERAGE", "TH-05/TH-10", "error", "Positive verdict lacks comparable evidence", json.dumps(payload.get("comparison"), ensure_ascii=False))
    freshness = ((payload.get("rl") or {}).get("freshness") or {})
    if freshness.get("status") == "stale":
        audit.add("TH10_EVIDENCE_FRESHNESS", "TH-10", "warning", "RL evidence is stale", json.dumps(freshness))


def audit_model_provenance(audit: Audit, root: Path = ROOT) -> None:
    latest = _load(root / ".runtime" / "reports" / "rl_train_latest.json")
    if not latest:
        audit.add("TH03_MODEL_PROVENANCE", "TH-03/TH-04", "warning", "No current RL/ranker report is available")
        return
    audit.checked("TH03_MODEL_PROVENANCE")
    provenance = latest.get("evaluation_provenance") or {}
    missing = [key for key in ("feature_time", "label_time", "label_definition", "evaluation_scope") if not provenance.get(key)]
    if missing:
        audit.add("TH03_MODEL_PROVENANCE", "TH-03/TH-04", "error", "Model achievement evidence lacks timing/holdout provenance", f"missing={missing}", "Keep model metrics diagnostic-only until training artifacts record immutable timing and chronological holdout scope.")
    elif provenance.get("evaluation_scope") != "out_of_sample_time_holdout":
        audit.add("TH04_MODEL_HOLDOUT", "TH-04", "error", "Model evaluation is not a chronological out-of-sample holdout", str(provenance.get("evaluation_scope")))


def audit_portfolio_alpha(audit: Audit, root: Path = ROOT) -> None:
    audit.checked("TH11_PORTFOLIO_ALPHA")
    candidates = sorted((root / ".runtime" / "reports").glob("*portfolio*alpha*.json"))
    valid = False
    for path in candidates[-5:]:
        payload = _load(path)
        if payload.get("net_alpha_after_costs") is not None and payload.get("benchmark"):
            valid = True
            break
    if not valid:
        audit.add("TH11_PORTFOLIO_ALPHA", "TH-11", "error", "Canonical unified portfolio alpha after costs is absent", "Per-trade/per-mode PnL cannot prove portfolio profitability.")


def audit_change(audit: Audit, root: Path = ROOT, staged: bool = True) -> None:
    files = _git_changed(root, staged)
    audit.checked("TH12_STAGED_SCOPE")
    if not files:
        audit.add("TH12_STAGED_SCOPE", "TH-12", "error", "No staged files to audit")
        return
    runtime = [path for path in files if _is_runtime(path)]
    if runtime:
        audit.add("TH12_STAGED_SCOPE", "TH-12", "error", "Runtime/generated state is staged", ", ".join(runtime))
    material = [path for path in files if _is_material(path)]
    if material:
        specs = [path for path in files if path.startswith("docs/specs/") and path.endswith(".md")]
        tests = [path for path in files if Path(path).name.startswith("test_") and path.endswith(".py")]
        if not specs:
            audit.add("TH12_STAGED_SCOPE", "TH-12", "error", "Material change has no staged feature spec", ", ".join(material))
        if not tests:
            audit.add("TH12_STAGED_SCOPE", "TH-12", "error", "Material change has no staged focused tests", ", ".join(material))
        if "docs/FEATURE_SPEC_INDEX.md" not in files:
            audit.add("TH12_STAGED_SCOPE", "TH-12", "error", "Material change does not update the feature spec index")
    behavior = [path for path in files if path in BEHAVIOR_NAMES]
    if behavior:
        staged_specs = "\n".join(_read(root / path) for path in files if path.startswith("docs/specs/") and path.endswith(".md"))
        for marker in ("Rollback", "maximum", "canary"):
            if marker.lower() not in staged_specs.lower():
                audit.add("TH07_CHANGE_GUARDS", "TH-07", "error", "Behavior change spec lacks a required safety contract", f"missing={marker}")
    audit.checked("TH12_HARNESS_SELF_TEST")
    if "files/truth_harness.py" in files and "files/test_truth_harness.py" not in files:
        audit.add("TH12_HARNESS_SELF_TEST", "TH-12", "error", "Harness changed without its focused test")


def run_profile(profile: str, root: Path = ROOT, staged: bool = True) -> Audit:
    audit = Audit(profile)
    if profile == "full":
        audit_enforcement(audit, root)
        audit_md_config(audit, root)
        audit_progress_report(audit, root)
        audit_model_provenance(audit, root)
        audit_portfolio_alpha(audit, root)
    elif profile == "change":
        audit_change(audit, root, staged=staged)
    else:
        raise ValueError(profile)
    return audit


def _render(payload: dict) -> str:
    lines = [
        f"Truth Harness: {payload['status'].upper()} profile={payload['profile']}",
        f"blocking={payload['blocking_count']} warnings={payload['warning_count']}",
    ]
    for finding in payload["findings"]:
        lines.append(f"- {finding['severity'].upper()} {finding['invariant']} {finding['check_id']}: {finding['message']}")
        if finding.get("evidence"):
            lines.append(f"  evidence: {finding['evidence']}")
        if finding.get("remediation"):
            lines.append(f"  remediation: {finding['remediation']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("full", "change"))
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        audit = run_profile(args.profile, ROOT, staged=args.staged or args.profile == "change")
        payload = audit.payload()
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "profile": args.profile,
            "status": "unknown",
            "blocking_count": 0,
            "warning_count": 0,
            "checks_run": [],
            "findings": [{
                "check_id": "HARNESS_INTERNAL",
                "invariant": "TH-10",
                "severity": "error",
                "message": "Harness could not complete",
                "evidence": f"{exc.__class__.__name__}: {exc}",
                "remediation": "Repair the Harness before approving another claim.",
            }],
        }
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(_render(payload))
        return 2
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(_render(payload))
    return 1 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
