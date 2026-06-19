"""
CLI runner — run debugger from terminal without starting a server.
Usage: python run_debug.py
       python run_debug.py --root /path/to/pd_multimodal_ai
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from core.debugger import PDAutoDebugger, Status

logging.basicConfig(level=logging.WARNING)

COLORS = {
    "ok":      "\033[92m",   # green
    "warning": "\033[93m",   # yellow
    "error":   "\033[91m",   # red
    "skip":    "\033[90m",   # gray
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "blue":    "\033[94m",
    "dim":     "\033[2m",
}
ICONS = {"ok": "✓", "warning": "!", "error": "✗", "skip": "—"}


def col(text, *keys):
    return "".join(COLORS[k] for k in keys) + str(text) + COLORS["reset"]


def main():
    parser = argparse.ArgumentParser(description="PD System Auto-Debugger")
    parser.add_argument("--root",  default="..",  help="Path to pd_multimodal_ai root")
    parser.add_argument("--json",  action="store_true", help="Output raw JSON")
    parser.add_argument("--out",   default="debug_report.json", help="Save JSON report to file")
    args = parser.parse_args()

    debugger = PDAutoDebugger(project_root=args.root)

    if args.json:
        report = debugger.run_and_save(args.out)
        print(json.dumps({
            "summary": report.summary,
            "checks":  [{"name": c.name, "status": c.status.value,
                         "message": c.message, "fix": c.fix}
                        for c in report.checks],
        }, indent=2))
        return

    # Pretty CLI output
    print()
    print(col("  PD SYSTEM AUTO-DEBUGGER", "bold", "blue"))
    print(col("  ─────────────────────────────────────────", "dim"))
    print(f"  Scanning: {args.root}")
    print()

    report = debugger.run_and_save(args.out)

    for c in report.checks:
        icon   = ICONS[c.status.value]
        color  = c.status.value
        status = col(f"[{icon}]", color, "bold")
        name   = col(c.name.ljust(32), "bold")
        msg    = c.message
        print(f"  {status} {name} {msg}")
        if c.fix:
            print(f"       {col('Fix: ' + c.fix, 'warning')}")
        if c.detail and c.status.value == "error":
            for line in c.detail.split("\n")[:4]:
                print(f"       {col(line, 'dim')}")

    # Summary
    s = report.summary
    print()
    print(col("  ─────────────────────────────────────────", "dim"))
    health_color = {
        "healthy":  "ok",
        "degraded": "warning",
        "critical": "error",
    }.get(s["health"], "dim")

    print(
        f"  {col('RESULT:', 'bold')} "
        f"{col(s['health'].upper(), health_color, 'bold')}  "
        f"[ {col(str(s['ok']), 'ok')} passed  "
        f"{col(str(s['warnings']), 'warning')} warnings  "
        f"{col(str(s['errors']), 'error')} errors ]"
    )
    print(f"  Report saved to: {args.out}")
    print()

    sys.exit(0 if s["errors"] == 0 else 1)


if __name__ == "__main__":
    main()
