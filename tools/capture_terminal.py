"""Run real commands and save their output as terminal-style screenshots for the report.

Nothing is simulated: each command is executed now and its exact stdout/stderr is
rendered, together with the command line and exit code.

usage:
    python tools/capture_terminal.py k8s        # cluster, workloads, services, LB requests
    python tools/capture_terminal.py docker     # image build, container run, smoke test
    python tools/capture_terminal.py tests      # lint + pytest summary
"""

from __future__ import annotations

import argparse
import html
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = ROOT / "screenshots"
PYTHON = f'"{Path(sys.executable).as_posix()}"'  # quoted: paths may contain spaces


def _bash() -> str:
    """Git Bash on Windows (not the WSL launcher in System32), plain bash elsewhere."""
    if os.name == "nt":
        for candidate in (
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ):
            if Path(candidate).exists():
                return candidate
    found = shutil.which("bash")
    if not found:
        sys.exit("bash is required to run the captured commands")
    return found


API = "http://127.0.0.1:8000"
PREDICT = (
    f'curl -s -X POST {API}/predict -H "Content-Type: application/json" -d @sample_request.json'
)

# Each shot: (file name, title, [commands]).
SETS: dict[str, list[tuple[str, str, list[str]]]] = {
    "k8s": [
        (
            "k8s_01_cluster_and_workloads.png",
            "Kubernetes: cluster, API deployment, pods, services",
            [
                "kubectl config current-context",
                "kubectl get nodes -o wide",
                "kubectl -n cardiorisk get deployments,replicasets,pods,pdb -o wide",
                "kubectl -n cardiorisk get services -o wide",
            ],
        ),
        (
            "k8s_02_rollout_and_probes.png",
            "Kubernetes: rollout status, image, probes and resources",
            [
                "kubectl -n cardiorisk rollout status deployment/cardiorisk-api",
                "kubectl -n cardiorisk rollout history deployment/cardiorisk-api",
                "kubectl -n cardiorisk describe deployment cardiorisk-api",
            ],
        ),
        (
            "k8s_03_loadbalancer_requests.png",
            "Kubernetes: requests through the LoadBalancer service",
            [
                f"curl -s {API}/health",
                f"curl -s {API}/ready",
                PREDICT,
                f"curl -s -o /dev/null -w 'invalid input -> HTTP %{{http_code}}\\n' -X POST "
                f"{API}/predict -H 'Content-Type: application/json' -d '{{\"age\": 5}}'",
                "kubectl -n cardiorisk logs deployment/cardiorisk-api --tail=6",
            ],
        ),
        (
            "k8s_04_monitoring_stack.png",
            "Kubernetes: monitoring namespace (Prometheus + Grafana)",
            [
                "kubectl -n monitoring get deployments,pods,services -o wide",
                "kubectl -n monitoring get configmaps,secrets",
            ],
        ),
    ],
    "docker": [
        (
            "docker_01_build.png",
            "Docker: build the serving image",
            [
                "docker build -t cardiorisk-api:1.0.0 . 2>&1 | tail -n 22",
                "docker images cardiorisk-api",
            ],
        ),
        (
            "docker_02_run_and_smoke_test.png",
            "Docker: run the container and smoke-test it",
            [
                "docker rm -f cardiorisk-api-demo >/dev/null 2>&1; "
                "docker run -d --name cardiorisk-api-demo -p 18000:8000 cardiorisk-api:1.0.0",
                "bash scripts/smoke_test.sh http://127.0.0.1:18000",
                "docker ps --filter name=cardiorisk-api-demo --format "
                "'table {{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}'",
                "docker logs cardiorisk-api-demo 2>&1 | tail -n 4",
                "docker rm -f cardiorisk-api-demo",
            ],
        ),
    ],
    "tests": [
        (
            "tests_01_lint_and_pytest.png",
            "Lint and unit tests",
            [
                f"{PYTHON} -m ruff check . && {PYTHON} -m ruff format --check .",
                f"{PYTHON} -m pytest --cov=cardiorisk --cov=app --cov-report=term -q 2>&1"
                " | tail -n 24",
            ],
        ),
    ],
}

STYLE = """
body { margin: 0; background: #1e1e2e;
       font: 13px/1.45 'Cascadia Mono', Consolas, 'DejaVu Sans Mono', monospace; }
.win { margin: 18px; border-radius: 8px; overflow: hidden; box-shadow: 0 6px 24px #0008; }
.bar { background: #313244; color: #cdd6f4; padding: 8px 14px; font-weight: 600; }
.bar span { color: #a6adc8; font-weight: 400; float: right; }
pre { margin: 0; padding: 14px 16px; background: #11111b; color: #cdd6f4;
      white-space: pre-wrap; word-break: break-word; }
.cmd { color: #a6e3a1; } .ps { color: #89b4fa; } .rc0 { color: #6c7086; } .rcx { color: #f38ba8; }
"""


def run(command: str) -> tuple[str, int]:
    shell = [_bash(), "-c", command]
    env = {**os.environ, "COLUMNS": "160", "NO_COLOR": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"}
    proc = subprocess.run(
        shell, cwd=ROOT, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace"
    )
    return (proc.stdout + proc.stderr).rstrip(), proc.returncode


def render(title: str, commands: list[str]) -> tuple[str, int]:
    body, worst = [], 0
    for command in commands:
        display = command.replace(PYTHON, "python")
        output, code = run(command)
        worst = max(worst, code)
        body.append(f'<span class="ps">$</span> <span class="cmd">{html.escape(display)}</span>')
        if output:
            body.append(html.escape(output))
        css = "rc0" if code == 0 else "rcx"
        body.append(f'<span class="{css}">[exit {code}]</span>\n')
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    page = (
        f"<!doctype html><html><head><meta charset='utf-8'><style>{STYLE}</style></head>"
        f"<body><div class='win'><div class='bar'>{html.escape(title)}<span>{stamp}</span></div>"
        f"<pre>{chr(10).join(body)}</pre></div></body></html>"
    )
    return page, worst


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture real command output as screenshots")
    parser.add_argument("set", choices=sorted(SETS))
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    SCREENSHOT_DIR.mkdir(exist_ok=True)
    failures = 0
    with sync_playwright() as pw, tempfile.TemporaryDirectory() as tmp:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 600})
        for file, title, commands in SETS[args.set]:
            markup, code = render(title, commands)
            source = Path(tmp) / "shot.html"
            source.write_text(markup, encoding="utf-8")
            page.goto(source.as_uri())
            page.locator(".win").screenshot(path=str(SCREENSHOT_DIR / file))
            status = "saved " if code == 0 else "saved (a command exited non-zero)"
            failures += code != 0
            print(f"{status} screenshots/{file}", flush=True)
        browser.close()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
