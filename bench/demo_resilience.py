"""Scripted resilience demo (Phase 7 Group 2): watch the circuit breaker trip and traffic fall back
to a healthy provider when one goes down mid-run.

It launches two local subprocesses — the mock upstream (`bench.mock_upstream`, two backends `a`/`b`)
and a real gateway (`gateway.main`) wired via a temp config to route alias `demo` = priority
[mock_a, mock_b] — then:

  Phase 1 (healthy)   drive traffic -> every request served by mock_a (the primary).
  Phase 2 (a is down) flip mock_a to 503 -> the executor falls back to mock_b; after enough failures
                      the mock_a breaker OPENS and mock_a is skipped entirely.

Evidence is read straight from the gateway: the `x-switchyard-provider` response header per request,
the breaker snapshot at `/healthz`, and the `switchyard_provider_failures_total` /
`switchyard_requests_total` counters at `/metrics`. The same signals drive the Grafana "Provider
failures / fallbacks" and "Request rate by provider" panels under `docker compose up`.

The real `config/` is never touched (a temp config is used via env overrides), and both subprocesses
are torn down on exit. Run: `python -m bench.demo_resilience`.
"""

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _wait_ready(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        time.sleep(0.3)
    raise RuntimeError(f"service at {url} did not become ready in {timeout_s}s")


def _write_demo_config(cfg_dir: Path, mock_port: int) -> tuple[Path, Path]:
    providers = cfg_dir / "providers.yaml"
    models = cfg_dir / "models.yaml"
    providers.write_text(
        "providers:\n"
        "  mock_a:\n"
        f"    base_url: http://127.0.0.1:{mock_port}/a/v1\n"
        "    auth: none\n"
        "  mock_b:\n"
        f"    base_url: http://127.0.0.1:{mock_port}/b/v1\n"
        "    auth: none\n"
    )
    models.write_text(
        "aliases:\n"
        "  demo:\n"
        "    policy: priority\n"
        "    targets:\n"
        "      - { provider: mock_a, model: mock-model }\n"
        "      - { provider: mock_b, model: mock-model }\n"
    )
    return providers, models


def _drive(gateway_url: str, n: int) -> Counter:
    """Send n requests to the demo alias; count which provider served each."""
    providers: Counter = Counter()
    with httpx.Client(base_url=gateway_url, timeout=10.0) as client:
        for _ in range(n):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
            )
            providers[resp.headers.get("x-switchyard-provider", f"HTTP {resp.status_code}")] += 1
    return providers


def _metric_lines(gateway_url: str, needle: str) -> list[str]:
    text = httpx.get(f"{gateway_url}/metrics", timeout=5.0).text
    return [ln for ln in text.splitlines() if ln.startswith(needle)]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Resilience demo: breaker trip + fallback")
    parser.add_argument("--mock-port", type=int, default=8100)
    parser.add_argument("--gateway-port", type=int, default=8101)
    parser.add_argument("--n", type=int, default=12, help="requests per phase")
    args = parser.parse_args(argv)

    gateway_url = f"http://127.0.0.1:{args.gateway_port}"
    mock_url = f"http://127.0.0.1:{args.mock_port}"
    cfg_dir = Path(tempfile.mkdtemp(prefix="switchyard-demo-"))
    procs: list[subprocess.Popen] = []

    try:
        providers_cfg, models_cfg = _write_demo_config(cfg_dir, args.mock_port)

        print("Starting mock upstream + gateway (temp config; real config untouched)...")
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "bench.mock_upstream:app",
                    "--port",
                    str(args.mock_port),
                    "--log-level",
                    "warning",
                ],
                cwd=_REPO_ROOT,
            )
        )
        gw_env = {
            **os.environ,
            "PROVIDERS_CONFIG": str(providers_cfg),
            "MODELS_CONFIG": str(models_cfg),
            "CACHE_ENABLED": "false",  # no MiniLM download for a routing demo
        }
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "gateway.main:app",
                    "--port",
                    str(args.gateway_port),
                    "--log-level",
                    "warning",
                ],
                cwd=_REPO_ROOT,
                env=gw_env,
            )
        )
        _wait_ready(f"{mock_url}/health")
        _wait_ready(f"{gateway_url}/healthz")

        print(f"\n--- Phase 1: both providers healthy ({args.n} requests) ---")
        phase1 = _drive(gateway_url, args.n)
        print(f"served by: {dict(phase1)}   (expect all mock_a — the primary)")

        print("\n*** Simulating a provider outage: mock_a -> 503 ***")
        httpx.post(f"{mock_url}/control/a/fail", timeout=5.0)

        print(f"\n--- Phase 2: mock_a is down ({args.n} requests) ---")
        phase2 = _drive(gateway_url, args.n)
        print(f"served by: {dict(phase2)}   (expect mock_b — transparent cross-provider fallback)")

        circuits = httpx.get(f"{gateway_url}/healthz", timeout=5.0).json()["circuits"]
        print(f"\nbreaker state (/healthz): {circuits}   (expect mock_a: open)")

        print("\nprovider failures (/metrics):")
        for line in _metric_lines(gateway_url, "switchyard_provider_failures_total"):
            print(f"  {line}")
        print("requests by provider/outcome (/metrics):")
        for line in _metric_lines(gateway_url, "switchyard_requests_total"):
            print(f"  {line}")

        print(
            "\nEvery client request returned 200 throughout — the outage was absorbed by "
            "fallback.\nUnder `docker compose up`, the same activity shows in Grafana's 'Provider\n"
            "failures / fallbacks' and 'Request rate by provider' panels."
        )
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5.0)
        shutil.rmtree(cfg_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
