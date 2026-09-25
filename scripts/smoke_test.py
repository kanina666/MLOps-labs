import argparse
import json
import os
from urllib.error import HTTPError
from urllib.request import urlopen


def check(path: str, expected_status: int, port: int) -> dict[str, object]:
    try:
        response = urlopen(  # noqa: S310 -- fixed local smoke-test target
            f"http://127.0.0.1:{port}{path}", timeout=10
        )
    except HTTPError as exc:
        response = exc
    with response:
        if response.status != expected_status:
            raise RuntimeError(f"{path}: expected {expected_status}, got {response.status}")
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degraded", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    check("/healthz", 200, args.port)
    version = check("/api/v1/version", 200, args.port)
    if version["version"] != os.environ["EXPECTED_VERSION"]:
        raise RuntimeError("Image version and /api/v1/version differ")
    health = check("/api/v1/health", 503 if args.degraded else 200, args.port)
    if health["status"] != ("degraded" if args.degraded else "ok"):
        raise RuntimeError("Unexpected health report")
    print("Smoke test passed: liveness, version and health")


main()
