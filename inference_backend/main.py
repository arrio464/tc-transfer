from __future__ import annotations

import argparse

import uvicorn

from inference_backend.app import create_app
from inference_backend.config import BackendSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the high-performance TC inference backend")
    parser.add_argument("--host", default=None, help="Bind host, defaults to TC_BACKEND_HOST or 0.0.0.0")
    parser.add_argument("--port", type=int, default=None, help="Bind port, defaults to TC_BACKEND_PORT or 8080")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--log-level", default=None, help="Uvicorn log level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = BackendSettings.from_env()

    host = args.host or settings.host
    port = args.port or settings.port
    workers = args.workers or settings.workers
    log_level = args.log_level or settings.log_level

    app = create_app(settings=settings)
    uvicorn.run(app=app, host=host, port=port, workers=workers, log_level=log_level)


if __name__ == "__main__":
    main()
