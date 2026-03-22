from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from llm_guard.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-guard",
        description="LLM reasoning guard proxy server",
    )
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--host", type=str, default=None, help="Override server host")
    parser.add_argument("--port", "-p", type=int, default=None, help="Override server port")
    parser.add_argument(
        "--log-level", type=str, default=None, help="Override log level (debug/info/warning/error)"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.log_level:
        config.server.log_level = args.log_level

    log_level = config.server.log_level.upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    logger = logging.getLogger("llm_guard")
    logger.info("Starting LLM-Guard on %s:%d", config.server.host, config.server.port)
    logger.info("Upstream: %s", config.upstream.base_url)

    from llm_guard.server import create_app

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
    )
