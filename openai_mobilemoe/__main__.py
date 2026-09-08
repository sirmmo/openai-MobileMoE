"""Command-line entry point: ``openai-mobilemoe`` / ``python -m openai_mobilemoe``."""

from __future__ import annotations

import argparse
import logging

from .config import DEFAULT_MODEL, MODEL_FAMILY, Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openai-mobilemoe",
        description="Serve a MobileMoE checkpoint behind an OpenAI-compatible API.",
        epilog="Known checkpoints: " + ", ".join(MODEL_FAMILY),
    )
    parser.add_argument("--host", default=None, help="bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default 8000)")
    parser.add_argument(
        "--model",
        default=None,
        help=f"HuggingFace repo id or local path to serve (default {DEFAULT_MODEL})",
    )
    parser.add_argument("--model-id", default=None, help="model id to advertise")
    parser.add_argument(
        "--device", default=None, help="auto, cpu, cuda, cuda:N or mps (default auto)"
    )
    parser.add_argument(
        "--dtype", default=None, help="auto, bfloat16, float16 or float32 (default auto)"
    )
    parser.add_argument(
        "--attn-implementation", default=None, help="sdpa or eager (default: let torch choose)"
    )
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument(
        "--api-key", default=None, help="require this bearer token on every request"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="default generation cap (default 512)"
    )
    parser.add_argument(
        "--max-context", type=int, default=None, help="override the model's context window"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="repetition penalty when a request omits one (default 1.0 = off)",
    )
    parser.add_argument(
        "--max-queue-depth",
        type=int,
        default=None,
        help="reject requests once this many are queued (default 16)",
    )
    parser.add_argument(
        "--request-timeout", type=float, default=None, help="per-request deadline in seconds"
    )
    parser.add_argument(
        "--strict-model",
        action="store_true",
        help="404 on requests naming a model other than the one served",
    )
    parser.add_argument(
        "--no-extras",
        action="store_true",
        help="omit the x_mobilemoe block (timings, warnings) from responses",
    )
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = Settings.from_env()
    for attr, value in (
        ("host", args.host),
        ("port", args.port),
        ("model", args.model),
        ("model_id", args.model_id),
        ("device", args.device),
        ("dtype", args.dtype),
        ("attn_implementation", args.attn_implementation),
        ("threads", args.threads),
        ("api_key", args.api_key),
        ("max_new_tokens", args.max_new_tokens),
        ("max_context", args.max_context),
        ("default_repetition_penalty", args.repetition_penalty),
        ("max_queue_depth", args.max_queue_depth),
        ("request_timeout", args.request_timeout),
    ):
        if value is not None:
            setattr(settings, attr, value)
    if args.strict_model:
        settings.strict_model = True
    if args.no_extras:
        settings.expose_extras = False

    import uvicorn

    from .server import create_app

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=args.log_level,
        # One worker only: generation serializes through a single thread, and
        # several processes would each load their own copy of the weights.
        workers=1,
    )


if __name__ == "__main__":
    main()
