"""FastAPI application exposing MobileMoE over an OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from . import __version__, translate
from .config import Settings
from .engine import (
    EngineError,
    EngineOverloaded,
    GenerationParams,
    GenerationResult,
    MobileMoEEngine,
    PromptTooLong,
)
from .schemas import ChatCompletionRequest, CompletionRequest, error_body
from .streaming import DONE, chat_chunk, sse, text_chunk, usage_chunk
from .translate import TranslationError

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, engine: Any = None) -> FastAPI:
    """Build the app. ``engine`` may be injected for testing."""
    settings = settings or Settings.from_env()
    owns_engine = engine is None
    fingerprint = f"fp_mobilemoe_{__version__}"
    started_at = int(time.time())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if owns_engine:
            app.state.engine = MobileMoEEngine(settings)
            log.info("loading %s ...", settings.model)
            await asyncio.to_thread(app.state.engine.start)
        else:
            app.state.engine = engine
        try:
            yield
        finally:
            if owns_engine:
                app.state.engine.shutdown()

    app = FastAPI(
        title="openai-MobileMoE",
        version=__version__,
        description="OpenAI-compatible API for Meta's MobileMoE on-device MoE models.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # -- plumbing ----------------------------------------------------------

    async def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not settings.api_key:
            return
        if authorization != f"Bearer {settings.api_key}":
            raise HTTPException(
                status_code=401,
                detail=error_body(
                    "Incorrect API key provided.", "invalid_request_error", code="invalid_api_key"
                ),
            )

    def get_engine(request: Request) -> Any:
        eng = request.app.state.engine
        if not eng.ready:
            raise HTTPException(
                status_code=503,
                detail=error_body(
                    "The model is still loading; retry shortly.", "server_error", code="not_ready"
                ),
            )
        return eng

    def resolve_model(name: str | None) -> str:
        if name and settings.strict_model and name not in settings.served_names():
            raise HTTPException(
                status_code=404,
                detail=error_body(
                    f"The model {name!r} does not exist.",
                    "invalid_request_error",
                    param="model",
                    code="model_not_found",
                ),
            )
        return name or settings.served_model_id

    def _translation_http(exc: TranslationError, status: int = 400) -> HTTPException:
        return HTTPException(
            status_code=status,
            detail=error_body(exc.message, param=exc.param, code=exc.code),
        )

    async def run_generation(
        eng: Any,
        prompt_ids: list[int],
        params: GenerationParams,
        on_text: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        """Submit to the generation thread and await the result with a deadline."""
        cancel = threading.Event()
        queued = time.monotonic()
        try:
            future = eng.submit(eng.generate, prompt_ids, params, on_text, cancel)
        except EngineOverloaded as exc:
            raise HTTPException(
                status_code=429,
                detail=error_body(str(exc), "rate_limit_error", code="engine_busy"),
            ) from exc
        try:
            result = await asyncio.wait_for(
                asyncio.wrap_future(future), timeout=settings.request_timeout
            )
        except asyncio.TimeoutError as exc:
            cancel.set()
            future.cancel()
            raise HTTPException(
                status_code=504,
                detail=error_body(
                    f"The request exceeded {settings.request_timeout:g}s waiting for the engine.",
                    "server_error",
                    code="engine_timeout",
                ),
            ) from exc
        except PromptTooLong as exc:
            raise HTTPException(
                status_code=400,
                detail=error_body(str(exc), param="messages", code="context_length_exceeded"),
            ) from exc
        except EngineError as exc:
            raise HTTPException(
                status_code=503, detail=error_body(str(exc), "server_error")
            ) from exc
        result.queue_wait_seconds = max(0.0, time.monotonic() - queued - result.compute_seconds)
        return result

    def prepare_params(payload: dict[str, Any], prompt_tokens: int, eng: Any) -> GenerationParams:
        params = translate.sampling_params(
            payload,
            default_max_new_tokens=settings.max_new_tokens,
            default_temperature=settings.default_temperature,
            default_repetition_penalty=settings.default_repetition_penalty,
        )
        requested = payload.get("max_completion_tokens")
        if requested is None:
            requested = payload.get("max_tokens")
        translate.check_budget(prompt_tokens, eng.max_context, requested)
        if requested is None:
            params.max_new_tokens = max(
                1, min(params.max_new_tokens, eng.max_context - prompt_tokens)
            )
        return params

    def render_chat_prompt(eng: Any, messages: list[dict[str, Any]]) -> str:
        try:
            return eng.render_chat(messages)
        except EngineError as exc:
            raise TranslationError(str(exc), param="messages", code="no_chat_template") from exc
        except Exception as exc:  # jinja2 TemplateError and friends
            error: Exception = exc
            if any(m["role"] == "tool" for m in messages):
                # Llama-3-style templates predate the tool role; fold tool
                # results into user turns and try once more.
                try:
                    return eng.render_chat(translate.tool_messages_to_user(messages))
                except Exception as retry_exc:  # pragma: no cover - template specific
                    error = retry_exc
            raise TranslationError(
                f"the chat template rejected these messages: {error}", param="messages"
            ) from error

    # -- streaming ---------------------------------------------------------

    async def stream_choices(
        eng: Any,
        jobs: list[list[int]],
        params: GenerationParams,
        *,
        opening: Callable[[int], dict[str, Any] | None],
        content: Callable[[int, str], dict[str, Any]],
        closing: Callable[[int, GenerationResult], dict[str, Any]],
        object_type: str,
        completion_id: str,
        created: int,
        model: str,
        include_usage: bool,
        warnings: list[str],
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, int, Any]] = asyncio.Queue()
        cancel = threading.Event()
        results: list[GenerationResult] = []
        deadline = time.monotonic() + settings.request_timeout
        try:
            for index, prompt_ids in enumerate(jobs):
                first = opening(index)
                if first is not None:
                    yield sse(first)

                def on_text(piece: str, _i: int = index) -> None:
                    loop.call_soon_threadsafe(queue.put_nowait, ("text", _i, piece))

                queued = time.monotonic()
                try:
                    future = eng.submit(
                        eng.generate,
                        prompt_ids,
                        translate.choice_params(params, index),
                        on_text,
                        cancel,
                    )
                except EngineOverloaded as exc:
                    yield sse(error_body(str(exc), "rate_limit_error", code="engine_busy"))
                    return
                future.add_done_callback(
                    lambda f, _i=index: loop.call_soon_threadsafe(queue.put_nowait, ("done", _i, f))
                )
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        cancel.set()
                        yield sse(
                            error_body(
                                f"The request exceeded {settings.request_timeout:g}s.",
                                "server_error",
                                code="engine_timeout",
                            )
                        )
                        return
                    try:
                        kind, i, item = await asyncio.wait_for(queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        continue
                    if kind == "text":
                        yield sse(content(i, item))
                        continue
                    try:
                        result: GenerationResult = item.result()
                    except PromptTooLong as exc:
                        yield sse(error_body(str(exc), code="context_length_exceeded"))
                        return
                    except EngineError as exc:
                        yield sse(error_body(str(exc), "server_error"))
                        return
                    except Exception as exc:
                        log.exception("generation failed mid-stream")
                        yield sse(error_body(f"{exc.__class__.__name__}: {exc}", "server_error"))
                        return
                    result.queue_wait_seconds = max(
                        0.0, time.monotonic() - queued - result.compute_seconds
                    )
                    results.append(result)
                    final = closing(i, result)
                    if settings.expose_extras:
                        final["x_mobilemoe"] = translate.extras([result], warnings)
                    yield sse(final)
                    break
            if include_usage:
                yield sse(
                    usage_chunk(
                        completion_id, created, model, object_type, translate.usage_of(results)
                    )
                )
            yield DONE
        finally:
            # Reached on normal completion (no-op) and when the client hangs up
            # mid-stream, in which case this stops the generation thread.
            cancel.set()

    # -- OpenAI surface ----------------------------------------------------

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def list_models(request: Request) -> dict[str, Any]:
        return {"object": "list", "data": [_model_card(request, settings, started_at)]}

    @app.get("/v1/models/{model_id:path}", dependencies=[Depends(require_auth)])
    async def get_model(model_id: str, request: Request) -> dict[str, Any]:
        resolve_model(model_id)
        card = _model_card(request, settings, started_at)
        card["id"] = model_id
        card["root"] = model_id
        return card

    @app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
    async def chat_completions(request: Request):
        payload = await _json_body(request)
        body = _validate(ChatCompletionRequest, payload)
        eng = get_engine(request)
        model = resolve_model(body.model)
        warnings = translate.collect_warnings(payload)

        try:
            messages = translate.normalize_messages(
                [m.model_dump(exclude_none=False) for m in body.messages]
            )
            hint = translate.response_format_hint(body.response_format)
            messages = translate.with_system_hint(messages, hint)
            prompt = render_chat_prompt(eng, messages)
            prompt_ids = eng.encode(prompt, rendered=True)
            params = prepare_params(payload, len(prompt_ids), eng)
            n = translate.choice_count(payload)
        except TranslationError as exc:
            raise _translation_http(exc) from exc

        if body.stream:
            completion_id = translate.new_id("chatcmpl")
            created = int(time.time())
            include_usage = bool((body.stream_options or {}).get("include_usage"))
            stream = stream_choices(
                eng,
                [prompt_ids] * n,
                params,
                opening=lambda i: chat_chunk(
                    completion_id,
                    created,
                    model,
                    i,
                    {"role": "assistant", "content": ""},
                    fingerprint=fingerprint,
                ),
                content=lambda i, piece: chat_chunk(
                    completion_id, created, model, i, {"content": piece}
                ),
                closing=lambda i, r: chat_chunk(
                    completion_id, created, model, i, {}, r.finish_reason
                ),
                object_type="chat.completion.chunk",
                completion_id=completion_id,
                created=created,
                model=model,
                include_usage=include_usage,
                warnings=warnings,
            )
            return _sse_response(stream)

        results = [
            await run_generation(eng, prompt_ids, translate.choice_params(params, i))
            for i in range(n)
        ]
        return translate.build_chat_completion(
            results,
            model,
            fingerprint=fingerprint,
            include_extras=settings.expose_extras,
            warnings=warnings,
        )

    @app.post("/v1/completions", dependencies=[Depends(require_auth)])
    async def completions(request: Request):
        payload = await _json_body(request)
        body = _validate(CompletionRequest, payload)
        eng = get_engine(request)
        model = resolve_model(body.model)
        warnings = translate.collect_warnings(payload)

        try:
            prompts = _completion_prompts(body.prompt, eng)
            n = translate.choice_count(payload)
            params: GenerationParams | None = None
            jobs: list[list[int]] = []
            for prompt_ids in prompts:
                params = prepare_params(payload, len(prompt_ids), eng)
                jobs.extend([prompt_ids] * n)
        except TranslationError as exc:
            raise _translation_http(exc) from exc
        assert params is not None

        if body.stream:
            completion_id = translate.new_id("cmpl")
            created = int(time.time())
            include_usage = bool((body.stream_options or {}).get("include_usage"))
            stream = stream_choices(
                eng,
                jobs,
                params,
                opening=lambda i: None,
                content=lambda i, piece: text_chunk(
                    completion_id, created, model, i, piece, fingerprint=fingerprint
                ),
                closing=lambda i, r: text_chunk(
                    completion_id, created, model, i, "", r.finish_reason
                ),
                object_type="text_completion",
                completion_id=completion_id,
                created=created,
                model=model,
                include_usage=include_usage,
                warnings=warnings,
            )
            return _sse_response(stream)

        results = [
            await run_generation(eng, prompt_ids, translate.choice_params(params, i))
            for i, prompt_ids in enumerate(jobs)
        ]
        return translate.build_text_completion(
            results,
            model,
            fingerprint=fingerprint,
            include_extras=settings.expose_extras,
            warnings=warnings,
        )

    # -- ops ---------------------------------------------------------------

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        eng = request.app.state.engine
        return {
            "status": "ok" if eng.ready else "loading",
            "model": settings.served_model_id,
            "repo": settings.model,
            "device": eng.device,
            "dtype": eng.dtype,
            "context_length": eng.max_context,
            "chat_template": eng.has_chat_template,
            "queue_depth": eng.queue_depth,
            "max_queue_depth": settings.max_queue_depth,
            "version": __version__,
        }

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "openai-MobileMoE",
            "version": __version__,
            "model": settings.served_model_id,
            "endpoints": [
                "/v1/models",
                "/v1/chat/completions",
                "/v1/completions",
                "/health",
            ],
        }

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        detail = exc.detail
        if not (isinstance(detail, dict) and "error" in detail):
            detail = error_body(
                str(detail), "invalid_request_error" if exc.status_code < 500 else "server_error"
            )
        return JSONResponse(status_code=exc.status_code, content=detail, headers=exc.headers)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        log.exception("unhandled error serving %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_body(f"{exc.__class__.__name__}: {exc}", "server_error"),
        )

    return app


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _model_card(request: Request, settings: Settings, created: int) -> dict[str, Any]:
    eng = request.app.state.engine
    served = settings.served_model_id
    return {
        "id": served,
        "object": "model",
        "created": created,
        "owned_by": "facebook",
        "root": served,
        "parent": None,
        "x_mobilemoe": {
            "repo": settings.model,
            "device": eng.device,
            "dtype": eng.dtype,
            "context_length": eng.max_context,
            "chat_template": eng.has_chat_template,
        },
    }


def _completion_prompts(prompt: Any, eng: Any) -> list[list[int]]:
    """Accept every ``prompt`` shape the legacy endpoint allows."""
    if prompt is None or prompt == "":
        raise TranslationError("'prompt' must not be empty", param="prompt")
    if isinstance(prompt, str):
        return [eng.encode(prompt)]
    if isinstance(prompt, list):
        if all(isinstance(p, str) for p in prompt):
            if not prompt:
                raise TranslationError("'prompt' must not be empty", param="prompt")
            return [eng.encode(p) for p in prompt]
        if all(isinstance(p, int) and not isinstance(p, bool) for p in prompt):
            return [list(prompt)]
        if all(
            isinstance(p, list) and all(isinstance(t, int) and not isinstance(t, bool) for t in p)
            for p in prompt
        ):
            return [list(p) for p in prompt]
    raise TranslationError(
        "'prompt' must be a string, a list of strings, or token ids", param="prompt"
    )


def _sse_response(stream: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _validate(model_cls: Any, payload: dict[str, Any]) -> Any:
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ())) or None
        raise HTTPException(
            status_code=400,
            detail=error_body(
                f"invalid request: {first.get('msg', 'validation failed')}", param=loc
            ),
        ) from exc


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=400, detail=error_body("request body is not valid JSON")
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400, detail=error_body("request body must be a JSON object")
        )
    return payload
