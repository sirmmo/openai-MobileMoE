# Architecture

```
client ──HTTP──▶ server.py ──▶ translate.py ──▶ engine.py ──▶ torch / Transformers
                   │              (pure)          │
                   │                              └─ one worker thread
                   └─ streaming.py (SSE)             ├─ CallbackStreamer
                                                     └─ IncrementalDecoder
```

| Module | Responsibility |
| --- | --- |
| `config.py` | `Settings` dataclass read from `MOBILEMOE_*` variables; the list of known checkpoints. |
| `schemas.py` | Permissive pydantic request models and the error envelope. |
| `translate.py` | Every OpenAI ↔ prompt/parameter/response mapping, as pure functions. |
| `engine.py` | Loads the model, owns the generation thread, streams and decodes tokens. |
| `streaming.py` | Builds the SSE chunk dictionaries. |
| `server.py` | FastAPI routes, auth, error mapping, the async side of streaming. |
| `__main__.py` | CLI. |

## Request flow

1. The route parses the JSON body itself (so an invalid body is a 400 in
   OpenAI's envelope, not FastAPI's 422) and validates it with a permissive
   pydantic model.
2. `translate.normalize_messages` reduces the messages to `{role, content}`
   dicts, folding text parts, mapping `developer` to `system`, and rendering
   `tool_calls` history as text. `response_format` becomes a system-prompt
   hint.
3. `engine.render_chat` applies the checkpoint's own chat template with the
   generation prompt appended. If the template rejects the `tool` role, tool
   messages are folded into user turns and rendering is retried.
4. `engine.encode` tokenizes. The rendered template already spells out
   `<|begin_of_text|>`, so BOS is not added twice; raw `/v1/completions`
   prompts do get it.
5. `translate.sampling_params` reads the sampling fields and
   `translate.check_budget` applies OpenAI's context rule.
6. The request is submitted to the engine thread and awaited with a deadline.
7. The result is rendered by `translate.build_*` or streamed.

## One generation thread

`MobileMoEEngine` runs every `generate` call on a single
`ThreadPoolExecutor(max_workers=1)`. Two reasons:

- Transformers' `generate` on one model instance is not safe to call from
  several threads at once; the KV cache and the streamer are per call, but
  nothing else is.
- On CPU, two concurrent generations share the same cores and each runs at
  half speed. Serializing them gives the same throughput with predictable
  latency, and it lets `queue_wait_seconds` tell you honestly how loaded the
  server is.

`submit` counts pending requests and raises `EngineOverloaded` (429) past
`max_queue_depth`. Timeouts (`request_timeout`) cover queue time plus
generation; on timeout the request's cancel event is set so the thread does
not keep working for nobody.

Model loading happens in the lifespan hook via `asyncio.to_thread`, and
`/health` reports `"loading"` until it finishes. Loading ends with one forward
pass on the BOS token: the QAT checkpoints materialize their INT4 weights into
the serving dtype on first use, and that belongs at startup, not in the first
user's latency.

## Streaming

Streaming is real, not a finished string cut into pieces.

```
generate() ──put(token)──▶ CallbackStreamer ──▶ IncrementalDecoder ──on_text──▶ loop.call_soon_threadsafe
                                                                                        │
     route ◀── asyncio.Queue ◀──────────────────────────────────────────────────────────┘
```

- `CallbackStreamer` is a Transformers `BaseStreamer`; `generate` calls
  `put()` with every sampled token (the first call carries the prompt and is
  skipped).
- `IncrementalDecoder` keeps the growing id list and re-decodes a trailing
  window each time, only releasing text once it no longer ends in a
  replacement character — so a three-token emoji arrives whole. It also
  matches `stop` strings on the released text and holds back any suffix that
  is a prefix of a stop string until it matches or diverges. On a match it
  flags `stop_hit`, and a `StoppingCriteria` watching the same event ends
  generation on the next step.
- Deltas cross into the event loop with `call_soon_threadsafe` onto an
  `asyncio.Queue`; the route drains it and writes SSE chunks. The future's
  done-callback lands in the same queue after the last delta, in order.
- If the client disconnects, the async generator's `finally` sets the cancel
  event; the `StoppingCriteria` sees it on the next token and generation
  stops.

The non-streaming path uses the same decoder without a callback, so streamed
and non-streamed text for a greedy request are byte-for-byte identical (the
live suite checks that).

## Stop tokens

At load time the engine collects every id from `generation_config.eos_token_id`
and `tokenizer.eos_token_id`, then adds `<|eot|>`, `<|eot_id|>`,
`<|end_of_text|>` and `<|im_end|>` if the tokenizer knows them. All of them
are passed to `generate` as `eos_token_id`, so a checkpoint whose
`generation_config` only lists `<|end_of_text|>` still stops at the end of
the assistant turn.
