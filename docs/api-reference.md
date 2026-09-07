# API reference

All routes accept and return JSON. Errors use OpenAI's envelope:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}}
```

When `MOBILEMOE_API_KEY` is set, every `/v1/*` route requires
`Authorization: Bearer <key>` and answers 401 otherwise. `/health` and `/`
never require auth.

## `POST /v1/chat/completions`

Standard OpenAI chat completion. The messages are rendered through the
checkpoint's own chat template (Llama-3 style, `<|eot|>` end-of-turn) and
generated with Transformers.

### Request

| Field | Support | Notes |
| --- | --- | --- |
| `model` | echoed | Any name unless `MOBILEMOE_STRICT_MODEL`; see [`/v1/models`](#get-v1models). |
| `messages` | yes | Roles `system`, `developer` (→ system), `user`, `assistant`, `tool`. Content may be a string or a list of `text` parts. Image parts → 400 `unsupported_content`. |
| `max_tokens`, `max_completion_tokens` | yes | Default `MOBILEMOE_MAX_NEW_TOKENS` (512), clamped to the remaining context. An explicit value that overflows the context → 400 `context_length_exceeded`, as OpenAI does. |
| `temperature` | yes | 0–2. **Omitted means 0 (greedy)**, per the model card. Any value above 0 turns sampling on. |
| `top_p` | yes | 0–1, used when sampling. |
| `top_k` | extension | Non-negative integer, 0 = off. Used when sampling. |
| `repetition_penalty` | extension | Transformers' multiplicative penalty; 1.0 = off. |
| `seed` | yes | `torch.manual_seed` before generation. With `n > 1` choice *i* uses `seed + i`. |
| `stop` | yes | String or list. Matched on decoded text, the match is not included in the output. |
| `n` | yes | Up to 8. Choices are generated one after another. |
| `stream` | yes | Server-sent events, one chunk per decoded text delta. |
| `stream_options.include_usage` | yes | Appends a final chunk with `usage` and empty `choices`. |
| `response_format` | guided | `json_object` / `json_schema` append an instruction to the system prompt. Not enforced; reported in `x_mobilemoe.warnings`. |
| `tools`, `functions`, `tool_choice`, `function_call`, `parallel_tool_calls` | ignored | Warning. `tool_calls` in assistant history are kept as text so transcripts still render. |
| `logprobs`, `top_logprobs`, `logit_bias` | ignored | Warning. |
| `presence_penalty`, `frequency_penalty` | ignored | Warning; use `repetition_penalty`. |
| `user`, `metadata`, `store`, anything else | accepted | Unknown fields never cause a 400. |

### Response

```json
{
  "id": "chatcmpl-6f1c...",
  "object": "chat.completion",
  "created": 1788000000,
  "model": "MobileMoE-S-QAT",
  "system_fingerprint": "fp_mobilemoe_0.1.0",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Open-source on-device models ...", "refusal": null},
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 32, "completion_tokens": 91, "total_tokens": 123},
  "x_mobilemoe": {
    "compute_seconds": 4.12,
    "queue_wait_seconds": 0.0,
    "tokens_per_second": 22.1,
    "warnings": ["'logprobs' ignored: token log-probabilities are not exposed"]
  }
}
```

`finish_reason` is `stop` when the model emitted an end-of-turn token or a
`stop` string matched, and `length` when `max_tokens` was reached. `usage` is
exact: prompt tokens are counted after the chat template is applied,
completion tokens are the ids actually generated (including the end-of-turn
token).

### Streaming

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","system_fingerprint":"fp_mobilemoe_0.1.0","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"content":"Open"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"content":"-source"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}],"x_mobilemoe":{...}}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[],"usage":{"prompt_tokens":32,"completion_tokens":91,"total_tokens":123}}

data: [DONE]
```

Each choice opens with a role chunk, streams content deltas, and closes with
an empty delta carrying `finish_reason` (and the `x_mobilemoe` block for that
choice). With `n > 1` the choices arrive one after another, each with its own
`index`. The `usage` chunk appears only when `stream_options.include_usage` is
true. An error after the stream has started is sent as
`data: {"error": {...}}`, which the OpenAI SDKs raise as an `APIError`.

Content deltas never split a multi-byte character, and text that could be the
start of a `stop` string is held back until it either matches or diverges.

## `POST /v1/completions`

Legacy text completion. Same sampling fields as above, minus `messages`.

| Field | Notes |
| --- | --- |
| `prompt` | A string, a list of strings (one set of choices per prompt, indices run `prompt × n + k`), a list of token ids, or a list of token-id lists. Tokenized with the model's BOS token added. |
| `echo`, `suffix`, `best_of` | Ignored, with a warning. |

```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1788000000,
  "model": "MobileMoE-S-QAT",
  "system_fingerprint": "fp_mobilemoe_0.1.0",
  "choices": [{"index": 0, "text": " red, blue and yellow.", "logprobs": null, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 7, "completion_tokens": 6, "total_tokens": 13},
  "x_mobilemoe": {"compute_seconds": 0.31, "queue_wait_seconds": 0.0, "tokens_per_second": 19.4}
}
```

Streaming emits `text_completion` chunks with `choices[].text` deltas. This is
the endpoint to use with the `*-Base` checkpoints, which have no chat
template; `/v1/chat/completions` returns 400 `no_chat_template` for them.

## `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    {
      "id": "MobileMoE-S-QAT",
      "object": "model",
      "created": 1788000000,
      "owned_by": "facebook",
      "root": "MobileMoE-S-QAT",
      "parent": null,
      "x_mobilemoe": {
        "repo": "facebook/MobileMoE-S-QAT",
        "device": "cpu",
        "dtype": "float32",
        "context_length": 8192,
        "chat_template": true
      }
    }
  ]
}
```

`id` is `MOBILEMOE_MODEL_ID`, defaulting to the last segment of the repo id.
Requests may also address the model by its full repo id or as `mobilemoe`.

## `GET /v1/models/{id}`

The same card with `id` and `root` set to the requested name. 404 only under
`MOBILEMOE_STRICT_MODEL`.

## `GET /health`

```json
{
  "status": "ok",
  "model": "MobileMoE-S-QAT",
  "repo": "facebook/MobileMoE-S-QAT",
  "device": "cpu",
  "dtype": "float32",
  "context_length": 8192,
  "chat_template": true,
  "queue_depth": 0,
  "max_queue_depth": 16,
  "version": "0.1.0"
}
```

`status` is `"loading"` until the model has been downloaded, loaded and warmed
up. Never requires auth.

## The `x_mobilemoe` block

Present on every non-streaming response, on each choice's final streaming
chunk, and on the model card. Disable with `MOBILEMOE_EXPOSE_EXTRAS=false` or
`--no-extras`.

| Field | Meaning |
| --- | --- |
| `compute_seconds` | Wall time spent inside `generate` for this response. |
| `queue_wait_seconds` | Time the request waited for the generation thread. |
| `tokens_per_second` | `completion_tokens / compute_seconds`. |
| `warnings` | Present only when some request field was accepted but ignored. |

The official SDKs expose it through `response.model_extra["x_mobilemoe"]`
(Python) or as an extra property on the parsed object (Node).

## Error codes

| Status | `code` | When |
| --- | --- | --- |
| 400 | — | Invalid JSON, invalid field types or ranges, empty prompt. |
| 400 | `unsupported_content` | A non-text content part (image, audio, file). |
| 400 | `context_length_exceeded` | Prompt too long, or `max_tokens` + prompt over the context. |
| 400 | `no_chat_template` | Chat request against a `*-Base` checkpoint. |
| 401 | `invalid_api_key` | `MOBILEMOE_API_KEY` set and the bearer token is missing or wrong. |
| 404 | `model_not_found` | Unknown `model` under `MOBILEMOE_STRICT_MODEL`. |
| 429 | `engine_busy` | `MOBILEMOE_MAX_QUEUE_DEPTH` requests already waiting. |
| 503 | `not_ready` | Model still loading. |
| 503 | — | Generation failed inside torch / Transformers. |
| 504 | `engine_timeout` | Queued + generation time exceeded `MOBILEMOE_REQUEST_TIMEOUT`. |
