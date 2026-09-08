# Fidelity notes

Where the OpenAI surface and MobileMoE do not line up, and what the server does
about it. Every case below either produces a clear 4xx or a warning in
`x_mobilemoe.warnings` — never a silent no-op.

## Things the model cannot do

**Tool calling.** MobileMoE is trained for chat, not function calling. `tools`,
`functions`, `tool_choice`, `function_call` and `parallel_tool_calls` are
accepted so agent frameworks do not crash, ignored, and reported. No response
ever contains `tool_calls`; `finish_reason` is never `tool_calls`. If you need
tool use from a tiny model, put a specialist in front of it (for example
[needle-openai](https://github.com/sirmmo/needle-openai)) and hand the
conversation to MobileMoE for the prose.

**Constrained JSON.** There is no grammar-constrained decoding.
`response_format: {"type": "json_object"}` appends *"Respond with a single
valid JSON object and nothing else."* to the system prompt; `json_schema`
appends the schema too. That is a nudge, not a guarantee — parse and validate
the output. `text` is a no-op; any other type is a 400.

**Log-probabilities.** `logprobs` and `top_logprobs` are ignored. `logprobs`
is always `null` in responses.

**Logit bias and OpenAI's penalties.** `logit_bias`, `presence_penalty` and
`frequency_penalty` are ignored. Transformers' `repetition_penalty`
(multiplicative, 1.0 = off) is exposed as an extension and is the closest
substitute.

**Multimodal input.** Content parts other than `text` return 400
`unsupported_content`. `input_text` is treated as text.

## Things that differ from OpenAI's defaults

**Omitted `temperature` means greedy.** OpenAI defaults to 1.0. Meta's model
card decodes with `do_sample=False`, so this server does the same when the
request does not say otherwise (`MOBILEMOE_DEFAULT_TEMPERATURE` changes it).
Greedy output is deterministic: two identical requests give identical text,
which the live suite checks.

**`n` with `seed`.** A fixed seed would make all `n` choices identical, so
choice *i* is generated with `seed + i`. Requests stay reproducible.

**`model` is echoed, not validated.** Unless `MOBILEMOE_STRICT_MODEL` is on,
the name in the request is returned in the response as-is. `GET /v1/models`
tells you what is really loaded.

**`max_tokens` is clamped when omitted.** When a request sends no
`max_tokens`, the default is reduced to whatever fits in the remaining
context. An explicit `max_tokens` that does not fit is a 400
`context_length_exceeded`, matching OpenAI.

## Things about the checkpoints

**Base checkpoints have no chat template.** `facebook/MobileMoE-*-Base` are raw
language models. Chat requests get 400 `no_chat_template`; use
`/v1/completions`.

**The `tool` role.** The SFT/QAT chat template is Llama-3 style and may not
know the `tool` role. Tool messages are first passed through as-is; if the
template rejects them, they are folded into user turns as *"Tool result
(name): ..."* and the prompt is rendered again. Assistant turns that carry
`tool_calls` are rendered with the calls serialized as JSON so a transcript
produced by another model still makes sense.

**Context is 8,192 tokens** for every checkpoint. Prompt tokens are counted
after the chat template is applied, so the template's own tokens count
against it.

**Repetition loops on list-like input.** Asked to summarise 20 rows in which
one label appears five times, MobileMoE-M-QAT repeated that label until the
token cap; with `repetition_penalty: 1.15` it stopped and listed the rows.
Clients that can send the extension should; deployments serving a fixed
client that cannot (a chat front-end that only sends `temperature`) can set
`MOBILEMOE_DEFAULT_REPETITION_PENALTY`. Leave it at 1.0 for code or query
generation, where repeating tokens is the point.

**Tokenizer space cleanup is off.** The checkpoint's tokenizer sets
`clean_up_tokenization_spaces=true`, which would rewrite `SELECT ?s ?label` as
`SELECT?s?label` and `x .` as `x.`. The server decodes with the cleanup
disabled, so text comes back with the spacing the model actually produced.

**Stop tokens.** Generation ends on any of the checkpoint's `eos_token_id`s
plus `<|eot|>`, `<|eot_id|>`, `<|end_of_text|>` and `<|im_end|>` where those
exist in the vocabulary. The terminating token is counted in
`completion_tokens` but never appears in `content`.

**Transformers 5 cannot load the checkpoints.** Its `from_pretrained` builds
the model on the meta device, and `MobileMoERotaryEmbedding.__init__` in the
remote code compares tensor values with `.item()`, which raises
`Tensor.item() cannot be called on meta tensors`. The package pins
`transformers>=4.57,<5`; the engine warns if it finds a 5.x anyway.

## Things about the server

**One request at a time.** Generation runs on a single thread; requests queue
in front of it and are rejected with 429 past `MOBILEMOE_MAX_QUEUE_DEPTH`.
`queue_wait_seconds` in `x_mobilemoe` tells you how long a request sat in that
queue. See [Architecture](architecture.md) for why.

**Disconnecting cancels.** Closing a streaming connection stops generation
within one token; the thread is free for the next request almost immediately.

**Errors mid-stream.** Once streaming has started the HTTP status is already
200. A failure is sent as a `data: {"error": ...}` event, which the OpenAI
SDKs turn into an exception; the stream then ends without `[DONE]`.

**Timings are per process.** `tokens_per_second` is what this container just
did, on this hardware, for this request. It is not a benchmark of the model.
