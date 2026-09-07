# openai-MobileMoE

An OpenAI-compatible HTTP API in front of Meta's
[MobileMoE](https://huggingface.co/facebook/MobileMoE-S-QAT) — sparse
mixture-of-experts language models with 272M–922M *active* parameters, designed
to run on phones ([paper](https://arxiv.org/abs/2605.27358)). Point any OpenAI
client at it and chat with a model whose INT4 weights fit in 0.7 GB. Serves
every checkpoint in the family, on CPU or CUDA.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="MobileMoE-S-QAT",
    messages=[{"role": "user", "content": "Why are open-source on-device language models great?"}],
)
print(response.choices[0].message.content)
```

!!! warning "The weights are gated and noncommercial"
    MobileMoE is published under Meta's
    [FAIR Noncommercial Research License](https://huggingface.co/facebook/MobileMoE-S-QAT/blob/main/LICENSE).
    Accept it on the model page, create a read token and pass it as `HF_TOKEN`
    — the [quickstart](quickstart.md) walks through it. This server is MIT; the
    weights are not.

## What it is good at

MobileMoE is a general chat model in a very small compute budget. Each token
touches 4 of 60 routed experts plus one shared expert, so the S checkpoint
runs on 272M parameters' worth of compute while drawing on 1.3B parameters'
worth of knowledge. Meta reports that at equal INT4 memory it matches or beats
dense on-device models with 2–4× fewer FLOPs.

- **Chat and instruction following** — the SFT and QAT rows carry a chat
  template with system, user and assistant turns.
- **Genuine streaming** — tokens leave the server as they are sampled.
- **Small enough to self-host anywhere** — the CPU image runs on x86 servers
  and ARM boards alike; a CUDA image is published for GPUs.
- **The whole family** — switch between the nine checkpoints with one
  environment variable.

## What it is not

It has no tool-calling head, no constrained decoding and no vision. The
[fidelity notes](fidelity.md) list every OpenAI parameter that is accepted but
cannot be honoured; each one is also reported back on the response in
`x_mobilemoe.warnings` so nothing fails silently.

## Where to go next

<div class="grid cards" markdown>

- **[Quickstart](quickstart.md)** — accept the license, run it in Docker, make
  your first call.
- **[API reference](api-reference.md)** — endpoints, parameter support matrix,
  the `x_mobilemoe` extension block.
- **[Fidelity notes](fidelity.md)** — every place the OpenAI mapping is lossy.
- **[Configuration](configuration.md)** — environment variables, CLI flags,
  memory per checkpoint.
- **[Architecture](architecture.md)** — the single generation thread, how
  streaming and stop strings work.
- **[Development](development.md)** — running the tests, releasing.

</div>

## License

MIT. The MobileMoE weights are licensed by Meta under the FAIR Noncommercial
Research License.
