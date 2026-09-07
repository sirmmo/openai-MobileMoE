# Quickstart

## 1. Get access to the weights

The MobileMoE repositories are gated. Once per HuggingFace account:

1. Open [facebook/MobileMoE-S-QAT](https://huggingface.co/facebook/MobileMoE-S-QAT)
   and accept the FAIR Noncommercial Research License. Access is granted
   automatically, and it covers all nine checkpoints.
2. Create a **read** token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Everything below expects that token in `HF_TOKEN`.

## 2. Run the server

=== "Docker"

    ```bash
    docker run -d --name mobilemoe -p 8000:8000 \
      -e HF_TOKEN=hf_... \
      -v mobilemoe-cache:/cache \
      ghcr.io/sirmmo/openai-mobilemoe:latest
    ```

    The volume keeps the downloaded checkpoint across restarts. Images are
    published for `linux/amd64` and `linux/arm64`.

=== "Docker Compose"

    ```bash
    git clone https://github.com/sirmmo/openai-MobileMoE
    cd openai-MobileMoE
    cp .env.example .env      # set HF_TOKEN, optionally MOBILEMOE_MODEL
    docker compose up -d
    ```

=== "pip"

    ```bash
    # CPU-only torch is a quarter of the size of the default CUDA build.
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install openai-mobilemoe
    HF_TOKEN=hf_... openai-mobilemoe --port 8000
    ```

    On a CUDA machine skip the first line; `pip install openai-mobilemoe`
    brings in the GPU build and the server picks the GPU up automatically.

=== "GPU container"

    ```bash
    docker run -d --gpus all -p 8000:8000 \
      -e HF_TOKEN=hf_... -v mobilemoe-cache:/cache \
      ghcr.io/sirmmo/openai-mobilemoe:latest-cuda
    ```

Wait for the model:

```bash
curl -s localhost:8000/health
# {"status":"ok","model":"MobileMoE-S-QAT","repo":"facebook/MobileMoE-S-QAT",
#  "device":"cpu","dtype":"float32","context_length":8192,"chat_template":true,
#  "queue_depth":0,"max_queue_depth":16,"version":"0.1.0"}
```

`status` reads `"loading"` while the checkpoint downloads (0.7 GB for S-QAT)
and its INT4 weights are dequantized. Requests during that window get a 503
with `code: "not_ready"`.

## 3. Make a request

=== "Python (openai)"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

    response = client.chat.completions.create(
        model="MobileMoE-S-QAT",
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Why are open-source on-device language models great?"},
        ],
        max_tokens=200,
    )
    print(response.choices[0].message.content)
    print(response.usage)
    ```

=== "Streaming"

    ```python
    stream = client.chat.completions.create(
        model="MobileMoE-S-QAT",
        messages=[{"role": "user", "content": "Count to ten in words."}],
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
        if chunk.usage:
            print("\n", chunk.usage)
    ```

=== "curl"

    ```bash
    curl -s localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "MobileMoE-S-QAT",
        "messages": [{"role": "user", "content": "Why are open-source on-device language models great?"}],
        "max_tokens": 200
      }'
    ```

=== "Node"

    ```js
    import OpenAI from "openai";

    const client = new OpenAI({ baseURL: "http://127.0.0.1:8000/v1", apiKey: "not-needed" });
    const res = await client.chat.completions.create({
      model: "MobileMoE-S-QAT",
      messages: [{ role: "user", content: "Why are open-source on-device language models great?" }],
    });
    console.log(res.choices[0].message.content);
    ```

The `model` field is echoed back and, by default, not checked — clients that
hard-code `gpt-4o-mini` keep working. Set `MOBILEMOE_STRICT_MODEL=true` to 404
on names other than the served one.

## 4. Pick a checkpoint

```bash
docker run -d -p 8000:8000 -e HF_TOKEN=hf_... -e MOBILEMOE_MODEL=facebook/MobileMoE-L-QAT \
  -v mobilemoe-cache:/cache ghcr.io/sirmmo/openai-mobilemoe:latest
```

| Checkpoint | Active / total | Download | Chat |
| --- | --- | --- | --- |
| `facebook/MobileMoE-S-QAT` (default) | 272M / 1.3B | 0.7 GB | yes |
| `facebook/MobileMoE-M-QAT` | 528M / 2.8B | 1.6 GB | yes |
| `facebook/MobileMoE-L-QAT` | 922M / 5.3B | 3.0 GB | yes |
| `facebook/MobileMoE-{S,M,L}-SFT` | as above | 2.6 / 5.6 / 10.6 GB | yes |
| `facebook/MobileMoE-{S,M,L}-Base` | as above | 2.6 / 5.6 / 10.6 GB | completions only |

See [Configuration](configuration.md#memory) for RAM per checkpoint and dtype.
A local directory (for example a BF16 export) works too:
`MOBILEMOE_MODEL=/models/MobileMoE-S-QAT-bf16`.

## 5. Lock it down

```bash
docker run -d -p 8000:8000 -e HF_TOKEN=hf_... -e MOBILEMOE_API_KEY=sk-local-secret ...
```

Every `/v1/*` route then requires `Authorization: Bearer sk-local-secret`;
`/health` stays open for orchestrators.
