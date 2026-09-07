"""Drive openai-MobileMoE with the official OpenAI Python SDK.

pip install openai
python examples/openai_client.py
"""

from __future__ import annotations

import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("MOBILEMOE_BASE_URL", "http://127.0.0.1:8000/v1"),
    # Only checked if the server was started with MOBILEMOE_API_KEY.
    api_key=os.environ.get("MOBILEMOE_API_KEY", "not-needed"),
)

MODEL = os.environ.get("MOBILEMOE_MODEL_ID", "MobileMoE-S-QAT")


def chat() -> None:
    print("--- chat ---")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Why are open-source on-device language models great?"},
        ],
        max_tokens=128,
    )
    print(response.choices[0].message.content)
    print(f"finish={response.choices[0].finish_reason} usage={response.usage}")
    # Timings and any warnings about parameters the model cannot honour.
    print("x_mobilemoe =", response.model_extra.get("x_mobilemoe"))


def stream() -> None:
    print("\n--- stream ---")
    for chunk in client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from one to ten in words."}],
        max_tokens=64,
        stream=True,
        stream_options={"include_usage": True},
    ):
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
        if chunk.usage:
            print(f"\nusage={chunk.usage}")


def sampling() -> None:
    print("\n--- sampling (temperature, seed, stop) ---")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Write one line about the sea."}],
        temperature=0.8,
        top_p=0.95,
        seed=7,
        stop=["\n"],
        max_tokens=48,
    )
    print(response.choices[0].message.content)


def completion() -> None:
    print("\n--- legacy completions ---")
    response = client.completions.create(
        model=MODEL, prompt="The three primary colours are", max_tokens=24
    )
    print(response.choices[0].text)


if __name__ == "__main__":
    print("models:", [m.id for m in client.models.list().data])
    chat()
    stream()
    sampling()
    completion()
