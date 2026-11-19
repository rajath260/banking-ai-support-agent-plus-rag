"""Hugging Face LLM connection helpers."""

import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

MODEL_ID = os.getenv("HF_MODEL", "openai/gpt-oss-120b")


def get_client() -> InferenceClient:
    """Create an authenticated Hugging Face InferenceClient on demand."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Create a .env file in the repository root "
            "and add HF_TOKEN=your_huggingface_token."
        )
    return InferenceClient(token=token, provider="auto")


def ask_llm(question: str) -> str:
    """Send one question to the configured Hugging Face chat model."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    response = get_client().chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful banking customer support assistant. "
                    "Explain general banking concepts clearly. Never invent "
                    "customer account details, balances, or transactions. "
                    "For account-specific information, say it must be retrieved "
                    "from an approved banking tool."
                ),
            },
            {"role": "user", "content": question.strip()},
        ],
        max_tokens=250,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    prompt = "Explain the difference between a savings account and a current account."
    print(f"Model: {MODEL_ID}")
    print("\nUser:", prompt)
    print("\nAssistant:", ask_llm(prompt))
