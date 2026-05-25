from __future__ import annotations

import os
import time
from pathlib import Path

SYSTEM_MARKER = "<<<DUAL_STREAM_SYSTEM>>>"
USER_MARKER = "<<<DUAL_STREAM_USER>>>"


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() == name and raw_value.strip():
            return raw_value.strip().strip('"').strip("'")
    return None


class OpenAIBackend:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key or _env_value("OPENAI_API_KEY")
        self.base_url = base_url or _env_value("OPENAI_BASE_URL")
        self.max_tokens = max_tokens or int(_env_value("OPENAI_MAX_TOKENS") or "128")

    def complete(self, context: str, model: str) -> str:
        """Call an OpenAI-compatible chat model with retries. [INV-4]"""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for OpenAIBackend") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        delay = 1.0
        last_error: Exception | None = None
        for _ in range(5):
            try:
                messages = self._messages(context)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                return content or ""
            except Exception as exc:
                last_error = exc
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError("OpenAIBackend failed after 5 retries") from last_error

    def _messages(self, context: str) -> list[dict[str, str]]:
        if context.startswith(SYSTEM_MARKER) and USER_MARKER in context:
            system_and_user = context[len(SYSTEM_MARKER) :].lstrip()
            system, user = system_and_user.split(USER_MARKER, 1)
            return [
                {"role": "system", "content": system.strip()},
                {"role": "user", "content": user.strip()},
            ]
        return [{"role": "user", "content": context}]
