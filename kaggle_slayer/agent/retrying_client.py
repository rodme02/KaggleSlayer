"""RetryingLLMClient — wraps any LLMClient with TransientLLMError retry.

The harness handles transient failures one of two ways:
  1. The provider client (GeminiClient) has built-in retry — happy path.
  2. Wrap any LLMClient in RetryingLLMClient for retry portability —
     used by the chaos tier to verify spec §11.3 / §13.

Retries only on TransientLLMError. Any other exception bubbles up
immediately. Backoff is exponential with a configurable base, mirroring
GeminiClient's sleep schedule (base * 2**attempt).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from kaggle_slayer.agent.llm_client import (
    LLMClient,
    Message,
    Response,
    TransientLLMError,
)


class RetryingLLMClient:
    """Wraps an LLMClient and retries TransientLLMError with exponential backoff.

    The wrapped client is invoked up to ``retry_max + 1`` times total. After
    each transient failure (except the final one), the adapter sleeps for
    ``retry_base_delay_s * (2 ** attempt)`` seconds before retrying. Non-
    transient exceptions propagate immediately without retry.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        retry_max: int = 3,
        retry_base_delay_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._inner = inner
        self._retry_max = retry_max
        self._retry_base_delay_s = retry_base_delay_s
        self._sleep = sleep

    def call(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> Response:
        last_exc: TransientLLMError | None = None
        for attempt in range(self._retry_max + 1):
            try:
                return self._inner.call(messages, tools=tools, model=model)
            except TransientLLMError as e:
                last_exc = e
                if attempt >= self._retry_max:
                    break
                delay = self._retry_base_delay_s * (2 ** attempt)
                self._sleep(delay)
        assert last_exc is not None  # noqa: S101 — invariant: loop only exits via return or transient
        raise last_exc
