"""Chaos-tier fixtures: FailureInjectingLLMClient + a deterministic seed."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import pytest

from kaggle_slayer.agent.llm_client import (
    Response,
    TransientLLMError,
)

DEFAULT_FAILURE_RATE: float = 0.05
DEFAULT_SEED: int = 12345


class FailureInjectingLLMClient:
    """Wraps a scripted client and fails `rate` of calls with TransientLLMError.

    Determinism: a seeded random.Random decides per-call whether to fail.
    Tests pass the same seed for reproducibility.
    """

    def __init__(
        self,
        inner_call: Callable[..., Response],
        *,
        rate: float = DEFAULT_FAILURE_RATE,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._inner_call = inner_call
        self._rate = rate
        self._rng = random.Random(seed)
        self.failures = 0
        self.successes = 0

    def call(self, messages: list[Any], *, tools: Any = None, model: Any = None) -> Response:
        if self._rng.random() < self._rate:
            self.failures += 1
            raise TransientLLMError("injected transient failure (chaos tier)")
        self.successes += 1
        return self._inner_call(messages, tools=tools, model=model)


@pytest.fixture
def chaos_seed() -> int:
    return DEFAULT_SEED
