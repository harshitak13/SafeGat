"""
llm/llm_gateway.py

LLMGateway — thin wrapper around a callable LLM backend.

Features
--------
- Pluggable backend: pass any callable(prompt: str) -> str.
  The default backend is a no-op mock for testing without an API key.
- Structured JSON parsing with regex fallback.
- Automatic retry (max_retries) on parse failure.
- Thread-safe proactive rate limiting (prevents simultaneous bursts).
- Exponential backoff on HTTP 429 / rate-limit errors.

Rate-limit defaults
-------------------
Groq free tier (llama-3.1-8b-instant): ~30 RPM / ~6 000 TPM.
With 2 parallel calls per step the effective rate is 2× RPM, so a 4 s
minimum gap between consecutive calls keeps usage within the free tier.
Adjust ``min_call_interval_s`` for paid tiers.

Sources
-------
- SafeGAT-LLM scaffold (llm/llm_gateway.py)
- iLLM-TSC2 (llm_agents/grid_tsc_agent.py) — rate-limiter design
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Callable, Optional

from loguru import logger

from .types import LLMDecision


# ── Shared rate limiter ────────────────────────────────────────────────────────

class _RateLimiter:
    """
    Thread-safe minimum-interval rate limiter.

    Each thread stamps the next allowed call time BEFORE releasing the lock,
    then sleeps independently — so other threads can compute their own wait
    concurrently rather than piling up on the lock.
    """

    def __init__(self, min_interval: float) -> None:
        self._lock          = threading.Lock()
        self._last_call_ts  = 0.0
        self._min_interval  = min_interval

    def acquire(self, label: str = "") -> None:
        with self._lock:
            now  = time.monotonic()
            wait = self._min_interval - (now - self._last_call_ts)
            self._last_call_ts = now + max(wait, 0)
        if wait > 0:
            logger.debug(
                f"[RateLimiter{f'-{label}' if label else ''}] "
                f"waiting {wait:.2f}s (min_interval={self._min_interval}s)"
            )
            time.sleep(wait)


class LLMGateway:
    """
    Calls an LLM backend and parses the JSON response into an LLMDecision.

    Parameters
    ----------
    backend              : callable(str) -> str — your LLM call; defaults to mock
    max_retries          : int   — parse-failure retries (default 2)
    min_call_interval_s  : float — seconds between consecutive API calls (default 4.0)
    max_backoff_retries  : int   — retries on HTTP 429 (default 5)
    backoff_wait_s       : float — base wait per retry on rate-limit (default 30.0)
    """

    def __init__(
        self,
        backend:             Optional[Callable[[str], str]] = None,
        max_retries:         int   = 2,
        min_call_interval_s: float = 4.0,
        max_backoff_retries: int   = 5,
        backoff_wait_s:      float = 30.0,
    ) -> None:
        self.backend             = backend or self._mock_backend
        self.max_retries         = max_retries
        self.max_backoff_retries = max_backoff_retries
        self.backoff_wait_s      = backoff_wait_s
        self._rate_limiter       = _RateLimiter(min_call_interval_s)

    # ── Mock ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _mock_backend(prompt: str) -> str:   # noqa: ARG002
        return '{"decision": "accept", "final_phase": 0, "reason": "mock backend default"}'

    # ── Public API ─────────────────────────────────────────────────────────────

    def query(self, prompt: str, label: str = "") -> LLMDecision:
        """
        Send ``prompt`` to the backend and return a parsed LLMDecision.

        Parameters
        ----------
        prompt : str  — the full LLM prompt
        label  : str  — optional label for logging (e.g. junction ID)

        Raises
        ------
        ValueError if all retries fail to produce parseable JSON.
        """
        last_text = ""
        for attempt in range(self.max_retries + 1):
            text = self._call_with_backoff(prompt, label)
            last_text = text
            parsed = self._parse_json(text)
            if parsed is not None:
                return LLMDecision(
                    decision    = str(parsed.get("decision", "accept")).strip().lower(),
                    final_phase = int(parsed.get("final_phase", 0)),
                    reason      = str(parsed.get("reason", "")),
                    raw_text    = text,
                    parsed      = parsed,
                )
            logger.warning(
                f"[LLMGateway{f'-{label}' if label else ''}] "
                f"parse failed on attempt {attempt + 1}/{self.max_retries + 1}"
            )
        raise ValueError(f"Unable to parse LLM output after {self.max_retries + 1} attempts: {last_text!r}")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _call_with_backoff(self, prompt: str, label: str) -> str:
        """Invoke backend with proactive rate limiting + exponential backoff on 429."""
        for attempt in range(self.max_backoff_retries):
            self._rate_limiter.acquire(label)
            try:
                return self.backend(prompt)
            except Exception as exc:
                if "429" in str(exc) or "rate_limit" in str(exc).lower():
                    wait = self.backoff_wait_s * (attempt + 1)
                    logger.warning(
                        f"[LLMGateway-{label}] rate-limit hit "
                        f"(attempt {attempt + 1}/{self.max_backoff_retries}), "
                        f"waiting {wait:.0f}s"
                    )
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(
            f"[LLMGateway-{label}] exhausted {self.max_backoff_retries} backoff retries"
        )

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """Try strict JSON parse, then regex-extract the first {...} block."""
        # Strip markdown fences if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
