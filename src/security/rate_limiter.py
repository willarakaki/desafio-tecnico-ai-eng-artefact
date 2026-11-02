"""Sliding-window rate limiter and message payload length guard."""

from __future__ import annotations

import threading
import time
from collections import deque

from pydantic import BaseModel, Field

from src.config import Settings, get_settings


class RateLimitResult(BaseModel):
    """Result of rate limit and payload validation."""

    is_allowed: bool = Field(
        ...,
        description="Whether the request is permitted to proceed",
    )
    reason: str | None = Field(
        default=None,
        description="Explanation when the request is rejected",
    )
    retry_after_seconds: float = Field(
        default=0.0,
        description="Recommended cooldown period in seconds before retrying",
    )
    remaining_requests: int = Field(
        default=0,
        description="Remaining requests permitted in the active time window",
    )


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding-window rate limiter per session_id.

    Enforces maximum requests within a rolling time window and caps
    maximum allowed message payload character length to prevent Denial-of-Wallet attacks.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or get_settings()
        self.max_requests: int = self.settings.rate_limit_max_requests
        self.window_seconds: int = self.settings.rate_limit_window_seconds
        self.max_input_chars: int = self.settings.max_input_chars

        self._sessions: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(
        self,
        session_id: str,
        message: str,
        current_time: float | None = None,
    ) -> RateLimitResult:
        """Validate payload size and sliding-window rate limit for a session.

        Args:
            session_id: Unique identifier for the client session or user.
            message: The incoming raw user text payload.
            current_time: Optional explicit UNIX timestamp (useful for deterministic tests).

        Returns:
            RateLimitResult indicating permission status and metadata.
        """
        # 1. Payload length validation
        if len(message) > self.max_input_chars:
            return RateLimitResult(
                is_allowed=False,
                reason=(
                    f"A mensagem excede o limite máximo permitido de {self.max_input_chars} "
                    f"caracteres (recebido: {len(message)})."
                ),
                retry_after_seconds=0.0,
                remaining_requests=0,
            )

        now = time.time() if current_time is None else current_time
        window_start = now - self.window_seconds

        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = deque()

            session_queue = self._sessions[session_id]

            # Evict timestamps outside the rolling window
            while session_queue and session_queue[0] <= window_start:
                session_queue.popleft()

            # Rate limit check
            if len(session_queue) >= self.max_requests:
                oldest_timestamp = session_queue[0]
                cooldown = max(0.1, round((oldest_timestamp + self.window_seconds) - now, 2))
                return RateLimitResult(
                    is_allowed=False,
                    reason=(
                        "Limite de requisições excedido para esta sessão. "
                        f"Por favor, aguarde {cooldown} segundos antes de enviar outra mensagem."
                    ),
                    retry_after_seconds=cooldown,
                    remaining_requests=0,
                )

            # Record current request timestamp
            session_queue.append(now)
            remaining = max(0, self.max_requests - len(session_queue))

            return RateLimitResult(
                is_allowed=True,
                reason=None,
                retry_after_seconds=0.0,
                remaining_requests=remaining,
            )

    def reset(self, session_id: str | None = None) -> None:
        """Reset the rate limiting state for a specific session or all sessions."""
        with self._lock:
            if session_id is not None:
                self._sessions.pop(session_id, None)
            else:
                self._sessions.clear()

    def get_history(self, session_id: str) -> list[float]:
        """Return a snapshot of active timestamps for a given session."""
        with self._lock:
            return list(self._sessions.get(session_id, []))


_global_rate_limiter: SlidingWindowRateLimiter | None = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter(settings: Settings | None = None) -> SlidingWindowRateLimiter:
    """Return a thread-safe singleton instance of SlidingWindowRateLimiter."""
    global _global_rate_limiter
    with _rate_limiter_lock:
        if _global_rate_limiter is None:
            _global_rate_limiter = SlidingWindowRateLimiter(settings=settings)
        return _global_rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter singleton."""
    global _global_rate_limiter
    with _rate_limiter_lock:
        if _global_rate_limiter is not None:
            _global_rate_limiter.reset()
        _global_rate_limiter = None
