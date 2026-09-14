# Notice:
# Carter Saar. Copyright (C) 2025 State of Utah
# Licensed under Apache License, Version 2.0 (Apache v2). This program is distributed on an "AS IS" BASIS, WITHOUT ANY WARRANTY OR CONDITIONS OF ANY KIND, either express or implied. See the Apache License, Version 2.0 (Apache v2) for more details.

"""
Request pacing for free-tier Gemini keys.

An assessment issues one call per evidence source plus one per risk category,
back to back, which exceeds a free-tier per-minute allowance outright. Pacing
the run under the quota is therefore the primary defence; the retry
configuration alongside it handles whatever still gets through.

A sliding window rather than fixed spacing, so a short run still bursts: the
first N requests go straight through, and a caller waits only once the window is
full, and then only until the oldest request ages out.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque

# Observed free-tier quotas, by model. Google does not publish these, and both
# limits matter: the per-minute figure paces a run, while the per-day figure
# decides whether a model is usable for this app at all. A full assessment costs
# roughly one call per risk category, so a model with a small daily allowance
# supports only a run or two before it is exhausted.
FREE_TIER_RPM = {
    "gemini-3.8-flash": 5,
    "gemini-3.5-flash-lite": 15,
}

# Known per-day request caps. Absent means unmeasured, not unlimited.
FREE_TIER_RPD = {
    "gemini-3.8-flash": 20,
}

# Applied to any model not measured above. Chosen to match the tightest limit
# actually observed, so an unknown model is paced safely rather than optimistically.
DEFAULT_FREE_TIER_RPM = 5

WINDOW_SECONDS = 60.0

# Leaves a little headroom inside the window, since the server's idea of "a
# minute" and ours will not align exactly.
SAFETY_MARGIN_SECONDS = 0.5


def rpm_for(model_id: str | None) -> int:
    return FREE_TIER_RPM.get(model_id or "", DEFAULT_FREE_TIER_RPM)


def rpd_for(model_id: str | None) -> int | None:
    """Known per-day cap, or None when we have not measured it."""
    return FREE_TIER_RPD.get(model_id or "")


def estimate_calls(url_count: int, pdf_count: int, category_count: int) -> int:
    """
    Model calls one run will make.

    PDFs are read locally and cost nothing, so only URLs and categories count.
    """
    return max(0, url_count) + max(0, category_count)


def daily_quota_warning(model_id, url_count, pdf_count, category_count):
    """
    A user-facing warning when a run will not fit inside the model's daily quota.

    Returns None when the model has room, or has no known per-day limit.
    """
    cap = rpd_for(model_id)
    if cap is None:
        return None
    needed = estimate_calls(url_count, pdf_count, category_count)
    if needed > cap:
        return (
            f"{model_id} allows about {cap} requests per day on the free tier and this run "
            f"needs roughly {needed}, so it will stop partway through. Pick a model with a "
            f"larger daily allowance, or reduce the number of risk categories or URLs."
        )
    runs_per_day = cap // max(1, needed)
    if runs_per_day <= 2:
        return (
            f"{model_id} allows about {cap} requests per day on the free tier and this run "
            f"uses roughly {needed} of them, so you can run it about {runs_per_day} time(s) "
            f"per day. A lite model has a much larger daily allowance."
        )
    return None


class RateLimiter:
    """
    Thread-safe sliding-window limiter for one model's quota.

    The clock and sleep function are injectable so tests can drive the waiting
    path without waiting. They must move together, or acquire() will spin.
    """

    def __init__(self, rpm: int, window_seconds: float = WINDOW_SECONDS,
                 monotonic=time.monotonic, sleep=time.sleep):
        self.rpm = max(1, int(rpm))
        self.window = window_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def time_until_slot(self) -> float:
        """Seconds a caller would wait right now. Does not consume a slot."""
        now = self._monotonic()
        with self._lock:
            self._evict(now)
            if len(self._times) < self.rpm:
                return 0.0
            return max(0.0, (self._times[0] + self.window) - now + SAFETY_MARGIN_SECONDS)

    def acquire(self) -> float:
        """Block until a slot is free, then consume it. Returns seconds waited."""
        waited = 0.0
        # Bounded so a misbehaving clock cannot hang a run forever; each pass
        # either returns immediately or sleeps at least until the window moves.
        for _ in range(self.rpm + 2):
            delay = self.time_until_slot()
            if delay <= 0:
                break
            self._sleep(delay)
            waited += delay
        with self._lock:
            self._times.append(self._monotonic())
        return waited

    async def acquire_async(self) -> float:
        """
        Async form of acquire(), for use inside the graph orchestrator.

        Uses asyncio.sleep so waiting for a quota slot does not block the event
        loop and stall progress events.
        """
        waited = 0.0
        for _ in range(self.rpm + 2):
            delay = self.time_until_slot()
            if delay <= 0:
                break
            await asyncio.sleep(delay)
            waited += delay
        with self._lock:
            self._times.append(self._monotonic())
        return waited

    def _evict(self, now: float) -> None:
        cutoff = now - self.window
        while self._times and self._times[0] <= cutoff:
            self._times.popleft()


def limiter_for(model_id: str | None) -> RateLimiter:
    """Build a limiter sized to one model's measured free-tier quota."""
    return RateLimiter(rpm_for(model_id))


def classify_quota_error(exc) -> tuple[str, str]:
    """
    Classify a 429 into ("minute" | "day" | "other", user-facing message).

    Only one is worth waiting for: a per-minute quota clears quickly, while a
    per-day quota will not clear within the run at all.
    """
    text = str(exc)
    if "PerDay" in text or "RequestsPerDay" in text:
        return (
            "day",
            "The free-tier daily request limit for this model has been reached. It resets "
            "every 24 hours. Switch to a lite model, which has a much larger daily "
            "allowance, or try again tomorrow.",
        )
    if "PerMinute" in text or "RequestsPerMinute" in text:
        return (
            "minute",
            "The free-tier per-minute request limit was hit. The run paces itself, so this "
            "should be rare; it usually means another process is sharing the same key.",
        )
    return ("other", f"The Gemini API rejected a request: {text[:200]}")
