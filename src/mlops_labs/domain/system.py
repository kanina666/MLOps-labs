from dataclasses import dataclass
from typing import Protocol

from typing_extensions import runtime_checkable


@dataclass(frozen=True, slots=True)
class DependencyHealth:
    name: str
    healthy: bool
    latency_ms: float
    version: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SystemHealth:
    healthy: bool
    checks: tuple[DependencyHealth, ...]


@runtime_checkable
class HealthCheck(Protocol):
    async def check(self) -> DependencyHealth: ...
