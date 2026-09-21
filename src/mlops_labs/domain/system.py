from dataclasses import dataclass


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
