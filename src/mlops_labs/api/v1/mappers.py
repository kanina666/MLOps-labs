from mlops_labs.domain.system import SystemHealth
from mlops_labs.schemas import ComponentHealth, HealthResponse


def to_health_response_dto(system_health: SystemHealth) -> HealthResponse:
    return HealthResponse(
        status="ok" if system_health.healthy else "degraded",
        checks={
            item.name: ComponentHealth(
                status="ok" if item.healthy else "error",
                response_time_ms=item.latency_ms,
                version=item.version,
                error=item.error,
            )
            for item in system_health.checks
        },
    )
