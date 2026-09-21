from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mlops_labs.core.config import Settings


def create_db_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout,
        pool_recycle=settings.pool_recycle,
        pool_pre_ping=True,
    )


async def ping_postgres(engine: AsyncEngine) -> str:
    async with engine.connect() as conn:
        result = await conn.execute(text("SHOW server_version"))
        return str(result.scalar_one())
