from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from mlops_labs.core.config import Settings
from mlops_labs.db.models import Base


def create_db_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout,
        pool_recycle=settings.pool_recycle,
        pool_pre_ping=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        # Serialize schema updates when API and worker start together
        if conn.dialect.name == "postgresql":
            await conn.execute(text("SELECT pg_advisory_xact_lock(724091)"))
        await conn.run_sync(Base.metadata.create_all)
        columns = await conn.run_sync(lambda sync: inspect(sync).get_columns("tasks"))
        if "attempts" not in {column["name"] for column in columns}:
            await conn.execute(
                text("ALTER TABLE tasks ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
            )


async def ping_postgres(engine: AsyncEngine) -> str:
    async with engine.connect() as conn:
        result = await conn.execute(text("SHOW server_version"))
        return str(result.scalar_one())
