from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mlops_labs.core.config import get_settings

engine: AsyncEngine = create_async_engine(get_settings().database_url, pool_pre_ping=True)


async def ping_postgres() -> str:
    async with engine.connect() as conn:
        result = await conn.execute(text("SHOW server_version"))
        return str(result.scalar_one())
