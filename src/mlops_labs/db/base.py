from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mlops_labs.core.config import get_settings

engine: AsyncEngine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
