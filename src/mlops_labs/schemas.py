from typing import Literal

from pydantic import BaseModel


class HealthzResponse(BaseModel):
    status: Literal["ok"]


class VersionResponse(BaseModel):
    version: str
