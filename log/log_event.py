from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class LogEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime
    severity: str
    message: str

