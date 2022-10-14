import logging
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


# Simple class to input log events to database
class LogEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime
    severity: str
    message: str

    def __init__(self, record:logging.LogRecord):
        self.timestamp = datetime.now()
        self.severity = record.levelname
        # TODO: msg = dict
        self.message = str(record.msg)

