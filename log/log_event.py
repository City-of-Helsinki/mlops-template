from logging import LogRecord
from sqlite3 import Timestamp
from typing import Optional

from sqlmodel import Field, SQLModel


# Simple class to input log events to database
class LogEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: Timestamp
    severity: str
    message: str

    def __init__(self, record: LogRecord):
        self.timestamp = Timestamp.fromtimestamp(record.created)
        self.severity = record.levelname
        self.message = record.msg

