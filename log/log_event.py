from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


# Simple class to input log events to database
class LogEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime
    severity: str
    message: str
    # request: object
    # response: object

    def __init__(self, record: dict):
        self.timestamp = datetime.now()
        self.severity = 'INFO'
        # TODO: msg = dict
        self.message = str(record)

