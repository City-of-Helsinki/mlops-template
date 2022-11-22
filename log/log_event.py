from logging import LogRecord
from sqlite3 import Timestamp
from typing import Optional

from sqlmodel import Field, SQLModel


# Simple class to input log events to database
class LogEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: Timestamp
    severity: Optional[str]
    message: Optional[str]
    request: Optional[str]
    response: Optional[str]
    type: Optional[str]

    def __init__(self, record: LogRecord):
        self.timestamp = Timestamp.fromtimestamp(record.created)
        self.severity = record.levelname
        if type(record.msg) is dict and record.msg:
            self.message = None
            if 'prediction' in record.msg:
                self.type = 'PREDICTION'
                self.request = record.msg['request_parameters'].json()
                self.response = record.msg['prediction']
        else:
            self.message = record.msg
            self.request = None
            self.response = None

