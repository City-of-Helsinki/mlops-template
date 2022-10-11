from sqlmodel import create_engine, Session, SQLModel

from log.log_event import LogEvent
from log.logging_handler import LoggingHandler


class SQLiteLoggingHandler(LoggingHandler):
    def __init__(self):
        self.engine = create_engine("sqlite:///logger.db")
        SQLModel.metadata.create_all(self.engine)

    def log(self, event: LogEvent):
        with Session(self.engine) as session:
            session.add(event)
            session.commit()