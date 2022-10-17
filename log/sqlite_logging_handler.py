import logging

from sqlmodel import create_engine, Session, SQLModel

from log.log_event import LogEvent

# Example usage with standard logging:
# import logging
# sqlite_logging_handler = SQLiteLoggingHandler()
# logging.getLogger().addHandler(sqlite_logging_handler)
# logging.getLogger().setLevel(logging.INFO)

class SQLiteLoggingHandler(logging.Handler):

    def __init__(self):
        logging.Handler.__init__(self)
        self.engine = create_engine("sqlite:///sqlite_logger_db.db")
        SQLModel.metadata.create_all(self.engine)

    def emit(self, record: logging.LogRecord):
        event: LogEvent = LogEvent(record)
        with Session(self.engine) as session:
            session.add(event)
            session.commit()