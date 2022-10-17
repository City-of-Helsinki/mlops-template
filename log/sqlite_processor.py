from sqlmodel import create_engine, Session, SQLModel

from log.log_event import LogEvent


class SQLiteProcessor:
    def __init__(self):
        self.engine = create_engine("sqlite:///sqlite_structlog_db.db")
        SQLModel.metadata.create_all(self.engine)

    def __call__(self, logger, name, event_dict):
        event: LogEvent = LogEvent(event_dict)
        with Session(self.engine) as session:
            session.add(event)
            session.commit()

        return event_dict