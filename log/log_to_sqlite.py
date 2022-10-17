import sqlite_utils


class LogToSqlite:
    def __init__(self, db_path, table="logs"):
        self.db = sqlite_utils.Database(db_path)
        self.table = table

    def __call__(self, logger, name, event_dict):
        self.db[self.table].insert({**event_dict, **{
            "name": name,
        }}, alter=True)
        return event_dict

