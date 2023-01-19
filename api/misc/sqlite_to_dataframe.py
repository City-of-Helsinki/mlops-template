# Example code: Read requests and responses from sqlite database to pandas dataframe
import json
from typing import List

import pandas as pd
from sqlalchemy import create_engine
from sqlmodel import SQLModel, Session, select

from api.log.log_event import LogEvent

engine = create_engine("sqlite:///../../logs.sqlite")
SQLModel.metadata.create_all(engine)
with Session(engine) as session:
    statement = select(LogEvent).where(LogEvent.request is not None)
    result: List[LogEvent] = session.exec(statement).all()
    y_str = [e.response for e in result]
    X_json = [json.loads(e.request) for e in result]
    df = pd.DataFrame.from_records(X_json)
    df["prediction"] = y_str
    print(df)
