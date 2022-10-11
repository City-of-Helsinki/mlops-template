import datetime
import json
from abc import abstractmethod

from log.log_event import LogEvent


class LoggingHandler:
    LEVEL_DEBUG = "DEBUG"
    LEVEL_INFO = "INFO"
    LEVEL_WARN = "WARNING"
    LEVEL_ERROR = "ERROR"

    def __init__(self):
        pass

    def create_event(self, message: object, severity):
        e: LogEvent = LogEvent()
        e.timestamp = datetime.datetime.now()   #TODO: timezone settings from environment
        e.message = json.dumps(message)
        e.severity = severity
        return e

    @abstractmethod
    def info(self, message: object):
        return self.log(self.create_event(message, self.LEVEL_INFO))

    @abstractmethod
    def warning(self, message: object):
        return self.log(self.create_event(message, self.LEVEL_WARN))

    @abstractmethod
    def error(self, message: object):
        return self.log(self.create_event(message, self.LEVEL_ERROR))

    @abstractmethod
    def debug(self, message: object):
        return self.log(self.create_event(message, self.LEVEL_DEBUG))

    @abstractmethod
    def log(self, event: LogEvent):
        pass
