from pydantic import BaseModel
from typing import List
from datetime import datetime

class Event(BaseModel):
    title: str
    time: str
    tag: str
    venue: str
    date: str = None
    description: str = None

class ScheduleQuery(BaseModel):
    query: str
    user_id: str

class ScheduleResponse(BaseModel):
    events: List[Event]