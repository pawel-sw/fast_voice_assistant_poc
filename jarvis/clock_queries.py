"""Deterministic time/date replies using the microphone client's local clock."""
from dataclasses import dataclass
from datetime import datetime
import re


@dataclass(frozen=True)
class ClockQuery:
    kind: str


def parse_clock(text):
    text = text.lower().replace('\u2019', "'")
    text = re.sub(r"\bwhat's\b", 'what is', text)
    text = ' '.join(re.findall(r'[a-z]+', text))
    text = re.sub(r'^please | please$', '', text)
    time_phrases = {
        'what time is it', 'what time it is', 'what is the time',
        'tell me the time', 'tell me what time it is',
        'what is the current time', 'what time is it now',
    }
    date_phrases = {
        'what day is it', 'what day it is', 'what day is today',
        'what day is it today', 'what is the date', 'what is the date today',
        'what is today s date', 'tell me the date', 'what is today',
    }
    if text in time_phrases:
        return ClockQuery('time')
    if text in date_phrases:
        return ClockQuery('date')
    return None


def clock_answer(query, now=None):
    now = now if now is not None else datetime.now().astimezone()
    if query.kind == 'time':
        hour = now.hour % 12 or 12
        return f'It is {hour}:{now.minute:02d} {"AM" if now.hour < 12 else "PM"}.'
    if query.kind == 'date':
        weekdays = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
        months = ('January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December')
        return f'{weekdays[now.weekday()]} {months[now.month - 1]} {now.day}.'
    raise ValueError('Unsupported clock query')
