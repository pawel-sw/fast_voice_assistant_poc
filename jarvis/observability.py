"""Wall-clock call timings and incremental, revision-aware transcript logging."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import json
from pathlib import Path
import time

log = logging.getLogger(__name__)
timing_log = logging.getLogger('jarvis.timing')
plain_transcript = None
candidate_segments = None
feedback_sink = ContextVar('feedback_sink', default=None)


def planning_feedback(stage, value):
    sink = feedback_sink.get()
    if sink is not None:
        sink(stage, value)


class PlainTranscript:
    """One mutable UTF-8 line per acoustic wake; no logging prefixes or escapes."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.offset = None
        self.segments = {}
        self.commands = []
        self.notes = {}

    def wake(self):
        with self.path.open('ab') as stream:
            if stream.tell():
                stream.write(b'\n')
            self.offset = stream.tell()
        self.segments = {}
        self.commands = []
        self.notes = {}

    def note(self, stage, value, key=None):
        if self.offset is None:
            return
        self.notes[key if key is not None else stage] = (stage, ' '.join(value.split()))
        self._write()

    def update(self, utterance, text):
        if self.offset is None:
            return
        self.segments[utterance] = ' '.join(text.split())
        self._write()

    def command(self, item, command, **status):
        if self.offset is None:
            return
        self.commands.append({'item': item, 'command': command, **status})
        self._write()

    def result(self, item, location, **values):
        if self.offset is None:
            return
        self.commands.append({'item': item, 'query': location + '_temperature', **values})
        self._write()

    def _write(self):
        line = ' '.join(text for text in self.segments.values() if text)
        for stage, value in self.notes.values():
            line += (' ' if line else '') + f'[{stage}: {value}]'
        if self.commands:
            line += (' ' if line else '') + json.dumps(self.commands, separators=(',', ':'), ensure_ascii=False)
        with self.path.open('r+b') as stream:
            stream.seek(self.offset)
            stream.write(line.encode('utf-8'))
            stream.truncate()
            stream.flush()


def configure_plain_transcript(path):
    global plain_transcript
    plain_transcript = PlainTranscript(path)


def plain_wake():
    if plain_transcript is not None:
        plain_transcript.wake()


def plain_candidate():
    global candidate_segments
    candidate_segments = {}


def plain_confirm_candidate():
    global candidate_segments
    plain_wake()
    if plain_transcript is not None:
        for utterance, text in (candidate_segments or {}).items():
            plain_transcript.update(utterance, text)
    candidate_segments = None


def plain_cancel_candidate():
    global candidate_segments
    candidate_segments = None


def plain_command(item, command, **status):
    if plain_transcript is not None:
        plain_transcript.command(item, command, **status)


def plain_result(item, location, **values):
    if plain_transcript is not None:
        plain_transcript.result(item, location, **values)


def plain_note(stage, value, key=None):
    if plain_transcript is not None:
        plain_transcript.note(stage, value, key)


@contextmanager
def timed(call, *, detail=False, **fields):
    start = time.perf_counter()
    outcome = 'ok'
    try:
        yield
    except BaseException as exc:
        outcome = type(exc).__name__
        raise
    finally:
        ms = (time.perf_counter() - start) * 1000
        context = ' '.join(f'{key}={value}' for key, value in fields.items())
        message = f'TIMING {call} {ms:.2f} ms outcome={outcome} {context}'.rstrip()
        timing_log.info(message)
        if not detail:
            log.info(message)


class Transcript:
    def __init__(self, utterance):
        self.utterance = utterance
        self.words = []
        self.started = time.perf_counter()

    def update(self, text, final=False, provisional=False):
        words = text.split()
        # Prefix rollback temporarily re-emits already displayed words while
        # decoding. Wait until it catches up instead of flickering backwards.
        if provisional and len(words) < len(self.words) and self.words[:len(words)] == words:
            return
        if candidate_segments is not None:
            candidate_segments[self.utterance] = text
        elif plain_transcript is not None:
            plain_transcript.update(self.utterance, text)
        # Partial text belongs only in transcript.txt. audio.py logs one final
        # "Heard" entry per utterance in the diagnostic log.
        self.words = words


def timestamp(event, session, *, at=None, **fields):
    """Client-clock event time; at is a captured monotonic timestamp."""
    from datetime import datetime, timezone
    mono = time.monotonic() if at is None else at
    wall = time.time() - (time.monotonic() - mono)
    utc = datetime.fromtimestamp(wall, timezone.utc).isoformat(timespec='microseconds')
    context = ' '.join(f'{key}={value}' for key, value in fields.items())
    timing_log.info('EVENT %s timestamp=%s monotonic=%.6f session=%s %s',
                    event, utc, mono, session, context)


@contextmanager
def timestamp_span(name, session, **fields):
    timestamp(name + '_start', session, **fields)
    outcome = 'ok'
    try:
        yield
    except BaseException as exc:
        outcome = type(exc).__name__
        raise
    finally:
        timestamp(name + '_done', session, outcome=outcome, **fields)
