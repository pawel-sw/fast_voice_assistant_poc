"""Client-owned, persistent countdown with asynchronous display and chime."""
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import queue
import re
import threading
import time

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
TIMER_ITEM = 'Jarvis_Timer_Remaining'


@dataclass(frozen=True)
class TimerAction:
    operation: str
    seconds: int = 0


ONES = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))


def amount(text):
    text = text.strip()
    if text in ('a', 'an'):
        return 1
    if text in ('half', 'a half', 'half a', 'half an'):
        return .5
    if text in ('quarter', 'a quarter', 'a quarter of an', 'a quarter of a'):
        return .25
    if text.endswith(' and a half'):
        return amount(text[:-11]) + .5
    if re.fullmatch(r'\d+(?:\.\d+)?', text):
        return float(text)
    total = 0
    previous = None
    for word in text.split():
        if word == 'and' and total:
            continue
        if word in ONES:
            if previous == 'ones':
                raise ValueError('Ambiguous number')
            total += ONES[word]
            previous = 'ones'
        elif word in TENS:
            if previous in ('ones', 'tens'):
                raise ValueError('Ambiguous number')
            total += TENS[word]
            previous = 'tens'
        elif word == 'hundred' and previous == 'ones' and 1 <= total <= 9:
            total *= 100
            previous = 'hundred'
        else:
            raise ValueError('Unknown duration')
    if previous is None:
        raise ValueError('Missing duration')
    return total


def duration_seconds(text):
    total = 0
    position = 0
    units = {'hour': 3600, 'hr': 3600, 'minute': 60, 'min': 60, 'second': 1, 'sec': 1}
    for match in re.finditer(r'(.+?)\s+(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b', text):
        if match.start() != position:
            raise ValueError('Invalid duration')
        part = match[1].strip().removeprefix('and ')
        total += amount(part) * units[match[2].rstrip('s')]
        position = match.end()
    if text[position:].strip() or not 1 <= total <= 86400 or total != int(total):
        raise ValueError('Duration must be 1 second to 24 hours')
    return int(total)


def parse_timer(text):
    text = re.sub(r'[,.!?]+$', '', text.lower().strip()).replace(',', ' ')
    text = re.sub(r'(?<=[a-z0-9])-(?=[a-z])', ' ', text)
    text = ' '.join(text.split())
    text = re.sub(r'^please\s+|\s+please$', '', text)
    if re.fullmatch(r'(?:cancel|stop|delete|clear) (?:the |my |a )?timer', text):
        return TimerAction('cancel')
    if text in ('stop', 'stop ringing', 'silence', 'silence the timer', 'dismiss timer', 'dismiss the timer'):
        return TimerAction('stop')
    if re.fullmatch(r'(?:how much time (?:is )?(?:left|remaining)(?: on (?:the |my )?timer)?|(?:what is |what\'s )?(?:the |my )?timer (?:status|remaining))', text):
        return TimerAction('status')
    if re.search(r"\b(?:don't|do not|never|not)\b", text):
        return None
    match = re.fullmatch(r'(?:(?:set|start|create) (?:a |an |the |my )?)?timer(?: (?:for|to))?(?: (.*))?', text)
    if not match:
        match = re.fullmatch(r'(?:(?:set|start|create) (?:a |an |the |my )?)?(.+?) timer', text)
    if not match:
        return None
    try:
        return TimerAction('start', duration_seconds(match[1] or ''))
    except ValueError:
        return TimerAction('invalid')


def human_duration(seconds):
    parts = []
    for unit, size in (('hour', 3600), ('minute', 60), ('second', 1)):
        value, seconds = divmod(int(seconds), size)
        if value:
            parts.append(f'{value} {unit}' + ('s' if value != 1 else ''))
    return ' and '.join(parts) or '0 seconds'


class TimerService:
    def __init__(self, publish, chime, path=None, clock=time.monotonic, wall=time.time, start_worker=True):
        self.publish, self.chime = publish, chime
        self.path = Path(path) if path else ROOT / 'data/timer.json'
        self.clock, self.wall = clock, wall
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.changed = threading.Event()
        self.ring_cancel = threading.Event()
        self.ring_thread = None
        self.events = queue.SimpleQueue()
        self.deadline = self.wall_deadline = None
        self.ringing = False
        self.latest_display = None
        self.worker = self.publisher = None
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                end = data.get('deadline')
                if isinstance(end, (int, float)) and math.isfinite(end):
                    self.wall_deadline = end
                    self.deadline = self.clock() + max(0, end-self.wall())
                self.ringing = data.get('ringing') is True
            except (ValueError, OSError):
                log.exception('Could not restore timer')
        if start_worker:
            self.publisher = threading.Thread(target=self._publish_loop, name='timer-openhab', daemon=True)
            self.worker = threading.Thread(target=self._run, name='timer-clock', daemon=True)
            self.publisher.start()
            self.worker.start()

    @property
    def is_ringing(self):
        with self.lock:
            return self.ringing

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps({'deadline': self.wall_deadline, 'ringing': self.ringing}))
        temp.replace(self.path)

    def remaining(self):
        with self.lock:
            return max(0, math.ceil(self.deadline-self.clock())) if self.deadline is not None else 0

    def display(self):
        with self.lock:
            if self.ringing:
                return '00:00:00 - Ringing'
            seconds = self.remaining()
            return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'

    def apply(self, action, dry_run=False):
        if action.operation == 'invalid':
            return 'Please specify a timer from one second to twenty four hours.'
        with self.lock:
            if action.operation == 'status':
                return 'Your timer is ringing.' if self.ringing else (f'{human_duration(self.remaining())} remaining.' if self.deadline is not None else 'There is no active timer.')
            if dry_run:
                return 'Dry run. ' + (f'Timer for {human_duration(action.seconds)}.' if action.operation == 'start' else 'Timer stopped.')
            if action.operation == 'start':
                if type(action.seconds) is not int or not 1 <= action.seconds <= 86400:
                    raise ValueError('Invalid timer duration')
                replaced = self.deadline is not None or self.ringing
                self.ring_cancel.set()
                self.ringing = False
                self.deadline, self.wall_deadline = self.clock()+action.seconds, self.wall()+action.seconds
                message = f'Timer {"replaced and " if replaced else ""}set for {human_duration(action.seconds)}.'
            elif action.operation == 'cancel':
                existed = self.deadline is not None or self.ringing
                self.ring_cancel.set()
                self.ringing = False
                self.deadline = self.wall_deadline = None
                message = 'Timer cancelled.' if existed else 'There is no active timer.'
            elif action.operation == 'stop':
                if not self.ringing:
                    return 'No timer is ringing.'
                self.ring_cancel.set()
                self.ringing = False
                self.deadline = self.wall_deadline = None
                message = 'Timer stopped.'
            else:
                raise ValueError('Unknown timer operation')
            self._save()
            self.latest_display = self.display()
            self.changed.set()
            log.info('TIMER: %s', message)
            return message

    def tick(self):
        with self.lock:
            if self.deadline is not None and self.clock() >= self.deadline:
                self.deadline = self.wall_deadline = None
                self.ringing = True
                self._save()
                self.events.put('expired; ringing')
                log.info('TIMER expired; ringing')
            display = self.display()
            if self.latest_display != display:
                self.latest_display = display
                self.changed.set()
            if self.ringing and (self.ring_thread is None or not self.ring_thread.is_alive()):
                self.ring_cancel = threading.Event()
                self.ring_thread = threading.Thread(target=self._ring, args=(self.ring_cancel,), daemon=True)
                self.ring_thread.start()

    def _ring(self, cancelled):
        while not cancelled.is_set() and not self.stop_event.is_set():
            try:
                self.chime(cancelled)
            except Exception:
                log.exception('Timer chime playback failed')
            # Quiet interval allows "stop" to be heard without acoustic feedback.
            if cancelled.wait(7):
                return

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception:
                log.exception('Timer update failed')
            self.stop_event.wait(.1)

    def _publish_loop(self):
        last_warning = 0
        while not self.stop_event.is_set():
            if not self.changed.wait(.5):
                continue
            if self.stop_event.is_set():
                return
            self.changed.clear()
            with self.lock:
                value = self.latest_display
            try:
                self.publish(value)
            except Exception:
                if self.clock()-last_warning > 30:
                    log.warning('Timer display unavailable; countdown continues locally')
                    last_warning = self.clock()
                self.stop_event.wait(2)
                self.changed.set()

    def close(self):
        self.stop_event.set()
        self.ring_cancel.set()
        self.changed.set()
        for thread in (self.worker, self.publisher, self.ring_thread):
            if thread:
                thread.join(timeout=4)
