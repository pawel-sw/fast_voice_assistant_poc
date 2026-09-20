"""Client owns wake sessions, grounding, openHAB delivery and reply wording."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import queue
import re
import threading
import time
import uuid

from .control import TemperatureQuery
from . import observability as obs
from .speech import confirmation
from .timers import TimerAction, parse_timer
from .weather import WeatherQuery, WeatherService
from .clock_queries import ClockQuery, clock_answer

log = logging.getLogger(__name__)


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cancelled: threading.Event = field(default_factory=threading.Event)
    pending: str = ''
    verified: bool = True
    listening: bool = True
    revision: int = 0
    deadline: float = 0
    plan: object = None
    planned_revision: int = 0
    delivery: object = None
    speech: object = None
    feedback: object = field(default_factory=queue.SimpleQueue)
    timer_only: bool = False


class ClientBrain:
    def __init__(self, api, catalog, planner, speaker, config, dry_run=False, timer=None):
        self.api, self.catalog, self.planner, self.speaker = api, catalog, planner, speaker
        self.config, self.dry_run = config, dry_run
        self.timer = timer
        self.weather = WeatherService(config)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='jarvis-control')
        self.current = None
        self.wake = re.compile(r'\b(?:hey[\s,!.?]+jarvis|alexa)\b', re.I)

    def start(self, tentative=False, timer_only=False):
        self.cancel()
        state = self.current = Session(verified=not tentative, timer_only=timer_only, deadline=time.monotonic()+self.config['continuation_seconds'])
        if not timer_only:
            obs.plain_candidate() if tentative else obs.plain_wake()
        return state.id

    def cancel(self):
        if self.current:
            self.current.cancelled.set()
            self.current.listening = False
        obs.plain_cancel_candidate()

    @property
    def listening(self):
        return bool(self.current and self.current.listening and not self.current.cancelled.is_set())

    def transcript(self, event):
        state = self.current
        if not self.listening or event.get('session') != state.id:
            return
        text = event['text']
        match = self.wake.search(text)
        if not state.verified and match:
            state.verified = True
            obs.plain_confirm_candidate()
        if not state.timer_only:
            obs.Transcript(event['utterance']).update(text, final=event.get('final', False))
        if not event.get('final'):
            return
        log.info('Heard: %s', text)
        if not state.verified:
            self.cancel()
            return
        if not text.strip():
            return
        state.pending = text[match.end():].lstrip(' ,.!?') if match else (state.pending+' '+text).strip()
        state.deadline = time.monotonic()+self.config['continuation_seconds']
        state.revision += 1
        self._plan(state)

    def _plan(self, state):
        if state.pending and state.plan is None:
            state.planned_revision = state.revision
            state.plan = self.pool.submit(self._run_plan, state, state.revision, state.pending)

    def _run_plan(self, state, revision, text):
        token = obs.feedback_sink.set(lambda stage, value: state.feedback.put((revision, stage, value)))
        try:
            if state.timer_only:
                action = parse_timer(text)
                return [action] if action and action.operation in ('stop', 'cancel') else []
            return self.planner.plan(text, state.id)
        finally:
            obs.feedback_sink.reset(token)

    def _deliver(self, state, actions):
        outcomes = []
        for action in actions:
            if state.cancelled.is_set():
                break
            try:
                # No retries: a failed HTTP response may still mean delivery.
                if isinstance(action, ClockQuery):
                    with obs.timed('clock.answer'):
                        result = clock_answer(action)
                elif isinstance(action, WeatherQuery):
                    result = self.weather.answer(action)
                elif isinstance(action, TimerAction):
                    if self.timer is None:
                        raise RuntimeError('Timer service unavailable')
                    result = self.timer.apply(action, dry_run=self.dry_run)
                elif isinstance(action, TemperatureQuery):
                    with obs.timestamp_span('openhab', state.id, item=action.item, operation='query'):
                        result = self.api.temperature(action.item)
                else:
                    result = None
                    if not self.dry_run:
                        with obs.timestamp_span('openhab', state.id, item=action.item, operation='command'):
                            result = self.api.command(action.item, action.command)
                outcomes.append((action, result, None))
            except Exception:
                log.exception('Action request failed; not retried')
                outcomes.append((action, None, 'Request failed'))
                break
        return outcomes

    def _drain_feedback(self, state):
        # Worker threads enqueue feedback; only the client loop edits the log.
        while True:
            try:
                revision, stage, value = state.feedback.get_nowait()
            except queue.Empty:
                break
            if revision == state.revision:
                obs.plain_note(stage, value, key=(revision, stage))

    def tick(self, input_active=False):
        if self.timer:
            while True:
                try:
                    event = self.timer.events.get_nowait()
                except queue.Empty:
                    break
                obs.plain_note('timer', event)
        state = self.current
        if not state or state.cancelled.is_set():
            return
        try:
            self._drain_feedback(state)
            if state.plan is not None and state.plan.done():
                future, state.plan = state.plan, None
                actions = future.result()
                self._drain_feedback(state)
                if state.planned_revision != state.revision:
                    self._plan(state)
                elif actions:
                    state.listening = False
                    state.delivery = self.pool.submit(self._deliver, state, actions)
            if state.delivery is not None and state.delivery.done():
                future, state.delivery = state.delivery, None
                replies = []
                for action, result, error in future.result():
                    if isinstance(action, ClockQuery):
                        reply = 'I could not read the local clock.' if error else result
                        obs.plain_note(action.kind, reply)
                        replies.append(reply)
                        continue
                    if isinstance(action, WeatherQuery):
                        reply = 'I could not get the weather right now. Please try again.' if error else result
                        obs.plain_note('weather', reply)
                        replies.append(reply)
                        continue
                    elif isinstance(action, TimerAction):
                        obs.plain_note('timer', error or result)
                        replies.append('I could not update the timer.' if error else result)
                        continue
                    elif isinstance(action, TemperatureQuery):
                        obs.plain_result(action.item, action.location, **({'error': error} if error else {'result': result}))
                    else:
                        obs.plain_command(action.item, action.command, **({'error': error} if error else {'dry_run': True} if self.dry_run else {}))
                    replies.append('I could not confirm that request. Please check the device.' if error else confirmation(action, self.catalog, result, self.dry_run))
                reply = ' '.join(replies)
                log.info('Reply: %s', reply)
                if reply:
                    state.speech = self.pool.submit(self.speaker.speak, reply, state.id, state.cancelled)
            if state.speech is not None and state.speech.done():
                future, state.speech = state.speech, None
                future.result()
            if self.listening and not input_active and state.plan is None and time.monotonic() > state.deadline:
                self.cancel()
        except Exception:
            log.exception('Request failed; say Hey Jarvis or Alexa again')
            self.cancel()

    def close(self):
        self.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
