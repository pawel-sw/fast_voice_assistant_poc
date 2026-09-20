"""FIFO admission to shared model runtimes, with bounded waiting."""
from collections import deque
import threading
import time
from .observability import timed


class FairGate:
    def __init__(self, name, timeout=30):
        self.name, self.timeout = name, timeout
        self.condition = threading.Condition()
        self.queue = deque()
        self.busy = False

    def __enter__(self):
        ticket = object()
        deadline = time.monotonic() + self.timeout
        with timed('queue.wait', detail=True, model=self.name):
            with self.condition:
                self.queue.append(ticket)
                try:
                    while self.busy or self.queue[0] is not ticket:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError('Inference queue is busy; try again later')
                        self.condition.wait(remaining)
                    self.queue.popleft()
                    self.busy = True
                except BaseException:
                    self.queue.remove(ticket)
                    self.condition.notify_all()
                    raise
        return self

    def __exit__(self, *exc):
        with self.condition:
            self.busy = False
            self.condition.notify_all()
