"""VM-only Needle inference. Grounding and execution remain on the client."""
from .observability import timed

SYSTEM = 'Map explicit home actions and temperature queries to tools. Incomplete, negated or unsupported requests return no calls.'
GENERIC_SYSTEM = 'Map the user request to the provided tools. Incomplete, negated or unsupported requests return no calls.'


class NeedleEngine:
    def __init__(self):
        import os
        os.environ['NEEDLE_TELEMETRY'] = '0'
        from needle import Needle
        self.factory = Needle
        self.model = None

    def infer(self, text, schemas, system=GENERIC_SYSTEM):
        if self.model is not None:
            self.model.close()
        with timed('needle.load', tools=len(schemas)):
            self.model = self.factory(generation=3, tools=schemas, auto_date=False, system=system)
        with timed('needle.complete'):
            return self.model.complete(text, max_new_tokens=256)
