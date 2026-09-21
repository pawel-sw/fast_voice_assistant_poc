import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading
from contextlib import ExitStack

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def main():
    with ExitStack() as stack:
        return run(stack)


def run(stack):
    parser = argparse.ArgumentParser(description='PoC Home Automation Voice Assistant')
    parser.add_argument('--devices', action='store_true')
    parser.add_argument('--text', help='Test Needle planning without microphone or wake phrase')
    parser.add_argument('--execute', action='store_true', help='Execute --text actions (text tests default to dry-run)')
    parser.add_argument('--dry-run', action='store_true', help='Listen but log commands without sending them')
    parser.add_argument('--duration', type=float, help='Stop listening after this many seconds')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if not args.text and not args.devices:
        from .instance import listener_lock
        try:
            stack.enter_context(listener_lock(ROOT / 'data/listener.lock'))
        except RuntimeError as exc:
            parser.exit(1, str(exc) + '\n')
    load_dotenv(ROOT / '.env')
    (ROOT / 'logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), RotatingFileHandler(ROOT / 'logs/jarvis.log', maxBytes=2_000_000, backupCount=3, encoding='utf-8')])
    # Every inference/request is timed. High-frequency idle calls stay in their
    # own rotating log so the live transcript remains readable.
    timing = logging.getLogger('jarvis.timing')
    timing.propagate = False
    timing.setLevel(logging.INFO)
    timing_handler = RotatingFileHandler(ROOT / 'logs/timings.log', maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    timing_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    timing.addHandler(timing_handler)
    if args.devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    config = json.loads((ROOT / 'config.json').read_text())
    if config.get('remote_url'):
        from .observability import configure_plain_transcript
        from .remote_client import listen, text_request
        if args.text:
            return text_request(config, args.text, execute=args.execute and not args.dry_run)
        configure_plain_transcript(ROOT / 'logs/transcript.txt')
        return listen(config, args.duration, args.dry_run)
    from .integrations import create_integration
    from .control import Catalog, Conversation, Planner, execute_action
    from .observability import configure_plain_transcript
    if not args.text:
        configure_plain_transcript(ROOT / 'logs/transcript.txt')
    api = create_integration(config)
    items = api.items()
    catalog = Catalog(items, config)
    logging.info('Discovered %d items; enabled %d controllable devices', len(items), len(catalog.items))
    planner = Planner(catalog, config['min_confidence'])

    def execute(action):
        return execute_action(api, action, dry_run=args.dry_run or bool(args.text and not args.execute))

    if args.text:
        actions = planner.plan(args.text)
        for action in actions:
            execute(action)
        if not actions:
            logging.info('No valid grounded tool call; nothing sent')
        return
    from .asr import R2T2
    from .audio import listen
    asr = R2T2(config['asr_model'])
    conversation = Conversation(planner, execute, config['wake_phrase'], config['continuation_seconds'])
    stop = threading.Event()
    try:
        listen(asr, conversation, config, args.duration, stop)
    except KeyboardInterrupt:
        logging.info('Stopped')


if __name__ == '__main__':
    main()
