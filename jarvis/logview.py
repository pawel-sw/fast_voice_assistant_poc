"""Follow the live log, including after rotation or listener restart."""
import argparse
from collections import deque
from pathlib import Path
import time
import sys


def follow_plain(path):
    previous = ''
    try:
        while True:
            try:
                # Read only the tail; current line can be rewritten by ASR.
                with path.open('rb') as stream:
                    stream.seek(0, 2)
                    start = max(0, stream.tell() - 65536)
                    stream.seek(start)
                    data = stream.read()
                if start and b'\n' in data:
                    data = data.split(b'\n', 1)[1]
                text = '\n'.join(data.decode('utf-8', errors='replace').split('\n')[-25:])
                if text != previous:
                    if text.startswith(previous):
                        sys.stdout.write(text[len(previous):])
                    else:
                        # Redraw corrections instead of duplicating text or
                        # inserting extra session lines. File stays plain text.
                        sys.stdout.write('\x1b[2J\x1b[H' + text)
                    sys.stdout.flush()
                    previous = text
            except FileNotFoundError:
                pass
            time.sleep(0.08)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--timings', action='store_true')
    group.add_argument('--detailed', action='store_true')
    args = parser.parse_args()
    if not args.timings and not args.detailed:
        follow_plain(Path(__file__).resolve().parents[1] / 'logs/transcript.txt')
        return
    path = Path(__file__).resolve().parents[1] / 'logs' / ('timings.log' if args.timings else 'jarvis.log')
    print(f'Live view: {path}  (Ctrl+C closes this view; assistant keeps running)', flush=True)
    position, identity = 0, None
    initial = True
    try:
        while True:
            try:
                with path.open('rb') as stream:
                    stat = path.stat()
                    current = (stat.st_dev, stat.st_ino)
                    if initial:
                        for line in deque(stream, maxlen=25):
                            print(line.decode('utf-8', errors='replace'), end='', flush=True)
                        position, initial = stream.tell(), False
                    else:
                        if current != identity or stat.st_size < position:
                            position = 0
                        stream.seek(position)
                        for line in stream:
                            if not line.endswith(b'\n'):
                                break
                            print(line.decode('utf-8', errors='replace'), end='', flush=True)
                            position = stream.tell()
                    identity = current
            except FileNotFoundError:
                pass
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
