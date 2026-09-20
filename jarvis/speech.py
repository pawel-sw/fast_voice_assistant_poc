"""Client-side reply wording; model-independent and based on API outcomes."""
import re
from .control import TemperatureQuery


def confirmation(action, catalog, result=None, dry_run=False):
    if isinstance(action, TemperatureQuery):
        if result == 'unavailable':
            return f'The temperature {action.location} is unavailable.'
        value = str(result).replace('°C', ' degrees Celsius').replace('°F', ' degrees Fahrenheit')
        return f'The temperature {action.location} is {" ".join(value.split())}.'
    item = catalog.items[action.item]
    label = item.get('label') or action.item.replace('_', ' ')
    label = re.sub(r'\s+', ' ', label).strip()
    command = action.command
    if command in ('ON', 'OFF'):
        phrase = f'{label} turned {command.lower()}.'
    elif command.isdigit():
        phrase = f'{label} set to {command} percent.'
    else:
        verb = {'UP': 'opening', 'DOWN': 'closing', 'STOP': 'stopped',
                'PLAY': 'playing', 'PAUSE': 'paused', 'NEXT': 'skipping to the next track',
                'PREVIOUS': 'going to the previous track'}.get(command, command.lower())
        phrase = f'{label} {verb}.'
    return ('Dry run. ' if dry_run else '') + phrase
