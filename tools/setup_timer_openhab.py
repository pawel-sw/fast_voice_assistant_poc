"""Create Jarvis's countdown item and add its card to the existing overview."""
import json
from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jarvis.control import OpenHAB
from jarvis.timers import TIMER_ITEM


def main():
    config = json.loads((ROOT/'config.json').read_text())
    api = OpenHAB(config['openhab_url'])
    url = api.url + '/rest/items/' + TIMER_ITEM
    existing = api.session.get(url, timeout=10)
    if existing.status_code == 404:
        response = api.session.put(url, json={'type': 'String', 'name': TIMER_ITEM,
            'label': 'Timer remaining', 'category': 'time', 'tags': ['Status'], 'groupNames': []}, timeout=10)
        response.raise_for_status()
        api.timer_state('00:00:00')
        print('Created countdown item:', TIMER_ITEM)
    else:
        existing.raise_for_status()
        if existing.json()['type'] != 'String':
            raise RuntimeError('Existing timer item has a different type; not overwritten')
    response = api.session.put(url+'/metadata/stateDescription',
        json={'value': '', 'config': {'readOnly': True}}, timeout=10)
    response.raise_for_status()
    page_url = api.url + '/rest/ui/components/ui:page/overview'
    response = api.session.get(page_url, timeout=10)
    response.raise_for_status()
    page = response.json()
    if TIMER_ITEM not in json.dumps(page):
        backup = ROOT/'data'/('overview-before-timer-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.json')
        backup.write_text(json.dumps(page, indent=2), encoding='utf-8')
        card = {'component': 'oh-label-card', 'config': {'item': TIMER_ITEM,
            'title': 'Voice timer', 'icon': 'oh:time'}}
        block = {'component': 'oh-block', 'config': {}, 'slots': {'default': [
            {'component': 'oh-grid-row', 'slots': {'default': [
                {'component': 'oh-grid-col', 'slots': {'default': [card]}}]}}]}}
        page.setdefault('slots', {}).setdefault('default', []).append(block)
        page.pop('editable', None)
        page.pop('timestamp', None)
        response = api.session.put(page_url, json=page, timeout=10)
        response.raise_for_status()
        print('Added overview card; backup:', backup)
    check = api.session.get(page_url, timeout=10)
    check.raise_for_status()
    assert TIMER_ITEM in json.dumps(check.json())
    print('Verified countdown item and overview card')


if __name__ == '__main__':
    main()
