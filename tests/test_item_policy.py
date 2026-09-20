from unittest.mock import Mock
import pytest
from jarvis.control import Catalog
from jarvis.remote_rpc import RemotePlanner


CONFIG = {'room_mappings': {'office': 'Guest Bedroom'},
          'room_defaults': {'bedroom': 'Master Bedroom', 'bathroom': 'Master Bathroom'},
          'room_groups': {'Guest Bedroom': ['GuestRoom']},
          'ignore_rooms': ['Piano Room'], 'min_confidence': .8}


def catalog(extra=None, **config):
    items = [
        {'name': 'Guest_Bedroom_Light', 'label': 'Guest Bedroom Light', 'type': 'Dimmer'},
        {'name': 'Master_Bedroom_Light', 'label': 'Master Bedroom Light', 'type': 'Dimmer'},
        {'name': 'Master_Bathroom_Light', 'label': 'Master Bathroom Light', 'type': 'Dimmer'},
        {'name': 'Basement_Bathroom_Light', 'label': 'Basement Bathroom Light', 'type': 'Dimmer'},
        {'name': 'Piano_Room_Lights', 'label': 'Piano Room Lights', 'type': 'Dimmer'},
        {'name': 'PianoRoom', 'label': 'Piano Room', 'type': 'Group'},
        {'name': 'Equipment', 'type': 'Group', 'groupNames': ['PianoRoom']},
        {'name': 'Hidden_Fan', 'label': 'Fan', 'type': 'Dimmer', 'groupNames': ['Equipment']},
        {'name': 'GuestRoom', 'type': 'Group', 'label': 'Guest Room'},
        {'name': 'Desk_Lamp', 'label': 'Desk Lamp', 'type': 'Dimmer', 'groupNames': ['GuestRoom']},
    ] + (extra or [])
    return Catalog(items, {**CONFIG, **config})


@pytest.mark.parametrize('text,expected', [
    ('turn on office lights', 'switch_guest_bedroom_light'),
    ('set office light to 25 percent', 'level_guest_bedroom_light'),
    ('turn off bedroom light', 'switch_master_bedroom_light'),
    ('turn off bathroom light', 'switch_master_bathroom_light'),
    ('turn on guest bedroom light', 'switch_guest_bedroom_light'),
    ('turn on master bedroom light', 'switch_master_bedroom_light'),
    ('turn on basement bathroom light', 'switch_basement_bathroom_light'),
    ('turn on office desk lamp', 'switch_desk_lamp')])
def test_room_resolution(text, expected):
    assert [s['name'] for s in catalog().candidates(text)] == [expected]


def test_normalization_is_idempotent_and_preserves_explicit_rooms():
    policy = catalog().policy
    text = 'turn on office light and bedroom light and guest bedroom light'
    normalized = policy.normalize(text)
    assert normalized == 'turn on Guest Bedroom light and Master Bedroom light and guest bedroom light'
    assert policy.normalize(normalized) == normalized
    assert policy.normalize('turn on upstairs bedroom light') == 'turn on upstairs bedroom light'


def test_exclusion_applies_to_ancestry_and_overrides_include_and_aliases():
    c = catalog(include_items=['Piano_Room_Lights', 'Hidden_Fan'], aliases={'Piano_Room_Lights': ['office light']})
    assert 'Piano_Room_Lights' not in c.items
    assert 'Hidden_Fan' not in c.items
    assert c.candidates('turn on piano room lights') == []
    with pytest.raises(ValueError):
        c.validate([{'name': 'switch_piano_room_lights', 'arguments': {'value': 'ON'}}])


def test_ignored_room_never_sent_to_models():
    p = RemotePlanner(catalog(), CONFIG)
    p._needle = Mock()
    p.groq = Mock()
    assert p.plan('turn on Piano Room lights', 'x') == []
    p._needle.assert_not_called()
    p.groq.repair.assert_not_called()


def test_item_patterns_and_group_ignores():
    c = catalog(ignore_item_patterns=['Master_Bathroom_*'], ignore_groups=['GuestRoom'], exclude_items=['Master_Bedroom_Light'])
    assert 'Master_Bathroom_Light' not in c.items
    assert 'Desk_Lamp' not in c.items
    assert 'Master_Bedroom_Light' not in c.items


def test_default_does_not_fall_back_to_other_unqualified_room():
    c = catalog(extra=[{'name': 'Bathroom_Fan', 'label': 'Bathroom Fan', 'type': 'Dimmer'}])
    assert c.candidates('turn on bathroom fan') == []


def test_planner_receives_canonical_room():
    p = RemotePlanner(catalog(), CONFIG)
    p._needle = Mock(return_value=['accepted'])
    assert p.plan('turn off office light', 'session') == ['accepted']
    p._needle.assert_called_once_with('turn off Guest Bedroom light', 'session')


def test_explicit_item_alias_can_use_a_mapped_room():
    c = catalog(aliases={'Guest_Bedroom_Light': ['office overhead light']})
    assert [s['name'] for s in c.candidates('turn on office overhead light')] == ['switch_guest_bedroom_light']


def test_bare_fan_default_preserves_explicit_room_commands():
    c = catalog(extra=[
        {'name': 'Balcony_Fan', 'label': 'Balcony Fan', 'type': 'Dimmer'},
        {'name': 'Master_Bathroom_Fan', 'label': 'Master Bathroom Fan', 'type': 'Dimmer'}],
        item_defaults={'fan': 'Balcony_Fan'})
    assert [s['name'] for s in c.candidates('turn the fan off')] == ['switch_balcony_fan']
    assert [s['name'] for s in c.candidates('set fan to 67 percent')] == ['level_balcony_fan']
    assert [s['name'] for s in c.candidates('turn bathroom fan off')] == ['switch_master_bathroom_fan']
    assert c.candidates('turn bedroom fan off') == []
    assert c.candidates('turn kitchen fan off') == []
    assert c.candidates('turn all fans off') == []
    assert c.policy.normalize('turn off Balcony Fan') == 'turn off Balcony Fan'


def test_excluded_default_item_does_not_become_accessible():
    c = catalog(extra=[{'name': 'Balcony_Fan', 'label': 'Balcony Fan', 'type': 'Dimmer'}],
        item_defaults={'fan': 'Balcony_Fan'}, exclude_items=['Balcony_Fan'])
    assert c.candidates('turn fan on') == []
