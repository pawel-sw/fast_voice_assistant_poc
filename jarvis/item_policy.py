"""Configured room vocabulary and exclusions shared by all client planners."""
import fnmatch
import re


def plain(value):
    value = re.sub(r'([a-z])([A-Z])', r'\1 \2', value)
    return ' '.join(re.findall(r'[a-z0-9]+', value.lower()))


def phrase(value):
    return re.compile(r'(?<![a-z0-9])' + r'[\s_]+'.join(map(re.escape, plain(value).split())) + r'(?![a-z0-9])', re.I)


class ItemPolicy:
    def __init__(self, items, config):
        self.available = {item['name']: item for item in items}
        self.mappings = config.get('room_mappings', {})
        self.defaults = config.get('room_defaults', {})
        self.item_defaults = config.get('item_defaults', {})
        self.room_groups = config.get('room_groups', {})
        self.ignore_rooms = config.get('ignore_rooms', [])
        self.ignore_groups = set(config.get('ignore_groups', []))
        self.patterns = config.get('ignore_item_patterns', [])
        self.excluded_names = set(config.get('exclude_items', []))
        self.qualified = {}
        # Learn explicit qualifiers from item/group names as well as configuration.
        names = [plain(value) for item in items for value in (item['name'], item.get('label') or '')]
        names += [plain(value) for value in self.mappings.values()]
        names += [plain(value) for value in self.defaults.values()]
        for alias in self.defaults:
            found = set()
            for name in names:
                found.update(match[0] for match in re.finditer(r'\b\w+\s+' + re.escape(plain(alias)) + r'\b', name))
            # Never redirect an explicitly qualified room that isn't in this catalog.
            found.update(prefix+' '+plain(alias) for prefix in
                         ('guest', 'master', 'basement', 'upstairs', 'downstairs', 'spare', 'kids', 'children', 'primary', 'main'))
            self.qualified[alias] = [phrase(value) for value in found]

    def ancestors(self, item):
        visited = set()
        pending = list(item.get('groupNames', []))
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            group = self.available.get(name)
            if group:
                pending.extend(group.get('groupNames', []))
        return visited

    def ignored_request(self, text):
        return any(phrase(room).search(text) for room in self.ignore_rooms)

    def excluded(self, item):
        if item['name'] in self.excluded_names:
            return True
        values = (plain(item['name']), plain(item.get('label') or ''))
        if any(fnmatch.fnmatchcase(value.lower(), pattern.lower())
               for value in (item['name'], item.get('label') or '') for pattern in self.patterns):
            return True
        ancestors = self.ancestors(item)
        if ancestors & self.ignore_groups:
            return True
        groups = [self.available[name] for name in ancestors if name in self.available]
        values += tuple(plain(value) for group in groups for value in (group['name'], group.get('label') or ''))
        return any(phrase(room).search(value) for room in self.ignore_rooms for value in values)

    def normalize(self, text):
        # Resolve aliases once, then expand defaults only outside explicit room names.
        if self.mappings:
            aliases = sorted(self.mappings, key=len, reverse=True)
            pattern = re.compile('|'.join('('+phrase(alias).pattern+')' for alias in aliases), re.I)
            text = pattern.sub(lambda match: self.mappings[aliases[match.lastindex-1]], text)
        for alias, canonical in self.defaults.items():
            protected = [(m.start(), m.end()) for pattern in self.qualified[alias] for m in pattern.finditer(text)]
            text = phrase(alias).sub(lambda match: match[0] if any(start <= match.start() < end for start, end in protected) else canonical, text)
        # A bare device noun may have a default item. Any extra target qualifier
        # (e.g. "bathroom fan") disables this default, even if that room lacks it.
        command_words = set('please turn switch set make put get the a an my on off to at percent percentage speed level brightness up down increase decrease dim low medium high maximum minimum full power is what s it can you could would now'.split())
        for alias, item_name in self.item_defaults.items():
            item = self.available.get(item_name)
            if not item or self.excluded(item) or not phrase(alias).search(text):
                continue
            extra = set(re.findall(r'[a-z]+', text.lower())) - set(plain(alias).split()) - command_words
            if not extra:
                target = item.get('label') or item_name.replace('_', ' ')
                text = phrase(alias).sub(lambda _: target, text)
        return text

    def aliases(self, item):
        labels = [item.get('label') or item['name'].replace('_', ' '), item['name'].replace('_', ' ')]
        ancestors = self.ancestors(item)
        for room, groups in self.room_groups.items():
            if ancestors.intersection(groups):
                labels.extend(room+' '+label for label in list(labels) if not phrase(room).search(label))
        generated = list(labels[2:])
        for alias, canonical in self.mappings.items():
            for label in labels:
                if phrase(canonical).search(label):
                    generated.append(phrase(canonical).sub(lambda _: alias, label))
        return list(dict.fromkeys(generated))

    def label_matches_room(self, label, text):
        for alias, patterns in self.qualified.items():
            if phrase(alias).search(label):
                explicit = [pattern for pattern in patterns if pattern.search(text)]
                if explicit and not any(pattern.search(label) for pattern in explicit):
                    return False
        return True
