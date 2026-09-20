"""Audit the Git index without displaying any matched private values.

This is a backstop, not a replacement for reviewing a diff before publishing.
Run after staging: python tools/check_public_repo.py
"""
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {'.venv', 'data', 'models', 'logs', 'vendor', 'drivers',
                'local-tools', '__pycache__', '.pytest_cache'}
PRIVATE_SUFFIXES = {'.dpapi', '.pem', '.key', '.pfx', '.p12', '.wav', '.mp3',
                    '.flac', '.onnx', '.safetensors', '.pt', '.pth', '.gguf', '.run'}
IPV4 = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])')
CREDENTIAL = re.compile(
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
    r'\b(?:gsk_|ghp_|github_pat_|sk-proj-)[A-Za-z0-9_-]{16,}|'
    r'https?://[^\s/@:]+:[^\s/@]+@')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def local_private_values():
    values = set()

    def visit(value, key=''):
        if isinstance(value, dict):
            for k, v in value.items():
                visit(v, k)
        elif isinstance(value, list):
            for v in value:
                visit(v, key)
        elif isinstance(value, str):
            if any(word in key.lower() for word in ('token', 'password', 'secret', 'api_key')) and len(value) >= 6:
                values.add(value)
            if 'url' in key.lower():
                host = urlsplit(value).hostname
                if host and not host.endswith('.invalid'):
                    values.add(host)

    path = ROOT / 'config.json'
    if path.exists():
        visit(json.loads(path.read_text(encoding='utf-8')))
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                visit(value.strip().strip('\"\''), key)
    return values


def main():
    paths = git('ls-files', '-z').decode().split('\0')
    secrets = local_private_values()
    problems = []
    count = 0
    for name in filter(None, paths):
        path = Path(name)
        count += 1
        if (set(path.parts) & PRIVATE_DIRS or path.suffix.lower() in PRIVATE_SUFFIXES
                or path.name == 'config.json'
                or path.name.startswith('.env') and path.name != '.env.sample'):
            problems.append((name, 'private artifact is indexed'))
        blob = git('show', ':' + name)
        try:
            content = blob.decode('utf-8-sig')
        except UnicodeDecodeError:
            problems.append((name, 'unexpected binary file'))
            continue
        if any(secret in content for secret in secrets):
            problems.append((name, 'matches local private configuration'))
        if CREDENTIAL.search(content):
            problems.append((name, 'possible embedded credential'))
        # Exact package==version lines contain four-part versions, not hosts.
        ip_content = re.sub(r'(?m)^[A-Za-z0-9_.-]+==[A-Za-z0-9.+_-]+\s*$', '', content) if path.name.startswith('requirements') else content
        for match in IPV4.findall(ip_content):
            try:
                address = ipaddress.ip_address(match)
            except ValueError:
                continue
            # This is a generic wildcard bind, never a destination host.
            if address.is_unspecified and name == 'jarvis/server.py':
                continue
            problems.append((name, 'literal IP address'))
            break
        if re.search(r'[A-Za-z]:[\\/](?:Users|Projects)[\\/]', content, re.I):
            problems.append((name, 'machine-specific filesystem path'))
    if not count:
        print('Nothing indexed; stage the intended public files first.')
        return 1
    for name, reason in problems:
        print(f'{name}: {reason} (value withheld)')
    if problems:
        return 1
    print(f'PASS: {count} indexed text files; no private artifacts, hosts, or detected secrets.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
