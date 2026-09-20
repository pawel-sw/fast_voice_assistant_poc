"""Deployment helper. Credentials come from VM_SSH_PASSWORD, never source."""
import argparse
import os
from pathlib import Path
import shlex
import sys

import paramiko

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

parser = argparse.ArgumentParser()
parser.add_argument('--put', nargs=2, metavar=('LOCAL', 'REMOTE'))
parser.add_argument('--get', nargs=2, metavar=('REMOTE', 'LOCAL'))
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
hosts = root / 'data/vm_known_hosts'
hosts.parent.mkdir(exist_ok=True)
ssh = paramiko.SSHClient()
if hosts.exists():
    ssh.load_host_keys(str(hosts))
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(os.environ['VM_SSH_HOST'], username=os.environ['VM_SSH_USER'], password=os.environ['VM_SSH_PASSWORD'], timeout=15)
ssh.save_host_keys(str(hosts))
try:
    if args.put or args.get:
        with ssh.open_sftp() as sftp:
            if args.put:
                sftp.put(*args.put)
            else:
                sftp.get(*args.get)
    else:
        command = sys.stdin.read()
        _, output, _ = ssh.exec_command('bash -lc ' + shlex.quote(command), timeout=3600)
        output.channel.set_combine_stderr(True)
        for line in output:
            print(line, end='', flush=True)
        sys.exit(output.channel.recv_exit_status())
finally:
    ssh.close()
