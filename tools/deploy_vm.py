"""Copy application/config to the VM, preserving its environment and models."""
import json
import os
from pathlib import Path
import secrets
import paramiko
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
DEST = os.getenv('VM_PROJECT_DIR', '/p/home_assistant')
ssh = paramiko.SSHClient()
ssh.load_host_keys(str(ROOT/'data/vm_known_hosts'))
ssh.connect(os.environ['VM_SSH_HOST'], username=os.environ['VM_SSH_USER'], password=os.environ['VM_SSH_PASSWORD'])
try:
    with ssh.open_sftp() as sftp:
        for name in ('jarvis', 'tests', 'tools', 'clients'):
            try:
                sftp.mkdir(DEST+'/'+name)
            except OSError:
                pass
            for path in (ROOT/name).glob('*.py'):
                sftp.put(str(path), DEST+'/'+name+'/'+path.name)
        for name in ('requirements-server.txt', 'jarvis-vm.service', 'pytest.ini', 'README.md', 'API.md'):
            sftp.put(str(ROOT/name), DEST+'/'+name)
        config = json.loads((ROOT/'config.json').read_text())
        server_keys = ('chunk_seconds', 'tts_voice', 'tts_speed', 'tts_device',
                       'asr_gpu_memory_utilization', 'server_max_connections')
        config = {key: config[key] for key in server_keys if key in config}
        config['server_asr_model'] = DEST+'/models/R2T2'
        with sftp.open(DEST+'/config.json', 'w') as file:
            file.write(json.dumps(config, indent=2)+'\n')
        values = dotenv_values(ROOT/'.env')
        token = values.get('JARVIS_REMOTE_TOKEN') or secrets.token_urlsafe(48)
        values['JARVIS_REMOTE_TOKEN'] = token
        # Preserve existing client secrets and formatting. Only append a missing token.
        if not dotenv_values(ROOT/'.env').get('JARVIS_REMOTE_TOKEN'):
            with (ROOT/'.env').open('a', encoding='utf-8') as file:
                file.write(f'\nJARVIS_REMOTE_TOKEN={token}\n')
        with sftp.open(DEST+'/.env', 'w') as file:
            file.write(f'JARVIS_REMOTE_TOKEN={token}\nNEEDLE_TELEMETRY=0\n')
        sftp.chmod(DEST+'/.env', 0o600)
    print('Application deployed to '+DEST)
finally:
    ssh.close()
