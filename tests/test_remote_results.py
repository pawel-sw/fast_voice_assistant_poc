from jarvis import observability as obs
from jarvis.remote_client import show


def test_remote_results_share_one_clean_wake_line(tmp_path, monkeypatch):
    path = tmp_path/'transcript.txt'
    monkeypatch.setattr(obs, 'plain_transcript', obs.PlainTranscript(path))
    monkeypatch.setattr(obs, 'candidate_segments', None)
    obs.plain_candidate()
    show({'type': 'transcript', 'utterance': 0, 'text': 'Hey Jarvis'})
    assert path.read_text() == ''
    show({'type': 'confirmed'})
    show({'type': 'transcript', 'utterance': 0, 'text': 'Hey Jarvis temperature outside?', 'final': True})
    show({'type': 'result', 'item': 'Outside_Temperature', 'location': 'outside', 'result': '17.85 °C'})
    show({'type': 'done'})
    text = path.read_text(encoding='utf-8')
    assert text == 'Hey Jarvis temperature outside? [{"item":"Outside_Temperature","query":"outside_temperature","result":"17.85 °C"}]'


def test_remote_api_error_is_visible_without_success_result(tmp_path, monkeypatch):
    path = tmp_path/'transcript.txt'
    monkeypatch.setattr(obs, 'plain_transcript', obs.PlainTranscript(path))
    obs.plain_wake()
    show({'type': 'result', 'item': 'Outside_Temperature', 'location': 'outside', 'error': 'Timeout'})
    assert '"error":"Timeout"' in path.read_text()
    assert '"result"' not in path.read_text()
