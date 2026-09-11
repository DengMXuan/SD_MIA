import hashlib
import json
import random
import string
from types import SimpleNamespace

from experiments.baseline import prepare_temporal_wiki as prep
from experiments.baseline.run import AuditRecord
from experiments.sd_membership_sft.data import SFTRecord, _hash_ids
from experiments.sd_membership_sft.pools import _write_pool


class Tokenizer:
    def __call__(self, text, **kwargs):
        ids = [ord(c) for c in text]
        return SimpleNamespace(input_ids=ids[:kwargs.get('max_length', len(ids))])


def test_temporal_audit_preserves_negative_ids_and_marks_proxy(tmp_path, monkeypatch):
    tokenizer = Tokenizer()
    rng = random.Random(10)
    def document(year):
        text = ''.join(rng.choices(string.ascii_lowercase + ' ', k=600))
        return dict(text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    creation_timestamp=f'{year}-06-01T00:00:00Z',
                    snapshot_timestamp=f'{year}-12-01T00:00:00Z', snapshot_revision=99,
                    title='Example', record_id=f'raw:{year}', source='wiki')
    positive, negative = document(2023), document(2026)
    historical = tmp_path / 'historical.jsonl'
    negatives = tmp_path / 'negative.jsonl'
    _write_pool(historical, [positive], dict(benchmark='wikitection', label_semantics='presumed_member_temporal_proxy'))
    _write_pool(negatives, [negative], dict(benchmark='wikitection'))
    ids = tokenizer(negative['text'], max_length=512).input_ids
    record = SFTRecord('original-negative-id', 'wiki', tuple(ids), _hash_ids(ids), prompt_ids=())
    cfg = SimpleNamespace(benchmark='wikitection', pool_path=negatives, data_seed=1)
    monkeypatch.setattr(prep, 'load_run_config', lambda _: cfg)
    monkeypatch.setattr(prep, '_target_tokenizer', lambda *a: tokenizer)
    monkeypatch.setattr(prep, 'load_audit_records', lambda *a: ([], [AuditRecord(record, 0)], [], {}))
    output = tmp_path / 'audit'
    prep.prepare([historical], tmp_path, output, count=1)
    rows = [json.loads(line) for line in (output / 'audit.jsonl').read_text().splitlines()]
    assert [r['label'] for r in rows] == [1, 0]
    assert rows[1]['audit_record_id'] == 'original-negative-id'
    assert rows[1]['response_ids'] == ids
    assert all(not r['membership_verified'] for r in rows)
    assert rows[0]['membership_status'] == 'presumed_member_temporal_proxy'
    manifest = json.loads((output / 'audit.manifest.json').read_text())
    assert manifest['cross_split_ngram_audit']['gate'] == 'PASS'


def test_collected_token_selection_survives_frozen_pool_reselection():
    from experiments.sd_membership_sft.pools import _select_wiki_records
    rng = random.Random(22)
    rows = [dict(record_id=str(i), title=str(i), text=''.join(rng.choices(string.ascii_letters + ' ', k=600)))
            for i in range(4)]
    selected = _select_wiki_records(rows, Tokenizer(), 3, 20260824)
    repeated = _select_wiki_records(selected, Tokenizer(), 3, 20260824)
    assert len(selected) == 3
    assert [row['record_id'] for row in selected] == [row['record_id'] for row in repeated]
