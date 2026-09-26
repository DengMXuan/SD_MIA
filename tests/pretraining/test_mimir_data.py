"""Real cache format, capacity and authorization handling without Hub access."""
import json

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError

from experiments.pretraining import prepare
from experiments.pretraining.data import MIMIR_REVISION, load_evaluation
from experiments.pretraining.datasets import download_mimir, inspect_mimir, prepare_mimir
from tests.pretraining.test_pretraining import pretrained_fixture


def test_official_download_preserves_source_and_revision(tmp_path, monkeypatch):
    calls = []

    def downloaded(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(tmp_path / filename)

    monkeypatch.setattr(prepare, 'hf_hub_download', downloaded)
    cache = download_mimir(source='full_pile', split='none', cache_size=10000,
                           local_dir=tmp_path, local_files_only=True)
    assert cache['source_provenance']['revision'] == MIMIR_REVISION
    assert [c[1] for c in calls] == ['cache_100_200_10000_512/train/full_pile.jsonl',
                                   'cache_100_200_10000_512/test/full_pile.jsonl']
    assert all(c[0] == 'iamgroot42/mimir' and c[2]['revision'] == MIMIR_REVISION
               and c[2]['local_files_only'] for c in calls)
    assert 'token' not in json.dumps(cache['source_provenance'])
    with pytest.raises(ValueError, match='full_pile/10000/none'):
        download_mimir(source='full_pile')
    assert len(calls) == 2


def test_gated_cache_error_explains_account_access(monkeypatch):
    def denied(*args, **kwargs):
        response = httpx.Response(403, request=httpx.Request('HEAD', 'https://huggingface.co'))
        raise GatedRepoError('restricted', response=response)
    monkeypatch.setattr(prepare, 'hf_hub_download', denied)
    with pytest.raises(RuntimeError, match='account logged into the local Hugging Face SDK'):
        download_mimir(source='wikipedia_(en)')


def test_readiness_uses_freeze_filters_and_preserves_official_labels(pretrained_fixture, tmp_path):
    root, _, models = pretrained_fixture
    member = tmp_path / 'member.jsonl'
    # A JSON object row is supported alongside official JSON-string rows.
    member.write_text((root / 'member.jsonl').read_text() + json.dumps({'text': 'document item0 alpha beta'}) + '\n')
    nonmember = root / 'nonmember.jsonl'
    check = inspect_mimir(member, nonmember, source='fixture', n_aux=12, models=models)
    assert check['max_balanced_test_per_class'] == 38  # 50 official negatives minus 12 held-out.
    assert check['filtering']['1']['duplicates_removed'] == 1
    assert check['filtering']['0']['token_lengths'] == dict(min=4, max=4, mean=4.)
    for seed in (1919, 1949, 1978):
        args = dict(source='fixture', split='test', n_per_class=38, n_aux=12, models=models, seed=seed)
        manifest = prepare_mimir(member, nonmember, tmp_path / str(seed), **args)
        frozen = json.loads(manifest.read_text())
        assert frozen['filtering'] == check['filtering']
        assert frozen['selection_seed'] == seed
        data = load_evaluation(manifest, verify_draft=True)
        assert (len(data.members), len(data.nonmembers), len(data.auxiliary)) == (38, 38, 12)
        repeat = prepare_mimir(member, nonmember, tmp_path / f'repeat{seed}', **args)
        assert repeat.with_name('records.jsonl').read_bytes() == manifest.with_name('records.jsonl').read_bytes()
    with pytest.raises(ValueError, match='not enough disjoint'):
        prepare_mimir(member, nonmember, tmp_path / 'too-large', source='fixture', split='test',
                      n_per_class=39, n_aux=12, seed=1919, models=models)
    assert not (tmp_path / 'too-large').exists()
    with pytest.raises(ValueError, match='cross-label'):
        inspect_mimir(member, member, source='fixture', models=models)
