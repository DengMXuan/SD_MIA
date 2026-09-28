"""Effectiveness: seven target-only baselines on MIMIR 13_gram_0.8."""
from experiments.paths import ROOT
from experiments.launchers.pretraining import Benchmark, run

SPEC = Benchmark(
    name='effectiveness_pythia_mimir13gram08', legacy_group='mimir08',
    sources=('arxiv', 'dm_mathematics', 'github', 'hackernews', 'pile_cc',
             'pubmed_central', 'wikipedia_(en)'),
    data_root=ROOT.parent / 'SD_MIA-pretraining-data/mimir/prepared',
    manifest_pattern='{source}/seed{seed}/manifest.json',
    benchmark_pattern='mimir/{source}/ngram_13_0.8',
    counts={source: (400, 400) for source in (
        'arxiv', 'dm_mathematics', 'github', 'hackernews', 'pile_cc',
        'pubmed_central', 'wikipedia_(en)')},
    historical_main=ROOT / 'artifacts/audits/pythia_mimir_v1/tasks',
    note='13_gram_0.8 primary MIMIR evidence; saved main results are read-only.',
)

if __name__ == '__main__':
    run(SPEC)
