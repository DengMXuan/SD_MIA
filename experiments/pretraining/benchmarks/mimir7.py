"""Robustness: Pythia main/baseline under low-overlap MIMIR 7_gram_0.2."""
from experiments.paths import ROOT
from experiments.launchers.pretraining import Benchmark, run

SPEC = Benchmark(
    name='robustness_pythia_mimir7gram02', legacy_group='mimir02',
    sources=('github', 'arxiv'),
    data_root=ROOT / 'artifacts/data/paper_positive_controls_v1/mimir_7_0.2',
    manifest_pattern='{source}/seed{seed}/manifest.json',
    benchmark_pattern='mimir/{source}/ngram_7_0.2',
    counts={'github': (268, 268), 'arxiv': (400, 400)},
    note='Low-overlap diagnostic; 600 disjoint auxiliary negatives come from 13_gram_0.8.',
)

if __name__ == '__main__':
    run(SPEC)
