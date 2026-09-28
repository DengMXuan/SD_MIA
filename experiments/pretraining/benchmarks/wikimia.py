"""Effectiveness diagnostic: Pythia main/baseline on WikiMIA 64/128 words."""
from experiments.paths import ROOT
from experiments.launchers.pretraining import Benchmark, run

SPEC = Benchmark(
    name='effectiveness_pythia_wikimia64_128', legacy_group='wikimia',
    sources=('64', '128'),
    data_root=ROOT / 'artifacts/data/paper_positive_controls_v1',
    manifest_pattern='wikimia_2024plus_{source}/seed{seed}/manifest.json',
    benchmark_pattern='wikimia/official_length{source}/temporal_proxy',
    counts={'64': (284, 258), '128': (139, 111)},
    kind='temporal_pretraining_v1', source_prefix='length',
    note='Temporal proxy labels, not verified Pile membership; 64 words is primary, 128 supplementary.',
)

if __name__ == '__main__':
    run(SPEC)
