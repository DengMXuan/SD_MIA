"""Independent WikiMIA data preparation and auxiliary collection; no GPU."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    from experiments.pretraining.benchmarks.prepare import DATA_ROOT, SEEDS, _safe_output, prepare_wikimia
    from experiments.pretraining.benchmarks.reuse_aux import (
        REUSED_AUXILIARY, WIKITECTION_POOL, COLLECTED_CHECKPOINT, build,
    )
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', allow_abbrev=False)
    prep.add_argument('--length', type=int, choices=(64, 128), required=True)
    prep.add_argument('--aux-file', type=Path, default=REUSED_AUXILIARY)
    prep.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    reuse = commands.add_parser('build-aux', allow_abbrev=False)
    reuse.add_argument('--output', type=Path, default=REUSED_AUXILIARY)
    reuse.add_argument('--wikitection', type=Path, default=WIKITECTION_POOL)
    reuse.add_argument('--checkpoint', type=Path, default=COLLECTED_CHECKPOINT)
    collect = commands.add_parser('collect-aux', allow_abbrev=False)
    collect.add_argument('--output', type=Path, default=DATA_ROOT / 'wikimia_aux/events_post2023.jsonl')
    collect.add_argument('--records', type=int, default=1000)
    collect.add_argument('--contact', required=True)
    collect.add_argument('--interval', type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        if len(set(args.seeds)) != len(args.seeds):
            parser.error('duplicate seeds')
        for seed in args.seeds:
            print(json.dumps(dict(manifest=str(prepare_wikimia(args.length, seed, args.aux_file)))))
        return
    if not args.output.resolve().is_relative_to(DATA_ROOT.resolve()):
        parser.error(f'auxiliary output must stay under {DATA_ROOT}')
    if args.command == 'build-aux':
        _safe_output(args.output, [args.wikitection, args.checkpoint])
        result = build(args.output, wikitection=args.wikitection, checkpoint=args.checkpoint)
    else:
        from experiments.pretraining.benchmarks.collect_aux import collect
        old_metadata = DATA_ROOT / 'wikimia_aux/from_wikitection_plus_events.manifest.json'
        if (args.output.with_suffix('.checkpoint.jsonl').resolve() == COLLECTED_CHECKPOINT.resolve()
                and (REUSED_AUXILIARY.with_suffix('.manifest.json').exists() or old_metadata.exists())):
            parser.error('default event checkpoint is frozen; collect to a new output path')
        _safe_output(args.output, [])
        result = collect(args.output, records=args.records, contact=args.contact, interval=args.interval)
    print(json.dumps(dict(auxiliary=str(result))))


if __name__ == '__main__':
    main()
