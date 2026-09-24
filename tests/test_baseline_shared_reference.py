"""A shared greedy reference must preserve robustness scores."""

from types import SimpleNamespace

import torch

from experiments.baseline import run as baseline_run


class Tokenizer:
    def __call__(self, text, **_kwargs):
        if _kwargs.get("return_tensors") == "pt":
            return SimpleNamespace(input_ids=torch.tensor([[11, 12]]))
        return SimpleNamespace(input_ids=[11, 12])

    def decode(self, ids, **_kwargs):
        return "generated " + " ".join(map(str, ids))


class Progress:
    def track(self, values, _stage, unit="records"):
        yield from values


class Scorer:
    def __init__(self):
        self.calls = []

    def generate_batch(self, inputs, max_new_tokens, seed, sample, num_return_sequences=1):
        self.calls.append((seed, len(inputs), max_new_tokens, sample, num_return_sequences))
        return [[[7, 8] for _ in range(num_return_sequences)] for _ in inputs]


def test_robustness_methods_share_one_reference_generation(monkeypatch):
    monkeypatch.setattr(baseline_run, "_response_prefix_text", lambda record, _tok, _ratio: ([1, 2], "prefix", "suffix"))
    monkeypatch.setattr(baseline_run, "prompt_prefix_ids", lambda _record, _tok: [3])
    monkeypatch.setattr(baseline_run, "_perturb_words", lambda text, _kind, _rate, _rng: text)
    monkeypatch.setattr(baseline_run, "rouge1_recall", lambda left, right: float(left == right))
    args = SimpleNamespace(prefix_ratio=.5, generation_batch_size=2, seed=17, perturbation_rate=.15,
                           samia_samples=10)
    records = [SimpleNamespace(record=object()) for _ in range(2)]
    tokenizer = Tokenizer()

    scorer = Scorer()
    reference_cache = {}
    scores = {}
    for method in ("ws", "rs", "bt"):
        scores.update(baseline_run._score_methods(args, Progress(), scorer, records, [], tokenizer,
                                                   (method,), reference_cache=reference_cache))

    assert set(scores) == {"ws", "rs", "bt"}
    assert all(values == [1., 1.] for values in scores.values())
    assert len(reference_cache["texts"]) == len(records)
    assert len(scorer.calls) == 5
    assert [call[0] for call in scorer.calls].count(args.seed + 10_000) == 1
