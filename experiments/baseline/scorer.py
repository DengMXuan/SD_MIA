"""Frozen target scoring and generation, independent of experiment orchestration."""
from __future__ import annotations
from functools import lru_cache
from typing import Any, Iterable
import numpy as np
import torch
import torch.nn.functional as F
from experiments.shared.data.data import SFTRecord, prompt_prefix_ids
from experiments.baseline.costs import CostMeter
from experiments.baseline.methods import mean_log_likelihood, sead_score

from .types import TokenStats

def _response_ids(record: SFTRecord, tokenizer: Any) -> list[int]:
    values = list(record.response_ids)
    if record.append_eos and tokenizer.eos_token_id is not None:
        values.append(int(tokenizer.eos_token_id))
    return values



class TargetScorer:
    """Teacher-forced and label-only probes against one fine-tuned target."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        sead_samples: int,
        sead_temperature: float,
        seed: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sead_samples = sead_samples
        self.sead_temperature = sead_temperature
        self.seed = seed
        self.cost_meter: CostMeter | None = None
        # Input embeddings can occupy >1 GB for an 8B model.  Keep them lazy:
        # only PETAL requests the target-only semantic proxy.
        self.embedding_weight: torch.Tensor | None = None

    def _target_embedding_weight(self) -> torch.Tensor | None:
        if self.embedding_weight is None:
            try:
                self.embedding_weight = self.model.get_input_embeddings().weight.detach()
            except (AttributeError, RuntimeError):
                return None
        return self.embedding_weight

    def _input_ids(self, record: SFTRecord, context_ids: Iterable[int] = ()) -> tuple[list[int], int]:
        prompt_ids = list(context_ids) + list(prompt_prefix_ids(record, self.tokenizer))
        return prompt_ids + _response_ids(record, self.tokenizer), len(prompt_ids)

    def _forward_response(
        self,
        record: SFTRecord,
        context_ids: Iterable[int] = (),
        need_similarity: bool = False,
        need_sead: bool = False,
    ) -> TokenStats:
        input_values, response_start = self._input_ids(record, context_ids)
        if len(input_values) < response_start + 1:
            raise ValueError(f"record {record.record_id} has no response tokens")
        if self.cost_meter is not None:
            self.cost_meter.forward(len(input_values))
        input_ids = torch.tensor([input_values], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[0, :-1]
        labels = input_ids[0, 1:]
        # The first response token at input index response_start is predicted by
        # logits[response_start - 1].
        response_mask = torch.arange(labels.numel(), device=self.device) >= response_start - 1
        # Select response positions before converting to FP32.  ArXivTection
        # uses 2,048-token responses; converting the complete [sequence,
        # vocabulary] projection first needlessly creates a multi-GB FP32
        # copy, especially for ICP contexts containing an auxiliary probe.
        selected_logits = logits[response_mask]
        del logits
        selected_labels = labels[response_mask]
        token_logp_parts = []
        expected_parts = []
        variance_parts = []
        for chunk_logits, chunk_labels in zip(
            selected_logits.split(256), selected_labels.split(256)
        ):
            chunk_logits = chunk_logits.float()
            log_probs = F.log_softmax(chunk_logits, dim=-1)
            probs = log_probs.exp()
            chunk_logp = log_probs.gather(-1, chunk_labels[:, None]).squeeze(-1)
            chunk_expected = (probs * log_probs).sum(-1)
            chunk_variance = (
                (probs * log_probs.square()).sum(-1) - chunk_expected.square()
            )
            token_logp_parts.append(chunk_logp.cpu().numpy())
            expected_parts.append(chunk_expected.cpu().numpy())
            variance_parts.append(chunk_variance.cpu().numpy())
            del chunk_logits, log_probs, probs, chunk_logp
        token_logp = np.concatenate(token_logp_parts)
        expected = np.concatenate(expected_parts)
        variance = np.concatenate(variance_parts)

        log_similarity = None
        if need_similarity:
            predicted = selected_logits.argmax(-1)
            embedding_weight = self._target_embedding_weight()
            if embedding_weight is None:
                similarity = (predicted == selected_labels).float()
            else:
                target_vectors = embedding_weight[selected_labels].float()
                predicted_vectors = embedding_weight[predicted].float()
                similarity = F.cosine_similarity(target_vectors, predicted_vectors, dim=-1)
                similarity = ((similarity + 1.0) / 2.0).clamp_min(1e-12)
                similarity = torch.maximum(
                    similarity, (predicted == selected_labels).float()
                )
            log_similarity = similarity.clamp_min(1e-12).log().cpu().numpy()

        sampled_token_ids = None
        sead_log_density = None
        sead_lexical_density = None
        if need_sead:
            temperature = max(float(self.sead_temperature), 1e-4)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed + int(record.record_id.encode().hex()[:8], 16) % 1_000_000)
            # Vectorize over token rows in bounded chunks.  Calling
            # ``multinomial`` once per token launches millions of tiny GPU
            # kernels for the 4,000-record audit, while chunking preserves the
            # same Monte Carlo estimator and keeps the vocabulary matrix
            # bounded for long ArXivTection responses.
            sampled_chunks = []
            for chunk in selected_logits.split(256):
                chunk_probs = F.softmax(chunk.float() / temperature, dim=-1)
                sampled_chunks.append(
                    torch.multinomial(
                        chunk_probs,
                        self.sead_samples,
                        replacement=True,
                        generator=generator,
                    ).cpu().numpy()
                )
                del chunk_probs
            sampled_token_ids = np.concatenate(sampled_chunks, axis=0)
            # This is SEAD's official surrogate-free frequency estimator.  The
            # NLI/semantic variant is intentionally not enabled here because
            # the user requested a single target-model implementation.
            sead_log_density, sead_lexical_density = sead_score(
                sampled_token_ids, selected_labels.cpu().numpy()
            )

        del selected_logits, input_ids
        return TokenStats(
            token_ids=selected_labels.cpu().numpy(),
            token_logp=token_logp,
            expected_logp=expected,
            variance_logp=variance,
            log_similarity=log_similarity,
            sampled_token_ids=sampled_token_ids,
            sead_log_density=sead_log_density,
            sead_lexical_density=sead_lexical_density,
        )

    def stats(
        self, record: SFTRecord, need_similarity: bool = False, need_sead: bool = False
    ) -> TokenStats:
        return self._forward_response(
            record, need_similarity=need_similarity, need_sead=need_sead
        )

    def mean_ll(self, record: SFTRecord, context_ids: Iterable[int] = ()) -> float:
        stats = self._forward_response(record, context_ids=context_ids)
        return mean_log_likelihood(stats.token_logp)

    @lru_cache(maxsize=None)
    def _probe_vector(self, record: SFTRecord) -> torch.Tensor | None:
        weight = self._target_embedding_weight()
        if weight is None:
            return None
        ids = list(prompt_prefix_ids(record, self.tokenizer)) + list(record.response_ids)
        vectors = weight[torch.tensor(ids, dtype=torch.long, device=weight.device)].float()
        return F.normalize(vectors.mean(dim=0), dim=0)

    def probe_similarity(self, left: SFTRecord, right: SFTRecord) -> float:
        """Rank ICP auxiliary probes with the target's own input embeddings.

        ICP-MIA's reference-data path retrieves semantically similar
        demonstrations before measuring the optimization gap.  The official
        implementation uses an external sentence encoder; this target-only
        version uses the fine-tuned target's input-embedding space instead.
        It falls back to token Jaccard only for architectures without input
        embeddings.
        """

        left_vector = self._probe_vector(left)
        right_vector = self._probe_vector(right)
        if left_vector is None or right_vector is None:
            return _jaccard(left.response_ids, right.response_ids)
        return float(torch.dot(left_vector, right_vector).detach().cpu())

    def build_probe_matrix(
        self, records: list[SFTRecord]
    ) -> torch.Tensor | None:
        """Precompute normalized target-embedding probes for ICP retrieval.

        The original per-candidate implementation synchronizes one GPU dot
        product to the CPU for every auxiliary record.  ICP-MIA ranks the same
        auxiliary pool repeatedly, so retaining the target-only vectors and
        using one matrix product per audit record is mathematically identical
        and avoids millions of tiny synchronization points.
        """

        if self._target_embedding_weight() is None:
            return None
        with torch.inference_mode():
            return torch.stack([self._probe_vector(record) for record in records])

    def top_probe_records(
        self,
        record: SFTRecord,
        candidates: list[SFTRecord],
        probe_matrix: torch.Tensor | None,
        top_k: int,
    ) -> list[SFTRecord]:
        """Return the target-only nearest auxiliary records for ICP-MIA."""

        limit = min(max(1, int(top_k)), len(candidates))
        if probe_matrix is None:
            return sorted(
                candidates,
                key=lambda candidate: self.probe_similarity(record, candidate),
                reverse=True,
            )[:limit]
        with torch.inference_mode():
            scores = torch.mv(probe_matrix, self._probe_vector(record))
            indices = torch.topk(scores, k=limit, largest=True).indices
        return [candidates[int(index)] for index in indices.detach().cpu().tolist()]

    def _max_context(self) -> int:
        value = getattr(self.model.config, "max_position_embeddings", None)
        if value is None or value > 1_000_000:
            value = 32768
        return int(value)

    def fit_context(self, context_ids: Iterable[int], record: SFTRecord) -> list[int]:
        context = list(context_ids)
        prompt = list(prompt_prefix_ids(record, self.tokenizer))
        response = _response_ids(record, self.tokenizer)
        budget = max(0, self._max_context() - len(prompt) - len(response))
        return context[-budget:] if budget else []

    def generate(
        self,
        input_ids: Iterable[int],
        max_new_tokens: int,
        seed: int,
        sample: bool,
        num_return_sequences: int = 1,
    ) -> list[list[int]]:
        values = list(input_ids)
        if max_new_tokens <= 0:
            return [[] for _ in range(num_return_sequences)]
        tensor = torch.tensor([values], dtype=torch.long, device=self.device)
        attention = torch.ones_like(tensor)
        kwargs: dict[str, Any] = {
            "input_ids": tensor,
            "attention_mask": attention,
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": int(self.tokenizer.pad_token_id),
        }
        if sample:
            kwargs.update({"temperature": 1.0, "top_k": 50, "top_p": 1.0})
        # Transformers 5.16 routes sampling through torch's global RNG and
        # does not accept ``generator=`` as a generate() kwarg.  Forking the
        # relevant RNG makes each probe reproducible without perturbing the
        # caller's global random state.
        cuda_devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            with torch.inference_mode():
                output = self.model.generate(**kwargs)
        width = tensor.shape[1]
        sequences = [row[width:].detach().cpu().tolist() for row in output]
        if self.cost_meter is not None:
            self.cost_meter.generation([values], [sequences], self._generation_eos_ids())
        return sequences

    def generate_batch(
        self,
        input_ids: list[list[int]],
        max_new_tokens: int,
        seed: int,
        sample: bool,
        num_return_sequences: int = 1,
    ) -> list[list[list[int]]]:
        """Generate for left-padded inputs in one target-only batch.

        The caller may trim each returned continuation to its own requested
        length.  Greedy generation is unchanged by that trimming, while the
        shared batch seed keeps sampled probes reproducible as a group.
        """

        if not input_ids:
            return []
        if max_new_tokens <= 0:
            return [[[] for _ in range(num_return_sequences)] for _ in input_ids]
        width = max(len(values) for values in input_ids)
        pad_id = int(self.tokenizer.pad_token_id)
        tensor = torch.full(
            (len(input_ids), width), pad_id, dtype=torch.long, device=self.device
        )
        attention = torch.zeros_like(tensor)
        for row, values in enumerate(input_ids):
            length = len(values)
            tensor[row, width - length :] = torch.tensor(
                values, dtype=torch.long, device=self.device
            )
            attention[row, width - length :] = 1
        kwargs: dict[str, Any] = {
            "input_ids": tensor,
            "attention_mask": attention,
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_id,
        }
        if sample:
            kwargs.update({"temperature": 1.0, "top_k": 50, "top_p": 1.0})
        cuda_devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            with torch.inference_mode():
                output = self.model.generate(**kwargs)
        continuations = output[:, width:].detach().cpu().tolist()
        grouped = []
        for row in range(len(input_ids)):
            start = row * num_return_sequences
            grouped.append(
                continuations[start : start + num_return_sequences]
            )
        if self.cost_meter is not None:
            self.cost_meter.generation(input_ids, grouped, self._generation_eos_ids())
        return grouped

    def _generation_eos_ids(self):
        config = getattr(self.model, "generation_config", None)
        if config is None:
            config = getattr(getattr(self.model, "model", None), "generation_config", None)
        ids = getattr(config, "eos_token_id", None)
        if ids is None:
            ids = self.tokenizer.eos_token_id
        return set(ids if isinstance(ids, (list, tuple)) else [ids]) if ids is not None else set()
