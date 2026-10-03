"""Inference interface for an exported, frozen q-reference experiment bundle.

Input is exactly six draft-side features, B=2 acceptance counts and document
lengths. Neither target probabilities nor membership labels enter this API.
Bundles are trusted local experiment artifacts; joblib is not a safe format for
loading files received from untrusted third parties.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib

from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.methods.q_feature_accept_only import (
    position_features, q_context, count_position_features, standardize, round3_fusions,
)


@dataclass
class QReferenceScorer:
    predictor: object
    columns: int
    vocabulary_size: int
    q_normalization: dict
    count_normalization: dict
    method: str = 'qref_multiscale_negative_q50_c25'

    @classmethod
    def from_artifact(cls, directory, *, vocabulary_size):
        directory = Path(directory)
        complete = json.loads((directory/'_COMPLETE.json').read_text())
        for name in ('REPORT.json','nonmember_model.joblib'):
            if sha256_file(directory/name) != complete[name]:
                raise ValueError('frozen scorer artifact checksum mismatch')
        report = json.loads((directory/'REPORT.json').read_text())
        if sha256_file(directory/'nonmember_model.joblib') != report['model_sha256']:
            raise ValueError('model checksum disagrees with report')
        bundle = joblib.load(directory/'nonmember_model.joblib')
        return cls(bundle['model'],bundle['columns'],vocabulary_size,
                   report['normalization']['q'],report['normalization']['learned'],report['primary'])

    def score(self, features, counts, lengths):
        q = position_features(features,counts,lengths,vocab_size=self.vocabulary_size)
        context = q_context(features,lengths)
        pi = self.predictor.predict_proba(context[:,:self.columns])
        learned = count_position_features(features,counts,lengths,pi)
        scores,_ = round3_fusions(standardize(q,self.q_normalization),
                                 standardize(learned,self.count_normalization))
        if self.method not in scores:
            raise ValueError('bundle does not describe a supported frozen q-reference rule')
        return scores[self.method]
