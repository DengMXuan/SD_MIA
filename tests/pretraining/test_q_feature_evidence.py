"""Observable boundaries and fixed-document behavior for q-feature exploration."""
import numpy as np
import pytest

from experiments.shared.methods.q_feature_accept_only import (
    position_features, fit_document_reference, standardize, round1_fusions,
    q_context, fit_nonmember_counts, count_position_features,
    round3_fusions,
)


def sample(length=10):
    x = np.zeros((length, 6))
    x[:, 0] = -np.linspace(5, .1, length)
    x[:, 1] = np.linspace(.2, .6, length)
    x[:, 2] = np.linspace(.4, 0, length)
    x[:, 3] = 1
    x[:, 4] = np.linspace(0, 1, length)
    return x


def test_q_positions_are_selected_before_acceptance_is_seen():
    x = sample()
    c = np.zeros(10, int)
    c[:2] = 2
    first = position_features(x, c, np.array([10]), vocab_size=50000)
    c[2:] = 2
    second = position_features(x, c, np.array([10]), vocab_size=50000)
    assert first['accept_lowq_20'][0] == second['accept_lowq_20'][0] == 1
    assert first['q_lowq_20'][0] == second['q_lowq_20'][0]
    assert first['accept_mean'][0] < second['accept_mean'][0]


def test_document_scoring_does_not_use_other_documents():
    x = sample()
    c = np.arange(10) % 3
    one = position_features(x, c, np.array([10]), vocab_size=50000)
    many = position_features(np.r_[x, sample(4)], np.r_[c, np.ones(4, int)],
                             np.array([10, 4]), vocab_size=50000)
    for name in one:
        assert np.isfinite(one[name]).all()
        assert one[name][0] == many[name][0]


def test_fusions_preserve_frozen_reference_and_disable_constant_channels():
    values = {k: np.arange(4, dtype=float) for k in position_features(sample(), np.ones(10, int), np.array([10]), vocab_size=50000)}
    reference = fit_document_reference(values)
    batch = {k: np.r_[v, 1e5] for k,v in values.items()}
    a,b = round1_fusions(standardize(values, reference)), round1_fusions(standardize(batch, reference))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k][:4])
    constant = fit_document_reference({k:np.ones(3) for k in values})
    assert all((v == 0).all() for v in standardize(values, constant).values())


def test_invalid_count_or_length_is_rejected():
    with pytest.raises(ValueError):
        position_features(sample(), np.full(10, 3), np.array([10]), vocab_size=50000)
    with pytest.raises(ValueError):
        position_features(sample(), np.zeros(10, int), np.array([9]), vocab_size=50000)


def test_context_windows_never_cross_document_boundaries():
    x = sample()
    one = q_context(x, np.array([len(x)]))
    other = sample(5)
    other[:, 0] = -10000
    batch = q_context(np.r_[x, other], np.array([len(x), len(other)]))
    np.testing.assert_allclose(one, batch[:len(x)], atol=0, rtol=0)


def test_nonmember_predictor_learns_difficulty_on_held_out_documents():
    rng = np.random.default_rng(44)
    train_x = rng.normal(size=(1600, 7))
    val_x = rng.normal(size=(400, 7))
    counts = rng.binomial(2, np.where(train_x[:,0] > 0, .9, .1))
    val_counts = rng.binomial(2, np.where(val_x[:,0] > 0, .9, .1))
    model, report = fit_nonmember_counts(train_x, counts, np.full(80,20),
                                        val_x, val_counts, np.full(20,20))
    p = model['model'].predict_proba(val_x[:,:model['columns']])
    assert p.shape == (400,3) and np.allclose(p.sum(1),1)
    assert min(report['nonmember_validation_nll'].values()) < .8
    assert report['train_member_count'] == report['validation_member_count'] == 0


def test_predicted_position_choice_is_independent_of_realized_accepts():
    pi = np.tile([.01, .08, .91], (10,1))
    pi[:4] = [.81,.18,.01]
    x, c = sample(), np.zeros(10,int)
    c[:4] = 1
    before = count_position_features(x,c,np.array([10]),pi)
    c[4:] = 2
    after = count_position_features(x,c,np.array([10]),pi)
    assert before['nm_low_accept_40'][0] == after['nm_low_accept_40'][0] == .5
    assert after['nm_difficulty_r0'][0] > before['nm_difficulty_r0'][0]


def test_final_gate_uses_fixed_nonmember_center_and_has_no_batch_dependence():
    z = {k:np.array([-2., 1.]) for k in
         ('accept_mean','q_mean','accept_qhard_5','accept_qcenter_60','q_qcenter_20')}
    learned = {k:np.array([.2,.3]) for k in ('nm_difficulty_r0','nm_low_accept_40')}
    before,recipes = round3_fusions(z,learned)
    after,_ = round3_fusions({k:np.r_[v,1e8] for k,v in z.items()},
                             {k:np.r_[v,-1e8] for k,v in learned.items()})
    assert len(before)==len(recipes)==57
    for k in before:
        np.testing.assert_array_equal(before[k],after[k][:2])
    final=before['qref_multiscale_negative_q50_c25']
    anchor=(z['accept_mean']+learned['nm_low_accept_40']+z['accept_qcenter_60'])/3
    assert final[1]==anchor[1]+.25*z['q_qcenter_20'][1]


def test_global_configuration_selection_obeys_each_condition_guardrail():
    from experiments.pretraining.q_feature_exploration import choose_global, PREVIOUS
    reports=[{'task_key':'a','development_auc':{PREVIOUS:.7,'unsafe':.9,'safe':.71}},
             {'task_key':'b','development_auc':{PREVIOUS:.6,'unsafe':.59,'safe':.601}}]
    chosen,_=choose_global(reports)
    assert chosen['method']=='safe'


def test_confirmation_measure_uses_only_protected_indices_and_keeps_ties():
    from experiments.pretraining.q_feature_confirmation import measure
    saved={'labels':np.array([0,0,1,0,1,0]),'calibration':np.array([0,1])}
    scores={'primary':np.array([0.,0.,1.,0.,-1e6,1e6]),'control':np.zeros(6)}
    result,pairs=measure(scores,saved,np.array([2,3]),'primary',1919,8)
    assert result['primary']['auc']==1 and result['control']['auc']==.5
    assert pairs[0]['auc_ci95']==[.5,.5]
    assert result['control']['roc_tpr_at_1pct_fpr']==0
