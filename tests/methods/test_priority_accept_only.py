import numpy as np
import json
import pytest
import torch
from scipy.special import logsumexp
from experiments.sd_membership_sft.archive.priority_accept_only import CausalCountTCN, aggregate, uncertainty_discount, expanded_calibration, group_pvalues, allocation_scores, fit, predict
from experiments.shared.methods.conditional_accept_only import directional_scores
from experiments.shared.protocols.collect_draft_difficulty import finalize_archive


def test_collection_can_finalize_existing_array_after_interruption(tmp_path):
    q=np.ones((4,6),dtype=np.float32)
    np.save(tmp_path/'q.npy',q)
    before=(tmp_path/'q.npy').read_bytes()
    finalize_archive(tmp_path,{'provenance':{'role':'draft_auxiliary_distilled'}},q.shape)
    assert (tmp_path/'q.npy').read_bytes()==before
    assert json.loads((tmp_path/'SOURCE.json').read_text())['models_frozen']
    with pytest.raises(ValueError,match='invalid final'):
        finalize_archive(tmp_path,{'provenance':{}},(5,6))


def test_causal_count_predictions_cannot_read_current_or_future_counts():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model=CausalCountTCN(channels=4).eval()
    x=torch.randn(2,12,2);mask=torch.ones(2,12,dtype=torch.bool)
    counts=torch.randint(0,3,(2,12))
    changed=counts.clone();changed[:,5:]=(changed[:,5:]+1)%3
    with torch.no_grad():
        before=model(x,mask,counts);after=model(x,mask,changed)
    torch.testing.assert_close(before[:,:6],after[:,:6])
    assert not torch.allclose(before[:,6:],after[:,6:])
    torch.testing.assert_close(before.exp().sum(-1),torch.ones(2,12))
    (-model(x,mask,counts)[:,:,0].mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_global_and_sparse_scores_match_explicit_likelihood_mixtures():
    p=np.array([[.6,.3,.1],[.1,.3,.6]])
    counts=np.array([2,1]);lengths=np.array([2])
    original,_=directional_scores(np.log(p),counts)
    np.testing.assert_allclose(aggregate(np.log(p),counts,lengths),[original])
    ratios=[]
    for rho in (.05,.1,.25):
        for eta in (.5,1.,2.):
            alternative=p*np.exp(eta*np.arange(3));alternative/=alternative.sum(1,keepdims=True)
            mixed=(1-rho)*p+rho*alternative
            ratios.append(np.prod(mixed[np.arange(2),counts]/p[np.arange(2),counts]))
    np.testing.assert_allclose(aggregate(np.log(p),counts,lengths,sparse=True),[np.log(np.mean(ratios))])


def test_uncertainty_is_zero_for_identical_models_and_grows_with_disagreement():
    p=np.array([[.2,.3,.5],[.6,.3,.1]])
    mean,discount=uncertainty_discount(np.log(np.stack([p,p,p])))
    np.testing.assert_allclose(np.exp(mean),p)
    np.testing.assert_allclose(discount,1.)
    _,varied=uncertainty_discount(np.log(np.stack([p,p,p[:,::-1]])))
    assert np.all(varied<1)


def test_expanded_calibration_excludes_fit_validation_and_test():
    labels=np.r_[np.zeros(2000),np.ones(400)]
    parts={'reference':np.arange(400),'calibration':np.arange(400,600),
           'test':np.r_[np.arange(600,1000),np.arange(2000,2400)]}
    full=expanded_calibration(labels,parts)
    assert len(full)==1200
    np.testing.assert_array_equal(full[:200],parts['calibration'])
    assert not np.intersect1d(full,np.r_[parts['reference'],parts['test']]).size
    p=group_pvalues(np.array([0.,1.,2.]),np.array([0]),np.array([1,2]),np.array([0,0,1]))
    np.testing.assert_allclose(p,[.5,1.])


def test_allocation_with_one_token_never_reads_second_bit():
    weights=np.log(np.full((2,18),1/18))
    bits=np.array([[1,0],[0,1]],dtype=np.uint8)
    original=allocation_scores(weights,bits,np.ones(2,dtype=int),seed=1)
    bits[:,1]=1-bits[:,1]
    altered=allocation_scores(weights,bits,np.ones(2,dtype=int),seed=1)
    for method in original:
        np.testing.assert_allclose(original[method],altered[method])
        assert np.isfinite(original[method]).all()
    np.testing.assert_allclose(original['allocation_uniform'],original['allocation_entropy'])


def test_training_is_invariant_to_heldout_features_and_counts():
    torch.set_num_threads(1)
    rng=np.random.default_rng(4)
    x=rng.normal(size=(32,2)).astype(np.float32)
    counts=rng.integers(0,3,size=32)
    parts={'train':np.array([0,1]),'validation':np.array([2])}
    first=fit(x,counts,np.full(4,8),parts,seed=5,device='cpu',causal=True,epochs=2)
    x[24:]*=30;counts[24:]=(counts[24:]+1)%3
    second=fit(x,counts,np.full(4,8),parts,seed=5,device='cpu',causal=True,epochs=2)
    assert first[3]==second[3]
    for name,value in first[0].state_dict().items():
        torch.testing.assert_close(value,second[0].state_dict()[name])
