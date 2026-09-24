import numpy as np
import torch
from experiments.shared.methods.combined_accept_only import calibrated_variants, restore
from experiments.shared.methods.conditional_accept_only import ConditionalCountTCN
from experiments.sd_membership_sft.analysis.directional_mia import conformal_tail_pvalues


def test_each_score_gets_its_own_calibration_and_reference_only_boundary():
    reference=np.array([0,1]);cal=np.array([2,3]);full=np.array([2,3,4,5]);test=np.array([6,7])
    difficulty=np.array([0.,2.,0.,2.,0.,2.,0.,2.])
    first=np.array([0.,0.,.2,.4,.6,.8,.5,.7])
    second=np.array([0.,0.,.9,.3,.7,.2,.5,.7])
    values,groups,boundary=calibrated_variants({'first':first,'second':second},cal,full,test,difficulty,reference)
    assert boundary==1.
    np.testing.assert_allclose(values['first__pooled1200'],conformal_tail_pvalues(first[test],first[full]))
    np.testing.assert_allclose(values['second__pooled1200'],conformal_tail_pvalues(second[test],second[full]))
    assert not np.array_equal(values['first__pooled200'],values['second__pooled200'])
    for name,scores in (('first',first),('second',second)):
        expected=[conformal_tail_pvalues(scores[[i]],scores[full[groups[full]==groups[i]]])[0] for i in test]
        np.testing.assert_allclose(values[name+'__grouped1200'],expected)
    changed=difficulty.copy();changed[2:]*=20
    _,_,same=calibrated_variants({'first':first},cal,full,test,changed,reference)
    assert same==boundary


def test_other_test_scores_do_not_change_a_records_calibrated_pvalue():
    scores=np.arange(8,dtype=float);reference=np.array([0,1]);cal=np.array([2,3]);full=np.array([2,3,4,5]);test=np.array([6,7])
    q=np.arange(8,dtype=float)%2
    before,_,_=calibrated_variants({'s':scores},cal,full,test,q,reference)
    scores[7]=-100
    after,_,_=calibrated_variants({'s':scores},cal,full,test,q,reference)
    for key in before:assert before[key][0]==after[key][0]


def test_restored_detector_reproduces_pmf_without_training(tmp_path):
    torch.set_num_threads(1)
    model=ConditionalCountTCN(5,2,channels=4).eval()
    path=tmp_path/'difficulty.pt'
    torch.save({'state_dict':model.state_dict(),'mean':torch.zeros(5),'scale':torch.ones(5),'causal':False},path)
    restored,_,_=restore(path,5,'cpu')
    x=torch.randn(1,7,5);mask=torch.ones(1,7,dtype=torch.bool)
    with torch.no_grad():torch.testing.assert_close(model(x,mask),restored(x,mask))
    assert not any(p.requires_grad for p in restored.parameters())


def test_shared_record_bootstrap_is_not_narrowed_by_duplicate_seeds_or_checkpoints():
    from experiments.sd_membership_sft.analysis.summarize_combination_validation import paired_intervals, DECISION_PAIRS
    rng=np.random.default_rng(54)
    archive={'labels':np.r_[np.zeros(16),np.ones(16)],'test':np.arange(32),
             'record_ids':np.array([f'r{i}' for i in range(32)])}
    base=rng.normal(size=32)
    for name in ('q_global','q_sparse','difficulty_global','difficulty_sparse'):
        archive[name]=base+rng.normal(scale=.4,size=32)
    values={name:rng.uniform(0,.04,size=32) for pair in DECISION_PAIRS for name in pair}
    single=paired_intervals([archive],[values],[{'benchmark':'wiki','epoch':1}],30)
    doubled=paired_intervals([archive,archive],[values,values],[{'benchmark':'wiki','epoch':1}]*2,30)
    checkpoints=paired_intervals([archive,archive],[values,values],[{'benchmark':'wiki','epoch':1},{'benchmark':'wiki','epoch':3}],30)
    assert single==doubled==checkpoints
