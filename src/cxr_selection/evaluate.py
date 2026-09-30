"""Tie-aware metrics and fixed-budget paired contrasts."""
import numpy as np


class CachedMetric:
    """Tie-aware weighted AUROC/AP with immutable score ordering."""
    def __init__(self, y, score):
        y,score=np.asarray(y),np.asarray(score)
        if y.ndim!=1 or score.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(score).all():
            raise ValueError('Invalid metric inputs')
        self.order=np.argsort(-score,kind='stable')
        self.y=y[self.order]
        self.starts=np.r_[0,np.flatnonzero(np.diff(score[self.order]))+1]

    def __call__(self, weight):
        w=np.asarray(weight,dtype=float)
        if w.shape!=self.y.shape or not np.isfinite(w).all() or (w<0).any():
            raise ValueError('Invalid patient multiplicities')
        w=w[self.order]
        pos=np.add.reduceat(w*self.y,self.starts)
        neg=np.add.reduceat(w*(1-self.y),self.starts)
        cp,cn=np.cumsum(pos),np.cumsum(neg)
        if cp[-1]<=0 or cn[-1]<=0: return np.array([np.nan,np.nan])
        auc=np.sum(pos*(cn[-1]-cn+.5*neg))/(cp[-1]*cn[-1])
        precision=np.divide(cp,cp+cn,out=np.zeros_like(cp),where=cp+cn>0)
        ap=np.sum(pos*precision)/cp[-1]
        return np.array([auc,ap])


def contrasts(metrics):
    # metrics maps group -> (AUROC, AP), with values already averaged over fits.
    out={}
    for arm in ('random','prevalence_matched'):
        prefix='3072__'
        exp_f=metrics[prefix+'explicit__fusion']; exp_i=metrics[prefix+'explicit__image']
        f=metrics[prefix+arm+'__fusion']; i=metrics[prefix+arm+'__image']
        out[arm+'_minus_explicit__fusion']=f-exp_f
        out[arm+'_minus_explicit__image']=i-exp_i
        out[arm+'_minus_explicit__gain']=(f-i)-(exp_f-exp_i)
    for arm in ('explicit','random','prevalence_matched'):
        out[arm+'__gain']=metrics['3072__'+arm+'__fusion']-metrics['3072__'+arm+'__image']
    out['full_all_minus_full_explicit__fusion']=metrics['full__full_all__fusion']-metrics['full__full_explicit__fusion']
    out['full_all_minus_full_explicit__gain']=(metrics['full__full_all__fusion']-metrics['full__full_all__image'])-(metrics['full__full_explicit__fusion']-metrics['full__full_explicit__image'])
    return out
