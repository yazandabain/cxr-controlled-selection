"""Class-matched report-state eligibility sampling."""
import numpy as np
ARMS = ("explicit_uncertain", "explicit_blank")


def sample_addback(ids,states,y,explicit,seed):
    ids,states,y,explicit=map(np.asarray,(ids,states,y,explicit))
    if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or states.shape!=ids.shape or y.shape!=ids.shape:
        raise ValueError('Exactly one row per patient is required')
    if not np.isin(y,[0,1]).all() or not np.isin(states,['explicit_positive','explicit_negative','uncertain','blank']).all():
        raise ValueError('Invalid labels or states')
    if len(explicit)!=len(np.unique(explicit)) or not np.isin(states[explicit],['explicit_positive','explicit_negative']).all():
        raise ValueError('Invalid explicit anchor membership')
    order=np.argsort(ids,kind='stable')
    priority=np.empty(len(ids));priority[order]=np.random.default_rng(seed).random(len(ids))
    result={}
    for arm,state in zip(ARMS,('uncertain','blank')):
        pool=np.isin(states,['explicit_positive','explicit_negative',state])
        parts=[]
        for label in (1,0):
            n=int((y[explicit]==label).sum())
            available=np.flatnonzero(pool&(y==label))
            if len(available)<n:raise ValueError('Insufficient coded-class support')
            parts.append(available[np.argsort(priority[available],kind='stable')[:n]])
        indices=np.concatenate(parts)
        assert len(indices)==len(explicit) and len(np.unique(indices))==len(explicit)
        assert int(y[indices].sum())==int(y[explicit].sum())
        result[arm]=indices
    return result
