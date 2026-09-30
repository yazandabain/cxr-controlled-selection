"""Patient multiplicities for paired resampling."""
import numpy as np


def shared_training_weights(left_ids,right_ids,rng):
    left_ids,right_ids=np.asarray(left_ids),np.asarray(right_ids)
    if len(np.unique(left_ids))!=len(left_ids) or len(np.unique(right_ids))!=len(right_ids):
        raise ValueError('Training panels must contain unique patients')
    union=np.union1d(left_ids,right_ids)
    counts=np.bincount(rng.integers(len(union),size=len(union)),minlength=len(union))
    return counts[np.searchsorted(union,left_ids)],counts[np.searchsorted(union,right_ids)]


def patient_weights(ids,rng):
    unique,inv=np.unique(ids,return_inverse=True)
    return np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inv]
