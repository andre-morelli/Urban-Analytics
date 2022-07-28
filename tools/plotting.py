import mapclassify
import numpy as np
from numba import jit
import warnings
@jit
def _kclasses(arr,k=5):
    kclasses=mapclassify.FisherJenks(arr,k=k)
    kclasses=kclasses.make()(arr)
    return kclasses

def linewidths_by_attribute_fisherjenks(gdf,attr,k=5,vmin=.5,vmax=6):
    arr = np.array(gdf[attr])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kclasses = _kclasses(arr,k=k)
    s = sorted(set(kclasses))
    sizes = [(vmin+n*((vmax-vmin)/(k-1))) for n in s]
    d = {n:m for n,m in zip(s,sizes)}
    return [d[n] for n in kclasses]