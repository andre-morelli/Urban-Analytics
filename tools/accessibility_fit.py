import numpy as np
from .curve_funcs import *
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.signal import savgol_filter
from math import ceil

def fit_impedance(d, func,bounds,target_func=None,crop_outliers=True,
                          outlier_threshold=3,p0=None,show=True,
                          plot_result=True,get_stats=True,plot_kws={},
                          plot_data = True,pop_size=256,smooth=True,
                          bins=None,bin_size=5):
    
    assert {((plot_result and target_func!=None) or (not plot_result)), 
           'Please provide a target function if plot_result=True'}
    assert {((get_stats and target_func!=None) or (not get_stats)), 
           'Please provide a target function if get_stats=True'}
    d = np.array(d)
    if crop_outliers:
        p75 = np.percentile(d,75)
        p25 = np.percentile(d,25)
        w = p75 + (p75-p25)*outlier_threshold
        d = np.delete(d,np.where(d>w))
    if bins==None:
        bins = ceil(max(d)/bin_size)
    v,b = np.histogram(d,bins=bins,range=(0,d.max()))
    v = v + (v==0)*1e-8 #avoid null values
    b = b[1:]
    if smooth:
        v = savgol_filter(v, 9, 3) # window size 9, polynomial order 3
    v = v/(max(v))
    mid = (b[1]-b[0])/2
    b = b-mid
    
    poly = differential_evolution(func, bounds=bounds,args=(b,v),
                                 popsize=pop_size)
    poly,lsq = poly['x'],poly['fun']
    poly = list(poly)
    if plot_result:
        a = np.linspace(0,max(b),num=1000)
        y = target_func(a,*poly)
        plt.plot(a,y,**plot_kws)
        if plot_data:
            plt.bar(b,v,width=mid*1.8,alpha=.3,color='orangered')
        if show:
            plt.show()
    #goodness of fit
    if get_stats:
        y = target_func(b,*poly)
        s,p = sts.wilcoxon(v,y)
        return poly,lsq,s,p
    else:
        return poly,lsq
    
def get_best_fit(d, boxcox=True,crop_outliers=True,
                 outlier_threshold=1.5,bins=None,
                 plot_result=True,pop_size=256,
                 smooth=True,bin_size=5):
    plt.figure(figsize=(9,4))
    funcs = ALL_ACCESSIBILITY_FUNCS
    
    best = (None, None, np.inf)
    plt_d = True
    for name,func_data in funcs.items():
        func,target_func,kws = func_data
        poly,s = fit_impedance(d,func=func,target_func=target_func,
                                     bins=bins,crop_outliers=crop_outliers,
                                     outlier_threshold=outlier_threshold,
                                     plot_result=plot_result,show=False,
                                     plot_kws={'label':name,'linewidth':2.5},
                                     plot_data=plt_d,get_stats=False,smooth=smooth,
                                     **kws)
        plt_d = False
        if best[-1]>s:
            best = (name,poly,s)
    if plot_result:
        plt.legend()
        plt.ylim(0,1.1)
        _,xmax = plt.xlim()
        if xmax>120:
            plt.xlim(0,120)
        plt.show()
    ##***return all the functions here
    return best