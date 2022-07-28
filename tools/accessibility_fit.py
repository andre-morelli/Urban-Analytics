from .curve_funcs import *
from .accessibility import *
from .utils import *

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, curve_fit
from scipy.signal import savgol_filter
import scipy.stats as sts
from math import ceil

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.argmax(ret[n - 1:])

def norm_func(x,func,func_kws={}):
    x_ = norm_x(x)
    y = func(x_,**func_kws)
    return x, y
def norm_x(x,p=85,k=100):
    factor = np.percentile(x,p)/k
    return x/factor,factor


def ecdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, 1 - cusum / cusum[-1]

def fit_impedance_cdf(d, func,crop_outliers=True,
                  outlier_threshold=3,show=True,
                  plot_result=True,plot_kws={},
                  plot_data = True,bounds=None,
                  p0=None,normx=True):
    d = np.array(d)
    if normx:
        d0 = d.copy()
        d,factor = norm_x(d,k=20)
    else:
        factor=1
    if crop_outliers:
        p75 = np.percentile(d,75)
        p25 = np.percentile(d,25)
        w = p75 + (p75-p25)*outlier_threshold
        d = np.delete(d,np.where(d>w))
    x,y = ecdf(d)
    poly,_ = curve_fit(func,x,y,p0=p0)
    if plot_data:
        plt.plot([0]+list(x*factor),[1]+list(y),':k',drawstyle='steps-post',zorder=10,label='Real Data',linewidth=2.5)
    msq = ((func(x,*poly)-y)**2).mean()
    if plot_result:
        a = np.linspace(0,max(x),5000)
        b = func(a,*poly)
        if 'label' in plot_kws:
            plot_kws['label'] += f'  mse: {msq:.03f}'
        plt.plot(a*factor,b,**plot_kws)
    if normx:
        a = np.linspace(0,max(x),num=5000)
        y = func(a,*poly)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            poly,_ = curve_fit(func,a*factor,y)
        poly = list(poly)
    if show:
        plt.show()
    return poly,msq
    
def get_best_fit_cdf(d, crop_outliers=True,
                 outlier_threshold=3,normx=True,
                 plot_result=True,show=True):
    plt.figure(figsize=(8,4))
    funcs = ALL_ACCESSIBILITY_FUNCS_CDF
    
    best = (None, None, np.inf)
    plt_d = True
    func_params = {}
    for name,func_data in funcs.items():
        func,target_func,kws,ps = func_data
        poly,s = fit_impedance_cdf(d,func=target_func,crop_outliers=crop_outliers,
                                   outlier_threshold=outlier_threshold,
                                   plot_result=plot_result,show=False,
                                   plot_kws={'label':name,'linewidth':2.5},
                                   plot_data=plt_d,normx=normx,**kws)
        plt_d = False
        if best[-1]>s:
            best = (name,poly,s)
        func_params[name] = {pname:pval for pname,pval in zip(ps,poly)}
        func_params[name]['mse'] = s
    if plot_result:
        plt.legend()
        plt.ylim(0,1.1)
        plt.xlim(0)
        if show:
            plt.show()
    return best,func_params


def fit_impedance_pdf(d, func,bounds,target_func=None,crop_outliers=True,
                  outlier_threshold=3,p0=None,show=True,
                  plot_result=True,get_stats=True,plot_kws={},
                  plot_data = True,pop_size=256,smooth=False,
                  bins=None,bin_size=5,fill_low=False,normx=True):
    assert {((plot_result and target_func!=None) or (not plot_result)), 
           'Please provide a target function if plot_result=True'}
    assert {((get_stats and target_func!=None) or (not get_stats)), 
           'Please provide a target function if get_stats=True'}
    d = np.array(d)
    if normx:
        d0 = d.copy()
        d,factor = norm_x(d)
    else:
        factor=1
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
    if fill_low:
        max_position = moving_average(v,n=3)
        v[:max_position+1] = max(v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        poly = differential_evolution(func, bounds=bounds,args=(b,v),
                                     popsize=pop_size)
    poly,lsq = poly['x'],poly['fun']
    poly = list(poly)
    if plot_result:
        a = np.linspace(0,max(b),num=1000)
        y = target_func(a,*poly)
        plt.plot(a*factor,y,**plot_kws)
        if plot_data:
            plt.bar(b*factor,v,width=mid*factor*1.8,alpha=.3,color='orangered')
        if show:
            plt.show()
    if normx:
        a = np.linspace(0,max(b),num=1000)
        y = target_func(a,*poly)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            poly,_ = curve_fit(target_func,a*factor,y)
        poly = list(poly)
    #goodness of fit
    if get_stats:
        y = target_func(b,*poly)
        s,p = sts.wilcoxon(v,y)
        return poly,lsq,s,p
    else:
        return poly,lsq
    
def get_best_fit_pdf(d, crop_outliers=True,
                 outlier_threshold=3,bins=None,
                 plot_result=True,pop_size=256,
                 smooth=False,bin_size=5,fill_low=False):
    plt.figure(figsize=(9,4))
    funcs = ALL_ACCESSIBILITY_FUNCS_PDF
    
    best = (None, None, np.inf)
    plt_d = True
    func_params = {}
    for name,func_data in funcs.items():
        func,target_func,kws,ps = func_data
        poly,s = fit_impedance_pdf(d,func=func,target_func=target_func,
                                     bins=bins,crop_outliers=crop_outliers,
                                     outlier_threshold=outlier_threshold,
                                     plot_result=plot_result,show=False,
                                     plot_kws={'label':name,'linewidth':2.5},
                                     plot_data=plt_d,get_stats=False,smooth=smooth,
                                     bin_size=bin_size,fill_low=fill_low,**kws)
        plt_d = False
        if best[-1]>s:
            best = (name,poly,s)
        func_params[name] = {pname:pval for pname,pval in zip(ps,poly)}
        func_params[name]['mse'] = s
    if plot_result:
        plt.legend()
        plt.ylim(0,1.1)
        plt.xlim(0,)
        plt.show
    return best,func_params
    
def get_cost_matrix(gdf,G,tripmatrix,zone_id='ID',
                       k=5,weight='length',seed=None,
                       round_trip=False):
    #tripmatrix necessary to match zone ids
    Gig = get_full_igraph(G)
    node_dict = {}
    for node in Gig.vs:
        node_dict[node['name']] = node.index
    
    gdf=gdf[gdf[zone_id].isin(tripmatrix.index)]
    #Turn everything to int to avoid problems
    try:
        tripmatrix = tripmatrix.set_index(np.int64(tripmatrix.index)).copy()
        tripmatrix.columns = [int(float(n)) for n in tripmatrix.columns]
        cntr = gdf.set_index(np.int64(gdf[zone_id])).copy()
        cntr = cntr.loc[np.int64(tripmatrix.index)]
    except:
        gdf=gdf.set_index(np.int64(gdf[zone_id])).copy()
        drop = [n for n in tripmatrix.index if n not in gdf[zone_id]]
        tripmatrix =tripmatrix.drop(drop,axis=0)
        tripmatrix =tripmatrix.drop(drop,axis=1)
        gdf=gdf[gdf[zone_id].isin(tripmatrix.index)]
        cntr = gdf.set_index(np.int64(gdf[zone_id])).copy()
        cntr = cntr.loc[np.int64(tripmatrix.index)]
        print(f'not all values from tripmatrix attached to zone, {len(cntr)} zones ok')

    Xs = []
    Ys = []
    for geom in cntr.geometry:
        X,Y= random_points_in_polygon(geom,k,seed=seed)
        Xs += list(X)
        Ys += list(Y)
    Xs,Ys = np.array([Xs,Ys])
    nodes = ox.get_nearest_nodes(G,Xs,Ys,method='balltree')
    g_nodes = np.array([node_dict[n] for n in nodes])
    g_nodes_cols = g_nodes.reshape(( int(g_nodes.shape[0]/k) , k))

    matching = list(set(g_nodes))#avoids same nodes in target
    dmat = np.zeros_like(tripmatrix,dtype=np.float64)
    dstd = np.zeros_like(tripmatrix,dtype=np.float64)
    for i,z1 in enumerate(g_nodes_cols):
        z1 = list(set(z1))
        
        dists = Gig.shortest_paths_dijkstra(z1,matching,weights=weight,
                                            mode='out')
        dists = np.array(dists)
        if round_trip:
            dists = dists + np.array(Gig.shortest_paths_dijkstra(z1,matching,
                                                                 weights=weight,
                                                                 mode='in'))
            dists = dists*.5
        
        dists = dists.sum(axis=0)/(dists!=0).sum(axis=0)
        
        d = {m:d for m,d in zip(matching,dists)}
        for j,z2 in enumerate(g_nodes_cols):
            a = [d[n] for n in z2]
            
            dmat[i][j] = np.mean(a)
            dstd[i][j] = np.std(a)
    return dmat,dstd,tripmatrix

def get_cost_counts(dmat,tripmat):
    assert dmat.shape == tripmat.shape, 'distance matrix must be same shape as trips matrix'
    dists = []
    for i in range(dmat.shape[0]):
        for j in range(dmat.shape[1]):
            dists = dists + [dmat[i][j]]*int(tripmat[i][j])
    return [n for n in dists if n!=np.inf and n==n]
    
def get_cost_counts_normal(dmat,dstd,tripmat,k=1):
    assert dmat.shape == tripmat.shape, 'distance matrix must be same shape as trips matrix'
    dists = []
    for i in range(dmat.shape[0]):
        for j in range(dmat.shape[1]):
            dists = dists + [n if n>=0 else 0 
                             for n in 
                             np.random.normal(loc=dmat[i][j], 
                                              scale=dstd[i][j], 
                                              size=int(tripmat[i][j]*k))]
    return [n for n in dists if n!=np.inf and n==n]