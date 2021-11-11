import numpy as np

def e_exp(parameters, *data):
    x,y=data
    b=parameters
    return ((np.exp(-b*x) - y)**2).mean()

def e_cumulative(parameters, *data):
    x,y=data
    t=parameters
    return (((x<=t)*1 - y)**2).mean()

def e_cumulative_linear(parameters, *data):
    x,y=data
    t=parameters
    return (((x<=t)*(1-x/t) - y)**2).mean()

def e_cumulative_gauss(parameters, *data):
    x,y=data
    t,v=parameters
    if t >= 0:
        return (((x<=t)*1 + (x>t)*np.exp(-(x-t)/v)-y)**2).mean()
    else:
        return np.inf

def e_mod_gauss(parameters, *data):
    x,y=data
    b=parameters
    return ((np.exp(-x**2/b)-y)**2).mean()
    
def e_soft_threshold(parameters, *data):
    x,y=data
    c,t=parameters
    
    if t > 0:
        return ((1/(1+np.exp(c*(x-t)/t)) * (1+np.exp(-c)) - y)**2).mean()
    else:
        return np.inf
    
def e_inv_pow(parameters, *data):
    x,y=data
    b=parameters
    x = x + (x==0)*1e-8
    return (((x<=1)*1 + (x>1)*(x**-b) - y)**2).mean()

def exp(x,b):
    return np.exp(-b*x)

def cumulative(x,t):
    return (x<=t)*1

def cumulative_linear(x,t):
    return (x<=t)*(1-x/t)

def cumulative_gauss(x,t,v):
    if t >= 0:
        return (x<=t)*1 + (x>t)*np.exp(-(x-t)/v)
    else:
        return -np.inf

def mod_gauss(x,b):
    return np.exp(-x**2/b)

def soft_threshold(x,c,t):
    if t > 0:
        return 1/(1+np.exp(c*(x-t)/t)) * (1+np.exp(-c))
    else:
        return -np.inf

def inv_pow(x,b):
    x = x + (x==0)*1e-8
    return (x<=1)*1 + (x>1)*(x**-b)

ALL_ACCESSIBILITY_FUNCS = {
        'Inverse Exponential' : (e_exp, exp, {'bounds': [(0,5)]}, ['beta']),
        'Cumulative Rectangular' : (e_cumulative,cumulative, {'bounds': [(0,1200)]},['t']),
        'Cumulative Linear' : (e_cumulative_linear,cumulative_linear, {'bounds': [(0,1200)]},['t']),
        'Cumulative Gaussian' : (e_cumulative_gauss,cumulative_gauss, {'bounds': [(0,200),(1,200)]},['t','v']),
        'Modified Gaussian' : (e_mod_gauss,mod_gauss,{'bounds':[(1e-3,5e4)]},['beta']),
        'Soft Threshold' : (e_soft_threshold,soft_threshold,{'bounds':[(0,10),(1,1200)]},['c','t']),
        'Inverse Power' : (e_inv_pow,inv_pow,{'bounds':[(1e-3,10)]},['beta'])
    }