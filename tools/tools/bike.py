# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:05:17 2020

@author: ANDRE BORGATO MORELLI
"""

import pickle
import osmnx as ox
import networkx as nx
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import random

def solve_for_speed(i, power, Ca=0.3, Cr=0.003):
    """
    Solves the polynomial for speed given a constant power level.

    Parameters
    ----------
    i: float 
        Slope of the edge in m/m.
    power: float
        Constant power level in Watts.
    Ca: float 
        Aerodynamical coefficient
    Cr: float
        Roller resistance coefficient
    
    Returns
    -------
    float
    """

    # polynomial: [Ca]*(s^3) + 0*(s^2) + [90*9.81*(i+Cr)]*(s) - power = 0
    poly  = [Ca, 0, 90*9.81*(i+Cr), -power]
    try:
        results = np.roots(poly)
    except np.linalg.LinAlgError:
        print(f'could not solve for i:{i}')
        results = [np.inf]
    real_results = []
    for item in results:
        if np.isreal(item):
            real_results.append(float(item))
    return max(real_results)

def impedance(length, i, power, Ca=0.3, Cr=0.0032, speed_ceil=40, 
              work_per_meter_floor=0):
    """
    Gets work as impedance for the edge from lenght, grade and the power equation coefficients.

    Parameters
    ----------
    length: float
        length of the edge
    i: float 
        Slope of the edge in m/m.
    power: float
        Constant power level in Watts.
    Ca: float 
        Aerodynamical coefficient
    Cr: float
        Roller resistance coefficient
    speed_ceil: float
        the maximum speed for the cyclist
    work_per_meter_floor: float
        minimum work on the edge.
    
    Returns
    -------
    tuple
    """
    #Solve for speed in this edge:
    speed = solve_for_speed(i, power, Ca=Ca, Cr=Cr)
    if speed > speed_ceil/3.6:
        speed=speed_ceil/40
        
    time = length/speed
    work = power*time
        
    #Apply restrictions 
    if work >= work_per_meter_floor*length:
        return (work, speed)
    else:
        return (work_per_meter_floor*length, speed)

def get_edge_impedance(G, power=300, Ca=0.3, Cr=0.0032, speed_ceil=40, work_per_meter_floor=0):
    """
    Alters the graph object adding work impedance for each of the graph's edges.

    Parameters
    ----------
    G: networkx graph
        graph of the network
    power: float
        Constant power level in Watts.
    Ca: float 
        Aerodynamical coefficient
    Cr: float
        Roller resistance coefficient
    speed_ceil: float
        the maximum speed for the cyclist
    work_per_meter_floor: float
        minimum work on the edge.
    
    Returns
    -------
    None
    """
    G=ox.add_edge_grades(G)
    for u, v, k, data in G.edges(keys=True, data=True):
        work,speed = impedance(data['length'], data['grade'], power=power, Ca=Ca, Cr=Cr, 
                               speed_ceil=speed_ceil, work_per_meter_floor=work_per_meter_floor)
        work_flat, speed_flat = impedance(data['length'], 0, power=power, Ca=Ca, Cr=.0032, 
                                        speed_ceil=speed_ceil, work_per_meter_floor=work_per_meter_floor)
        data['time'] = data['length']/speed
        data['work'] = work
        data['work_flat'] = work_flat
        data['time_flat'] = data['length']/speed_flat
        data['work per meter'] = work/data['length']


def get_accessibility(G, work_cutoff, p=1, seed=42):
    """
    Compares metrics of the graph to an idealized flat terrain with smooth alphalt in all edges.

    Parameters
    ----------
    G: networkx graph
        graph of the network with impedances calculated
    work_cutoff: float
        Maximum work a user can do in a one-way trip.
    p: float
        proportion of nodes to be used in the analysis (must be between 0 and 1)
    seed: int
        random seed
    
    Returns
    -------
    tuple
    """
    work_index = {}
    speed_index = {}
    sp1,sp2 = {},{}
    random.seed(42)
    Gig = _get_full_igraph(G)
    for v in Gig.vs:
        if random.random() > p:
            continue
        work_index[int(v['osmid'])] = []
        sp1[int(v['osmid'])] = []
        sp2[int(v['osmid'])] = []
        lens = list(enumerate(Gig.shortest_paths_dijkstra(v, weights='work_flat')[0]))
        lens = [n for n,l in lens if l<=work_cutoff]
        paths = Gig.get_shortest_paths(v=v, to=lens, weights='work', output="epath")
        for path in paths:
            try:
                work_index[int(v['osmid'])].append((sum(Gig.es[path]['work_flat'])/sum(Gig.es[path]['work'])))
                sp1[int(v['osmid'])].append(sum(Gig.es[path]['time']))
                sp2[int(v['osmid'])].append(sum(Gig.es[path]['time_flat']))
            except:
                continue
    for node in work_index:
        if len(work_index[node])>0:
            work_index[(node)] = np.mean(work_index[node])
            speed_index[node] = np.sum(sp2[node])/np.sum(sp1[node])
        else:
            work_index[node] = np.nan
            speed_index[node] = np.nan
    return work_index, speed_index

def get_importance(G, work_cutoff):
    G = nx.DiGraph(G)
    G = nx.MultiDiGraph(G)

    paths = _fast_betweenness(G, cutoff=work_cutoff, norm=False, weight='work_flat')
    try:
        paths = {key:round(val) for key,val in paths.items()}
    except:
        for key,val in paths.items():
            if val!=None and val==val and (-1e50<val<1e50):
                paths[key] = val
            else:
                paths[key] = 0

    nx.set_edge_attributes(G, paths, 'importance_flat')

    paths = _fast_betweenness(G, cutoff=work_cutoff, norm=False, weight='work')
    try:
        paths = {key:round(val) for key,val in paths.items()}
    except:
        for key,val in paths.items():
            if val!=None and val==val and (-1e50<val<1e50):
                paths[key] = val
            else:
                paths[key] = 0
    nx.set_edge_attributes(G, paths, 'importance')
    return G

def get_overall_importance(G):
    i = [n['importance']*n['length'] for _,_,n in G.edges(data=True)]
    i0 = [n['importance_flat']*n['length'] for _,_,n in G.edges(data=True)]

    return sum(i)/sum(i0)


def _get_igraph(G, edge_weights=None, node_weights=None):
    """
    Transforms a NetworkX graph into an iGraph graph.

    Parameters
    ----------
    G : NetworkX DiGraph or Graph 
        The graph to be converted.
    edge_weights: list or string
        weights stored in edges in the original graph to be kept in new graph. 
        If None, no weight will be carried. See get_full_igraph to get all 
        weights and attributes into the graph.
    node_weights: list or string
        weights stored in nodes in the original graph to be kept in new graph. 
        If None, no weight will be carried. See get_full_igraph to get all 
        weights and attributes into the graph.

    Returns
    -------
    iGraph graph
    """
    if type(edge_weights) == str:
        edge_weights = [edge_weights]
    if type(node_weights) == str:
        node_weights = [node_weights]
    G = G.copy()
    G = nx.relabel.convert_node_labels_to_integers(G)
    Gig = ig.Graph(directed=True)
    Gig.add_vertices(list(G.nodes()))
    Gig.add_edges(list(G.edges()))
    if 'kind' not in G.graph.keys():
        G.graph['kind']=primal # if not specified, assume graph id primal
    if G.graph['kind']=='primal':
        Gig.vs['osmid'] = list(nx.get_node_attributes(G, 'osmid').values())
    elif G.graph['kind']=='dual':
        Gig.vs['osmid'] = list(G.edges)
    if edge_weights != None:
        for weight in edge_weights:
            Gig.es[weight] = [n for _,_,n in G.edges(data=weight)]
            
    if node_weights != None:
        for weight in node_weights:
            Gig.vs[weight] = [n for _,n in G.nodes(data=weight)]
    for v in Gig.vs:
        v['name'] = v['osmid']
    return Gig


def _get_full_igraph(G):
    """
    Transforms a NetworkX graph into an iGraph graph keeping all possible info.

    Parameters
    ----------
    G : NetworkX DiGraph or Graph 
        The graph to be converted.
    
    Returns
    -------
    iGraph graph
    """
    all_edge_attrs = []
    all_node_attrs = []
    for edge in G.edges:
        for attr in G.edges[edge].keys():
            if attr in all_edge_attrs:
                continue
            all_edge_attrs.append(attr)
                
    for node in G.nodes:
        G.nodes[node]['osmid'] = str(G.nodes[node]['osmid'])
        for attr in G.nodes[node].keys():
            if attr in all_node_attrs:
                continue
            all_node_attrs.append(attr)       
    
    return _get_igraph(G, all_edge_attrs, all_node_attrs)

def _fast_betweenness(G, weight=None, kind = 'edge', norm=True, cutoff=None):
    """
    Gets betweenness centrality. For relativelly large graphs, this func is 
    faster than networkx

    Parameters
    ----------
    G : NetworkX DiGraph or Graph 
        The graph to be considered.
    weight: string
        edge weights for shortest paths.
    kind: 'edge' or 'node'
        Betweenness for edges or nodes.
    norm: bool
        If True, returns norm betweenness (bet/((N-1)*(N-2))).

    Returns
    -------
    dict
    """
    if weight != None:
        Gig = _get_igraph(G, edge_weights = weight)
    else:
        Gig = _get_igraph(G)
    norm_val = len(G.nodes)*(len(G.nodes)-1)
    if kind=='edge':
        bet = Gig.edge_betweenness(weights=weight, cutoff=cutoff)
        if norm==True:
            return {e:b/norm_val for e,b in zip(G.edges,bet)}
        else:
            return {e:b for e,b in zip(G.edges,bet)}
    elif kind=='node':
        bet = Gig.betweenness(weights=weight, cutoff=cutoff)
        if norm==True:
            return {e:b/norm_val for e,b in zip(G.nodes,bet)}
        else:
            return {e:b for e,b in zip(G.nodes,bet)}