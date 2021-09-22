import random
import numpy as np
from .utils import get_igraph, get_full_igraph
import osmnx as ox
import networkx as nx
import geopandas as gpd

import math

#all array types
def acc_comulative(d, t=500):
    """
    Calculate cumulative accessibility for array of observations.

    Parameters
    ----------
    d : numpy array
        Array where every entry is a travel cost.
    t : float
        Cost cap (maximum cost such as walking distance or time spent on 
        trip).
    
    Returns
    -------
    Numpy array
    """
    return (d<=t)*1

def acc_soft_threshold(d, t=500, k=5):
    """
    Calculate soft threshold accessibility for array of observations.
    source: Higgs, C., Badland, H., Simons, K. et al. The Urban Liveability 
    Index: developing a policy-relevant urban liveability composite measure 
    and evaluating associations with transport mode choice. Int J Health 
    Geogr 18, 14 (2019). https://doi.org/10.1186/s12942-019-0178-8
    
    Parameters
    ----------
    d : numpy array
        Array where every entry is a travel cost.
    t : float
        function parameter (point at which access score = 0.5).
    k : float
        function parameter
    
    Returns
    -------
    Numpy array
    """
    return (1+np.exp(k*(d-t)/t))**-1

def acc_cumulative_gaussian(d, t=500, v=129_842):
    """
    Calculate soft threshold accessibility for array of observations.
    source: Vale DS, Pereira M. The influence of the impedance function on 
    gravity-based pedestrian accessibility measures: A comparative analysis. 
    Environment and Planning B: Urban Analytics and City Science. 
    2017;44(4):740-763. doi:10.1177/0265813516641685
    
    Parameters
    ----------
    d : numpy array
        Array where every entry is a travel cost.
    t : float
        Cost cap (size of flat region where access score = 1.0).
    v : float
        function parameter.
    
    Returns
    -------
    Numpy array
    """
    
    return (d<=t)*1 + np.exp(-(d-t)**2/v)*(d>t)

def random_points_in_polygon(n_pts, polygon,seed=None):
    """
    Gets random points within a shapely polygon
    
    Parameters
    ----------
    n_pts : int
        Nunber of points to generate.
    polygon : shapely Polygon
        Reference polygon.
    seed : int
        random seed.
    
    Returns
    -------
    tuple of coordinate arrays (X, Y)
    """
    
    np.random.seed(seed)
    x_min, y_min, x_max, y_max = polygon.bounds
    
    sft = polygon.area/((x_max-x_min)*(y_max-y_min))
    
    n = int(round(n_pts*(1+2/sft)))
    # generate random data within the bounds
    while True:
        x = np.random.uniform(x_min, x_max, n)
        y = np.random.uniform(y_min, y_max, n)

        # convert them to a points GeoSeries
        gdf_points = gpd.GeoSeries(gpd.points_from_xy(x, y))
        # only keep those points within polygons
        gdf_points = gdf_points[gdf_points.within(polygon)]

        X = [n.coords[0][0] for n in gdf_points.geometry][:n_pts]
        Y = [n.coords[0][1] for n in gdf_points.geometry][:n_pts]
        if len(X)<n_pts:
            continue
        break
    return X,Y

def calc_tract_accessibility(tracts, pois, G, weight='length',
                             func=acc_cumulative_gaussian,k=5, 
                             random_seed=None, func_kws={},
                             pois_weight_column=None,iter_cap=1_000):
    """
    Calculate accessibility by census tract using given accessibility function.
    
    Parameters
    ----------
    tracts : GeoDataframe
        Area GeoDataFrame containing census tract information
    pois : GeoDataFrame
        Point GeoDataFrame containing points of interest
    G : NetworkX graph structure
        Network Graph.
    weight : string
        Graph´s weight attribute for shortest paths (such as length or travel time)
    func : function
        Access score function to use. Options are: acc_cumulative, 
        acc_soft_threshold, and acc_cumulative_gaussian
    func_kws : dictionary
        arguments for the access score function
    k : int
        number of sampled points per tract
    pois_weight_column : string
        Column in the pois GeoDataFrame with location weights.
    random_seed : int
        random seed.
    iter_cap : int
        Parameter to limit memory usage. If the code raises memory error, lowering this 
        parameter might help.
    
    Returns
    -------
    Dictionary in the form {tract index: average accessibility score}
    """
    # get places on the gdf
    X = np.array([n.coords[0][0] for n in pois['geometry']])
    Y = np.array([n.coords[0][1] for n in pois['geometry']])
    #set places to nodes
    nodes = ox.get_nearest_nodes(G,X,Y, method='balltree')
    attrs = {}.fromkeys(G.nodes,0)
    if pois_weight_column is None:
        pois_weight_column = 'temp'
        pois = pois.copy()
        pois[pois_weight_column] = 1
    for node, val in zip(nodes,pois[pois_weight_column]):
        attrs[node] += val
    nx.set_node_attributes(G,attrs,pois_weight_column)    
    # get igraph object for fast computations
    Gig = get_full_igraph(G)
    #create a dictionary for cross-references
    node_dict = {}
    for node in Gig.vs:
        node_dict[int(node['osmid'])] = node
    
    #get nodes to target (for faster shortest paths)
    n_targets = [n for n in G.nodes if G.nodes[n][pois_weight_column]>0]
    nig_targets = [node_dict[n] for n in n_targets]
    vals = [G.nodes[n][pois_weight_column] for n in n_targets]
    
    loop = tracts.iterrows()
    X,Y = [],[]
    for tract in tracts.iterrows():
        tract = tract[1]
        poly = tract['geometry']
        # get k points within the polygon
        X_,Y_ = random_points_in_polygon(k,poly,seed=random_seed)
        #match points to graph
        X+=X_
        Y+=Y_
    ###here
    X = np.array(X)
    Y = np.array(Y)
    trackt_ns = ox.get_nearest_nodes(G,X,Y,method='balltree')
    ig_nodes = [node_dict[n] for n in trackt_ns]
    
    #initiate total accessibility as zero 
    #calc distances to nodes
    acc=[]
    
    if len(ig_nodes)>=iter_cap*k:
        loop = list(tracts.iterrows())
        loop = [_[1] for _ in loop]
        sects = [ig_nodes[x:x+iter_cap*k] for x in range(0,int((len(ig_nodes)//(iter_cap*k)+1)*(iter_cap*k))+1,iter_cap*k)]
        loops = [loop[x:x+iter_cap] for x in range(0,int((len(loop)//(iter_cap)+1)*iter_cap)+1,iter_cap)]
#         print(len(loops),len(sects))
        for section,l in zip(sects,loops):
            distances = Gig.shortest_paths_dijkstra(source=section, target=nig_targets, weights=weight)
            n=0
            for tract in l:
                total_acc=0
                for ds in distances[n:n+k]:
                    new = np.array(vals)*func(np.array(ds), **func_kws)
                    total_acc += new.sum()
                acc.append(total_acc/k)
                n+=k
    else:
        distances = Gig.shortest_paths_dijkstra(source=ig_nodes, target=nig_targets, weights=weight)
        n=0
        for tract in loop:
            total_acc=0
            for ds in distances[n:n+k]:
                new = np.array(vals)*func(np.array(ds), **func_kws)
                total_acc += new.sum()
            acc.append(total_acc/k)
            n+=k
    return {i:a for i,a in zip(tracts.index,acc)}