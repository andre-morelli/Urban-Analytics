import random
import numpy as np
from .utils import get_igraph, get_full_igraph
import osmnx as ox
import networkx as nx
import geopandas as gpd
import math
from tqdm.notebook import tqdm
import warnings

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

def random_points_in_polygon(polygon,n_pts,seed=None):
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
    return np.array(X),np.array(Y)

def calc_accessibility(zones, G, opportunities_column=None,
                        pois=None, pois_weight_column=None,
                        weight='length', func=acc_cumulative_gaussian,
                        func_kws={},k=5, random_seed=None, 
                        array_cap=1_000, round_trip = False,
                        competition = False,population_column=None,
                        competition_kws=None,competition_graph=None,
                        node_subset=None):
    """
    Calculate accessibility by zone using given accessibility function.
    
    Parameters
    ----------
    zones : GeoDataframe
        Area GeoDataFrame containing census zone information
    G : NetworkX graph structure
        Network Graph.
    opportunities_column : string
        Column in the "zones" GeoDataFrame with opportunities.
    pois : GeoDataFrame
        Point GeoDataFrame containing points of interest (opportunities). Ignored if
        "opportunities_column" is set.
    pois_weight_column : string
        Column in the pois GeoDataFrame with point of interest weights.
    weight : string
        Graph´s weight attribute for shortest paths (such as length or travel time)
    func : function
        Access score function to use. Options are: acc_cumulative, 
        acc_soft_threshold, and acc_cumulative_gaussian
    func_kws : dictionary
        arguments for the access score function
    k : int
        number of sampled points per zone
    random_seed : int
        random seed.
    array_cap : int
        set a cap for the size of array to uso for computations of costs. Larger sizes may give
        out MemoryError
    round_trip : boolean
        if True, the cost of a trip is considered to be the average of the two legs (inbound + outbound)/2.
        This may be interesting for cycling network analysis where topography can have very different impact
        on the two legs of the journey, creating distortions when one leg is made primarilly on downhill movement
        disconsidering that the uphill leg will have to be executed uppon comming back from work.
    competition : boolean
        if True, calculate betweenness considering competion
    population_column : string
        Column in the "zones" GeoDataFrame with area weights (such as population). Only necessary if competition
        is True
    competition_kws : dictionary
        arguments for the distance decay function of the competition. This allows for the competition
        to have a different decay function than the mode evaluated. If not provided, is set equal to
        "func_kws"
    competition_graph : NetworkX graph structure
        If provided, the competition will be calculated on this graph, otherwise, it will be calculated
        on the same graph as the mode analysed "G"
    node_subset : array like
        Subset of nodes to consider as Origins/Destinations. This is important when transit
        networks are provided to force Origins and Destinations to start in the pedestrian 
        network. If None use all.
    
    Returns
    -------
    Dictionary {zone index: average accessibility score}
    """
    if node_subset is None:
        node_subset=G.nodes
    assert 0<k and type(k)==int, '"k" must be a positive integer'
    assert pois is not None or opportunities_column is not None, 'must provide "pois" or "opportunities_column"'
    if competition:
        assert population_column is not None, 'needs a population_column for copetition analysis'
    #get nodes for people and opportunities if applicable
    if competition_graph is None:
        competition_graph=G
    if competition_kws is None:
        competition_kws=func_kws
    Gig = get_full_igraph(G)
    if competition:
        Gigc = get_full_igraph(competition_graph)
    else:
        Gigc = Gig
    #create a dictionary for cross-references
    node_dict = {}
    for node in Gig.vs:
        try:
            node_dict[node['osmid']] = node
        except:
            node_dict[node['osmid']] = node
    
    X,Y = [],[]
    for zone in zones.iterrows():
        zone = zone[1]
        poly = zone['geometry']
        # get k points within the polygon
        X_,Y_ = random_points_in_polygon(poly,k,seed=random_seed)
        #match points to graph
        X+=list(X_)
        Y+=list(Y_)
    X = np.array(X)
    Y = np.array(Y)
    zone_ns = ox.get_nearest_nodes(nx.subgraph(G,node_subset),X,Y,method='balltree')
    ig_nodes = [node_dict[n] for n in zone_ns]
    
    # if analysis is disagregated
    if pois is not None:
        # get places on the gdf
        X = np.array([n.coords[0][0] for n in pois['geometry']])
        Y = np.array([n.coords[0][1] for n in pois['geometry']])
        #set places to nodes
        nodes = ox.get_nearest_nodes(nx.subgraph(G,node_subset),X,Y,method='balltree')
        attrs = {}.fromkeys(G.nodes,0)
        if pois_weight_column is None:
            pois_weight_column = 'temp'
            pois = pois.copy()
            pois[pois_weight_column] = 1
        for node, val in zip(nodes,pois[pois_weight_column]):
            attrs[node] += val
        nx.set_node_attributes(G,attrs,pois_weight_column)    
        
        #get nodes to target (for faster shortest paths)
        n_targets = [n for n in G.nodes if G.nodes[n][pois_weight_column]>0]
        nig_targets = [node_dict[n] for n in n_targets]
        vals = [G.nodes[n][pois_weight_column] for n in n_targets]
        
        if competition:
            pop_targs = []
            pops_of_targs = []
            #avoid duplicates
            for n,v in enumerate(zones[population_column]):
                nns = list(set(ig_nodes[n*k:(n+1)*k]))
                nns = [nn for nn in nns if nn not in pop_targs]
                if len(nns)==0:
                    nns = list(set(ig_nodes[n*k:(n+1)*k]))
                    for nn in nns:
                        i=np.where(np.array(pop_targs)==nn)[0][0]
                        pops_of_targs[i]+=v/k
                else:
                    pop_targs+=nns
                    pops_of_targs+=[v/len(nns)]*len(nns)

            pop_acc = []
            for target_id,target in enumerate(nig_targets):
                total_acc=0
                dists = Gigc.shortest_paths_dijkstra(source=target, target=pop_targs, 
                                                    weights=weight,mode='out')
                for ds in dists:
                    new = np.array(pops_of_targs)*func(np.array(ds), **competition_kws)
                    total_acc += np.nansum(new)
                pop_acc.append(max(1,total_acc))
            vals = [v/pacc for v,pacc in zip(vals,pop_acc)]
        else:
            pop_acc=[1]*len(nig_targets)
    else:
        if competition:
            #do calculation recursevely
            opp_acc = calc_zone_accessibility(zones, competition_graph, opportunities_column=population_column,
                                              weight=weight, func=func,
                                              func_kws=competition_kws,k=k, random_seed=random_seed, 
                                              array_cap=array_cap, round_trip = round_trip,
                                              competition = False)
            for key,val in opp_acc.items():
                if val<1: opp_acc[key]=1
        else:
            opp_acc = {zid:1 for zid in zones.index}
    if opportunities_column is not None:
        vals = []
        nig_targets = []
        #avoid duplicates
        for n,(op,acc) in enumerate(zip(zones[opportunities_column],opp_acc.values())):
            nns = list(set(ig_nodes[n*k:(n+1)*k]))
            nns = [nn for nn in nns if nn not in nig_targets]
            if len(nns)==0:
                nns = list(set(ig_nodes[n*k:(n+1)*k]))
                for nn in nns:
                    i=np.where(np.array(nig_targets)==nn)[0][0]
                    vals[i]+=op/k
            else:
                nig_targets+=nns
                vals+=[op/len(nns)/acc]*len(nns)
    #initiate total accessibility as zero 
    #calc distances to nodes
    acc=[]
    loop = [l[1] for l in zones.iterrows()]
    sects = [ig_nodes[x:x+array_cap*k] for x in range(0,int((len(ig_nodes)//(array_cap*k)+1)*(array_cap*k))+1,array_cap*k)]
    loops = [loop[x:x+array_cap] for x in range(0,int((len(loop)//(array_cap)+1)*array_cap)+1,array_cap)]
    for section,l in zip(sects,loops):
        if round_trip:
            distances = np.array(Gig.shortest_paths_dijkstra(source=section, target=nig_targets, 
                                                             weights=weight,mode='in'))
            distances = distances + np.array(Gig.shortest_paths_dijkstra(source=section, target=nig_targets, 
                                                                         weights=weight,mode='out'))
            #distance will be an average of the two
            distances = distances/2
        else:
            distances = Gig.shortest_paths_dijkstra(source=section, target=nig_targets, weights=weight,mode='out')
        for n,zone in enumerate(l):
            total_acc=0
            for ds in distances[n*k:n*k+k]:
                new = np.array(vals)*func(np.array(ds), **func_kws)
                total_acc += np.nansum(new)
            acc.append(total_acc/k)
    
    return {i:a for i,a in zip(zones.index,acc)}
    
def plot_element_hists(hists, elements, label='',plt_kws={'bins':20}):
    h=[]
    for el in elements:
        h+=hists[el]
    tot = ''.join(reversed(str(len(h))))
    
    tot = [''.join(reversed(tot[n*3:(n+1)*3])) for n in range(math.ceil(len(tot)/3))]
    tot = ' '.join(reversed(tot))
    plt.hist(h,label=f'{label} [{tot}]',**plt_kws)
    return None

def edge_statistics(G,path_mat,filter_areas=None,
                   name='load', inplace=True,
                   return_cost_hist=False):
    if not inplace:
        G = G.copy()
    loads = {}.fromkeys(G.edges,0)
    tot_time = {}.fromkeys(G.edges,0)
    tot_routes = {}.fromkeys(G.edges,0)
    if return_cost_hist:
        cost_hist = {}.fromkeys(G.edges)
        for e in cost_hist:
            cost_hist[e] = []
    for w,z,t,path,opp,pop,dist_decay in path_mat:
        if filter_areas is None or z in filter_areas:
            for e in path:
                loads[e] += w
                tot_time[e] += t
                tot_routes[e] += 1
                if return_cost_hist:
                    cost_hist[e].append(t)
    avg_time = {e:tt/tr for e,tt,tr in zip(tot_time,tot_time.values(),tot_routes.values()) if tr>0}
    nx.set_edge_attributes(G,loads,name)
    nx.set_edge_attributes(G,avg_time,name+'avgtime')
    if inplace: return (cost_hist if return_cost_hist else None)
    else: return (G,cost_hist if return_cost_hist else G)

def add_edge_loads(G,path_mat,filter_areas=None,
                   name='load', inplace=True,
                   return_cost_hist=False):
    if not inplace:
        G = G.copy()
    loads = {}.fromkeys(G.edges,0)
    tot_time = {}.fromkeys(G.edges,0)
    tot_routes = {}.fromkeys(G.edges,0)
    if return_cost_hist:
        cost_hist = {}.fromkeys(G.edges)
        for e in cost_hist:
            cost_hist[e] = []
    for w,z,t,path,opp,pop,dist_decay in path_mat:
        if filter_areas is None or z in filter_areas:
            for e in path:
                loads[e] += w
                tot_time[e] += t
                tot_routes[e] += 1
                if return_cost_hist:
                    cost_hist[e].append(t)
    avg_time = {e:tt/tr for e,tt,tr in zip(tot_time,tot_time.values(),tot_routes.values()) if tr>0}
    nx.set_edge_attributes(G,loads,name)
    nx.set_edge_attributes(G,avg_time,name+'avgtime')
    if inplace: return (cost_hist if return_cost_hist else None)
    else: return (G,cost_hist if return_cost_hist else G)

def add_node_loads(G,path_mat,filter_areas=None,name='load', inplace=True):
    if not inplace:
        G = G.copy()
    loads = {}.fromkeys(G.nodes,0)
    tot_time = {}.fromkeys(G.nodes,0)
    tot_routes = {}.fromkeys(G.nodes,0)
    if return_cost_hist:
        cost_hist = {}.fromkeys(G.nodes)
        for n in cost_hist:
            cost_hist[n] = []
    for w,z,t,path,opp,pop,dist_decay in path_mat:
        if filter_areas is None or z in filter_areas:
            for e in path:
                loads[e[1]] += w
                tot_time[e[1]] += t
                tot_routes[e[1]] += 1
                if return_cost_hist:
                    cost_hist[e[1]].append(t)
    avg_time = {e:tt/tr for e,tt,tr in zip(tot_time,tot_time.values(),tot_routes.values()) if tr>0}
    nx.set_edge_attributes(G,loads,name)
    nx.set_edge_attributes(G,avg_time,name+'avgtime')
    
    if inplace: return (cost_hist if return_cost_hist else None)
    else: return (G,cost_hist if return_cost_hist else G)

def _get_edges(e_seqs,Gig,weight='length'):
    r = []
    s = []
    for seq in e_seqs:
        if len(seq)<1:
            r.append([])
            s.append(0)
        else:
            r.append(Gig.es[seq]['orig_name'])
            s.append(sum(Gig.es[seq][weight]))
    return r,s

def _update_elements(el_mat,path_seq,weights,z1,time,opp,pop,dist_decay,w_bottom=1e-4):
    if weights is None: 
        weights=[1]*len(e_seq)
    else:
        weights=[n if n==n else 0 for n in weights]
    el_mat+=[(w,z,t,p,o,pp,dd) for w,z,t,p,o,pp,dd in zip(weights,z1,time,path_seq,opp,pop,dist_decay) if w>w_bottom] #this also filter out nans
    return el_mat

def betweenness_accessibility(zones, G, weight='length',func=acc_cumulative_gaussian, 
                              func_kws={},k=5,random_seed=None,
                              opportunities_column=None, node_subset=None,pois=None,
                              pois_weight_column=None, population_column=None,
                              track_progress=False, norm=False, write_name='load',
                              competition=False, competition_kws=None,
                              competition_graph=None):
    """
    Calculate Betweenness Accessibility through a distance decay function.
    Betweenness Accessibility is a centrality measure indicating how important
    an edge is to maintain the current level of accessibility on a network.
    Betweenness accessibility was first proposed by Sarlas et al. (2020)
    https://doi.org/10.1016/j.jtrangeo.2020.102680
    
    Parameters
    ----------
    zones : GeoDataframe
        Area GeoDataFrame containing census tract information
    G : NetworkX graph structure
        Network Graph.
    weight : string
        Graph´s weight attribute for shortest paths (such as length or travel time)
    func : function
        distance decay function to use.
    func_kws : dictionary
        arguments for the distance decay function
    k : int
        number of sampled points per zone
    random_seed : int
        random seed.
    opportunities_column : string
        Column in the "zones" GeoDataFrame with opportunities.
    node_subset : array like
        Subset of nodes to consider as Origins/Destinations. This is important when transit
        networks are provided to force Origins and Destinations to start in the pedestrian 
        network. If None use all.
    pois : GeoDataFrame
        Point GeoDataFrame containing points of interest (opportunities). Ignored if
        "opportunities_column" is set.
    pois_weight_column : string
        Column in the pois GeoDataFrame with point of interest weights.
    population_column : string
        Column in the "zones" GeoDataFrame with area weights (such as population).
    track_progress : boolean
        if True show progression bar **only available for running in jupyter-notebook**
    norm : boolean
        if True, normalize the results (maximum betweenness will be 1, minimum will be 0)
    write_name : string
        name of the attribute to write on the Graph
    competition : boolean
        if True, calculate betweenness considering competion
    competition_kws : dictionary
        arguments for the distance decay function of the competition. This allows for the competition
        to have a different decay function than the mode evaluated. If not provided, is set equal to
        "func_kws"
    competition_graph : NetworkX graph structure
        If provided, the competition will be calculated on this graph, otherwise, it will be calculated
        on the same graph as the mode analysed "G"
    
    Returns
    -------
    Numpy array containing trips and parameter associated. To build it on a graph, use "add_edge_nodes" or 
    "add_node_loads"
    """
    if node_subset is None:
        node_subset=G.nodes
    if competition_graph is None:
        competition_graph=G
    if competition_kws is None:
        competition_kws=func_kws
    attr='load'
    assert 0<k and type(k)==int, '"k" must be a positive integer'
    
    for e in G.edges:
        G.edges[e]['orig_name'] = e
    Gig = get_full_igraph(G)
    Gigc = get_full_igraph(competition_graph)
    if pois_weight_column is None and pois is not None:
        pois_weight_column = 'temp'
        pois = pois.copy()
        pois[pois_weight_column] = 1
    if population_column is None:
        tratcs_weight_column = 'temp'
        zones = zones.copy()
        zones[population_column] = 1
    
    node_dict_r = {}
    for node in Gig.vs:
        node_dict_r[node.index] = node['osmid']
    node_dict = {}
    for node in Gig.vs:
        node_dict[node['osmid']] = node.index
    if pois is not None:
        for e_ in G.edges:
            G.edges[e_][attr]=0
        
        # get places on the gdf
        X = np.array([n.coords[0][0] for n in pois['geometry']])
        Y = np.array([n.coords[0][1] for n in pois['geometry']])
        #set places to nodes
        nodes = ox.get_nearest_nodes(nx.subgraph(G,node_subset),X,Y,method='balltree')
        attrs = {}.fromkeys(G.nodes,0)
        for node, val in zip(nodes,pois[pois_weight_column]):
            attrs[node] += val
        nx.set_node_attributes(G,attrs,pois_weight_column)    
        # get igraph object for fast computations
        Gig = get_full_igraph(G)
        
        #get nodes to target (for faster shortest paths)
        n_targets = [n for n in G.nodes if G.nodes[n][pois_weight_column]>0]
        nig_targets = [node_dict[n] for n in n_targets]
        t_w = np.array([G.nodes[n][pois_weight_column] for n in n_targets])
        vals = [G.nodes[n][pois_weight_column] for n in n_targets]
    
    X,Y = [],[]
    pops = []
    t_id = []
    for i,tract in zones.iterrows():
        t_id+=[i]*k
        poly = tract['geometry']
        # get k points within the polygon
        X_,Y_ = random_points_in_polygon(poly,k,seed=random_seed)
        #match points to graph
        X+=list(X_)
        Y+=list(Y_)
        pops += [tract[population_column]/k]*k
    pops = np.array(pops)
    tot = np.nansum(pops)
    X = np.array(X)
    Y = np.array(Y)
    tract_ns = ox.get_nearest_nodes(nx.subgraph(G,node_subset),X,Y,method='balltree')
    ig_nodes = [node_dict[n] for n in tract_ns]
    
    if competition or pois is None:
        pop_targs = []
        pops_of_targs = []
        if pois is None:
            vals = []
        for n in range(len(zones)):
            nns = list(set(ig_nodes[n*k:(n+1)*k]))
            nns = [nn for nn in nns if nn not in pop_targs]
            if len(nns)==0: 
                nns = list(set(ig_nodes[n*k:(n+1)*k]))
                for nn in nns:
                    i=np.where(np.array(pop_targs)==nn)[0][0]
                    pops_of_targs[i]+=pops[n*k]/k
                    if pois is None:
                        v=list(zones[opportunities_column])[n]
                        vals[i]+=v/k
            else:
                pop_targs+=nns
                pops_of_targs+=[pops[n*k]*k/len(nns)]*len(nns)
                if pois is None:
                    v=list(zones[opportunities_column])[n]
                    vals+=[v/len(nns)]*len(nns)
        if pois is None:
            t_w = np.array(vals)
            nig_targets = pop_targs
        if competition:
            pop_acc = []
            for target_id,target in enumerate(nig_targets):
                total_acc=0
                dists = Gigc.shortest_paths_dijkstra(source=target, target=pop_targs, 
                                                    weights=weight,mode='out')
                for ds in dists:
                    new = np.array(pops_of_targs)*func(np.array(ds), **competition_kws)
                    total_acc += np.nansum(new)
                pop_acc.append(max(1,total_acc))
            pop_acc = np.array(pop_acc)
        else:
            pop_acc=np.array([1]*len(nig_targets))
    elif not competition:
        pop_acc = np.array([1]*len(nig_targets))
    #use tqdm if track_progress    
    loop = (tqdm(zip(t_id,ig_nodes,pops),total=len(ig_nodes),leave=False)
            if track_progress 
            else zip(t_id,ig_nodes,pops))
    
    edge_attrs = []
    for e in G.edges:
        for key in G.edges[e].keys():
            if key in edge_attrs: continue
            edge_attrs.append(key)
    node_attrs = []
    for n in G.nodes:
        for key in G.nodes[n].keys():
            if key in node_attrs: continue
            node_attrs.append(key)
    mat = []
    for s_id,source,pop in loop:
        if pop != pop or pop==0:
            continue
        e_seq = Gig.get_shortest_paths(source,nig_targets,weights=weight,output='epath')
        e_seq,ts = _get_edges(e_seq,Gig,weight=weight)
        dist_decay = func(np.array(ts),**func_kws)
        w = dist_decay * t_w * pop / pop_acc
        if norm:
            w = w/tot
        mat = _update_elements(mat,e_seq,weights=w,z1=[s_id]*len(w),time=ts,opp=t_w,pop=[pop]*len(ts),dist_decay=dist_decay)
    return mat