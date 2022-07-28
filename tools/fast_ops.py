import igraph as ig
from .utils import get_igraph
#mudancinha
def fast_betweenness(G, weight=None, kind = 'edge', norm=True, cutoff=None):
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
        Gig = get_igraph(G, edge_weights = weight)
    else:
        Gig = get_igraph(G)
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
    
    
    
def fast_closeness(G, kind = 'edge', weight=None, norm=True):
    """
    Gets closeness centrality. For relativelly large graphs, this func is 
    faster than networkx

    Parameters
    ----------
    G : NetworkX DiGraph or Graph 
        The graph to be considered.
    weight: string
        edge weights for shortest paths.
    kind: 'edge' or 'node'
        closeness for edges or nodes. for edges, the closeness id the average
        of extreme nodes. 
    norm: bool
        If True, returns norm betweenness (clo*N).

    Returns
    -------
    dict
    """
    if weight != None:
        Gig = get_igraph(G, edge_weights = weight)
    else:
        Gig = get_igraph(G)
    
    clo = Gig.closeness(weights=weight)
    if kind == 'node':
        if norm:
            return {n:c for n,c in zip(G.nodes,clo)}
        else:
            return {n:c/len(G.nodes) for n,c in zip(G.nodes,clo)}
    elif kind == 'edge':
        
        n_clo = {n:c for n,c in zip(G.nodes,clo)}
        if norm:
            e_clo = {edge:(n_clo[edge[0]]+n_clo[edge[1]]) for edge in G.edges}
        else:
            e_clo = {edge:(n_clo[edge[0]]+n_clo[edge[1]])/len(G.nodes)/2 for edge in G.edges}
        return e_clo