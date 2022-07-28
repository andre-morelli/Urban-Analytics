# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:05:17 2020

@author: ANDRE BORGATO MORELLI
"""

from shapely.geometry import Polygon, Point
import geopandas as gpd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import random
import math

def remove_excess_polys(G, polys, buffer=20, exclude_highways=None, drop_duplicates=True, advanced_cutt=True):
    if advanced_cutt:
        G_new = nx.MultiDiGraph(nx.DiGraph(G))
        G_proj = ox.project_graph(G_new)
        if exclude_highways != None:
                for item in exclude_highways.copy():
                    exclude_highways.append(item+'_link')
                for e1, e2, hway in G_proj.edges(data='highway'):
                    if type(hway) == list:
                        G_proj.edges[(e1,e2,0)]['highway'] = tuple(G_proj.edges[(e1,e2,0)]['highway'])
                        for ref_hway in exclude_highways:
                            for item in hway:
                                if item in ref_hway and tuple(hway) not in exclude_highways:
                                    exclude_highways.append(tuple(hway))
                                    
        G_gdf = ox.graph_to_gdfs(G_proj,nodes=False, node_geometry=False)
        if buffer > 0:
            G_gdf['geometry'] = G_gdf.geometry.buffer(buffer)
        G_gdf = G_gdf.to_crs(G.graph['crs'])
        if exclude_highways != None:
            for hway in exclude_highways:
                G_gdf = G_gdf[G_gdf['highway']!=hway]
    else: 
        G_gdf = ox.graph_to_gdfs(G,nodes=False, node_geometry=False)
    if 'id' not in polys.columns:
        polys['id'] = [n for n in range(len(polys))]
    new_polys = gpd.sjoin(polys,G_gdf)
    
    
    if drop_duplicates:
        return new_polys[['id','geometry']].drop_duplicates()
    else:
        return new_polys

def cutt_from_predef_geometry(G, polys, geom_type='enclosing_rectangle', G_proj=None, cutt_buffer=200, buffer=20):
    G_gdf = ox.graph_to_gdfs(G, nodes=False)
    if G_proj==None:
        G_proj = ox.project_graph(G)
    proj = G_gdf.to_crs(G_proj.graph['crs'])
    
    if geom_type == 'convex_hull':
        proj['geometry'] = proj.geometry.buffer(cutt_buffer)
        proj = proj.to_crs(G.graph['crs'])
        geom = proj.unary_union.convex_hull

    elif geom_type == 'buffer':
        proj['geometry'] = proj.geometry.buffer(cutt_buffer)
        proj = proj.to_crs(G.graph['crs'])
        geom = proj.geometry.unary_union
    
    elif geom_type == 'enclosing_rectangle':
        a = proj.unary_union.bounds #xmin, ymin, xmax, ymax
        a=list(a)
        for index, val in enumerate(a):
            if index < 2:
                a[index] = val - buffer
            else:
                a[index] = val + buffer
        geom = Polygon([(a[0], a[1]), (a[0], a[3]), (a[2], a[3]), (a[2],a[1])])
        temp_gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs=proj.crs)
        geom = temp_gdf.to_crs(G.graph['crs']).geometry[0]
    extr = gpd.GeoDataFrame({'geometry': [geom]},
                                crs=G.graph['crs'])
                                
    resu = gpd.sjoin(extr, polys, how="inner", op='contains')
    resu_li = resu['index_right'].unique().tolist()
    inters = [re_indx for re_indx in polys.index.tolist() if re_indx not in resu_li]
    n = 1
    for ind in inters:
        try:
            polys.loc[ind, 'geometry'] = polys['geometry'][ind].intersection(extr['geometry'][0])
        except:
            geometry = polys['geometry'][ind].intersection(extr['geometry'][0])
            if len(geometry) > 1:
                polys.loc[ind, 'geometry'] = geometry[0]
                for entry in geometry[1:]:
                    polys = polys.append(gpd.GeoDataFrame({'id': [len(inters)+n],'geometry': [entry]}, crs=G.graph['crs']),sort=True)
                    n += 1
            elif len(polys['geometry'][ind].intersection(extr['geometry'][0])) == 0:
                polys = polys.drop(ind)
    return polys

def get_regular_grid(G, tolerance = 0, remove_empty = True, unit_size=200, exclude_highways=None,
                    buffer=20, cutt_geom='buffer', cutt_buffer=200, drop_duplicates=True,
                    advanced_cutt=True):
    if not advanced_cutt:
        cutt_geom=None
        buffer = 0
        exclude_highways=None
        cutt_buffer = 0
    #remove parallel:
    if exclude_highways != None:
        exclude_highways = list(exclude_highways)
    G_new = nx.MultiDiGraph(nx.DiGraph(G))
    G_proj = ox.project_graph(G_new)
    xmax, xmin = -1e26,1e26
    ymax, ymin = -1e26,1e26
    for n, d in G_proj.nodes(data=True):
        if d['x']<xmin:
            xmin=d['x']
        if d['x']>xmax:
            xmax=d['x']
        if d['y']<ymin:
            ymin=d['y']
        if d['y']>ymax:
            ymax=d['y']
    xmin, ymin = xmin-tolerance, ymin-tolerance
    xmax, ymax = xmax+tolerance, ymax+tolerance
        
    size = max((xmax-xmin), (ymax-ymin))
    num_units = int(size//unit_size +1)
    real_size = num_units * unit_size
    
    xinit = xmin-(real_size-(xmax-xmin))/2
    xfinal = xmax+(real_size-(xmax-xmin))/2
    yinit = ymin-(real_size-(ymax-ymin))/2
    yfinal = ymax+(real_size-(ymax-ymin))/2
    
    polys = []
    for x_axis in range(num_units):
        for y_axis in range(num_units):
            x1,y1 = (xinit+unit_size*x_axis),(yinit+unit_size*y_axis)
            poly = Polygon([(x1,y1), (x1+unit_size,y1), (x1+unit_size,y1+unit_size), (x1,y1+unit_size)])
            polys.append(poly)
    ids = [n for n in range(len(polys))]
    polys = gpd.GeoDataFrame({'id':ids,'geometry':polys},crs=G_proj.graph['crs'])
    polys = polys.to_crs(G.graph['crs'])

    if cutt_geom != None:
        polys = cutt_from_predef_geometry(G, polys, G_proj=G_proj, geom_type = cutt_geom, 
                                          cutt_buffer=cutt_buffer, buffer=buffer)
            
    G_proj = None    
    if remove_empty:
        return remove_excess_polys(G_new, polys, buffer=buffer, exclude_highways=exclude_highways, 
                                    drop_duplicates=drop_duplicates, advanced_cutt=advanced_cutt)
    else:
        return polys

def get_voronoi_grid(G, tolerance=15, remove_random=None, buffer=20, cutt_buffer= 400, remove_empty = True, 
                    exclude_highways=None, is_dual=False, primal_graph=None, max_area_m=None, 
                    cutt_geom = 'convex_hull', drop_duplicates=True, advanced_cutt=True):
    
    """ Função que recebe um grafo e retorna os poligonos das regiões voronoi associadas a cada ponto.
        G: grafo (geralmente obtido com o osmnx);
        cut_extremes: bool - se True, constroe um retangulo que abrange os pontos extremos do grafo e recorta regiões que
                            possivelmente saem desses limites.
        Retorna:
                voronoi_gdf: GeoDataFrame com os pontos (coordenadas) e os poligonos das regiões criadas;
                p_pol: dicionário com os tuplas das coordenadas dos pontos como key e os poligonos como values
                vor: instancia voronoi, que pode ser usada para plotar os pontos e as regiões
    """
    # Criando um vetor com os pontos do grafo a partir do gdf do mesmo
    if not is_dual:
        primal_graph = G
    if exclude_highways != None:
        exclude_highways = list(exclude_highways)
    ids = []
    G_ = G.copy()
    G_proj = ox.project_graph(G)
    for node, d in G.nodes(data=True):
        if d['osmid'] in ids:
            G_.remove_node(node)
        else:
            ids.append(d['osmid'])
    
    intersections = ox.clean_intersections(G_proj, tolerance=tolerance)
    points = gpd.GeoDataFrame({'geometry':intersections},crs=G_proj.graph['crs'])
    points = points.to_crs(G_.graph['crs']).geometry
    if remove_random != None:
        points = [(p.x,p.y) for p in points if random.random()>=remove_random]
    else:
        points = [(p.x,p.y) for p in points]
    point_list = np.array(points)
    # Criação da classe Voronoi, com os respectivos vertices e regiões
    vor = Voronoi(point_list)
    verts = vor.vertices.tolist()
    regs = vor.regions
    pr = vor.point_region
    p_reg = {}
    p_pol = []
    p_coo = []

    # cada pr[i] é um índice da lista de regiões relacionada ao ponto de índice i da lista verts  
    for i in range(len(pr)):
        point = tuple(point_list[i])
        r = regs[pr[i]] # cada r é uma região, expressa por uma lista com os índices dos vertices que a compõem

        if (-1 in r) or (len(r)==0): continue # ignora regiões inexistentes ou infinitas
        else:
            p_reg[point] = [verts[v] for v in r] # dicionário com ponto as key e região as value
            p_pol.append(Polygon(p_reg[point]))
            p_coo.append(point)
    voronoi_gdf = gpd.GeoDataFrame({'id': [n for n in range(len(p_pol))],'geometry': list(p_pol), 'coord': p_coo}, 
                                   crs=G_.graph['crs'])
    if cutt_geom!=None:
        voronoi_gdf = cutt_from_predef_geometry(primal_graph, voronoi_gdf, G_proj=G_proj, geom_type = cutt_geom, 
                                                cutt_buffer=cutt_buffer, buffer=buffer)
        
    if remove_empty:
        polys = remove_excess_polys(primal_graph, voronoi_gdf, buffer=buffer, exclude_highways=exclude_highways, 
                                    drop_duplicates=drop_duplicates, advanced_cutt=advanced_cutt)
        if max_area_m != None:
            polys_proj = polys.to_crs(G_proj.graph['crs'])
            polys = polys[polys_proj.geometry.area<=max_area_m]
        return polys
    return voronoi_gdf
    
    
    
def create_hexagon(l, x, y):
    """
    Create a hexagon centered on (x, y)
    :param l: length of the hexagon's edge
    :param x: x-coordinate of the hexagon's center
    :param y: y-coordinate of the hexagon's center
    :return: The polygon containing the hexagon's coordinates
    """
    c = [[x + math.cos(math.radians(angle)) * l, y + math.sin(math.radians(angle)) * l] for angle in range(0, 360, 60)]
    return Polygon(c)

def create_hexgrid(bbox, side):
    """
    returns an array of Points describing hexagons centers that are inside the given bounding_box
    :param bbox: The containing bounding box. The bbox coordinate should be in Webmercator.
    :param side: The size of the hexagons'
    :return: The hexagon grid
    """
    grid = []

    v_step = math.sqrt(3) * side
    h_step = 1.5 * side

    x_min = min(bbox[0], bbox[2])
    x_max = max(bbox[0], bbox[2])
    y_min = min(bbox[1], bbox[3])
    y_max = max(bbox[1], bbox[3])

    h_skip = math.ceil(x_min / h_step) - 1
    h_start = h_skip * h_step

    v_skip = math.ceil(y_min / v_step) - 1
    v_start = v_skip * v_step

    h_end = x_max + h_step
    v_end = y_max + v_step

    if v_start - (v_step / 2.0) < y_min:
        v_start_array = [v_start + (v_step / 2.0), v_start]
    else:
        v_start_array = [v_start - (v_step / 2.0), v_start]

    v_start_idx = int(abs(h_skip) % 2)

    c_x = h_start
    c_y = v_start_array[v_start_idx]
    v_start_idx = (v_start_idx + 1) % 2
    while c_x < h_end:
        while c_y < v_end:
            grid.append((c_x, c_y))
            c_y += v_step
        c_x += h_step
        c_y = v_start_array[v_start_idx]
        v_start_idx = (v_start_idx + 1) % 2

    return grid

def get_hex_grid(region, side=200):
    """
    returns hexagon grid in graph's region
    :param G: city Graph
    :param side: The size of the hexagons' edge (half the diagonal)
    :return: Hexagon grid gdf
    """
    if type(region)==nx.MultiDiGraph:
        G_ = region
        G = ox.project_graph(G_)
        edges = ox.graph_to_gdfs(G, nodes=False)
        bbox = edges.total_bounds
        crs = G.graph['crs']
    elif type(region)==gpd.GeoDataFrame:
        region = ox.project_gdf(region)
        edges = region
        bbox = region.total_bounds
        crs = edges.crs
    
    hex_centers = create_hexgrid(bbox, side)
    geom = [create_hexagon(side, center[0], center[1]) for center in hex_centers]
    ids = [n for n in range(len(geom))]
    gdf=gpd.GeoDataFrame({'id':ids},geometry=geom,crs = edges.crs)
    
    gdf = gpd.GeoDataFrame(geometry=gpd.sjoin(gdf,edges).geometry.unique(),crs = crs)
    
    return gdf.to_crs('epsg:4326')