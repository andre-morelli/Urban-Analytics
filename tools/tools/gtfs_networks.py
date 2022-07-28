import osmnx as ox
import networkx as nx
import gtfs_kit as gk
import numpy as np

def convert_time_string(df,cols=['departure_time','arrival_time']):
    df=df.copy()
    for col in cols:
        df[col] = [(int(n.split(':')[0])*3600+
                    int(n.split(':')[1])*60+
                    int(n.split(':')[2]))
                   for n in df[col]]
    return df

def get_complete_gtfs_database(gtfs_path,start=6*3600,stop=9*3600,
                             day_of_the_week='wednesday'):
    feed = gk.read_feed(gtfs_path, dist_units='m')
    
    complete=feed.stop_times
    for df,col in [(feed.trips,'trip_id'),
                   (feed.routes,'route_id'),
                   (feed.stops,'stop_id')]:
        complete=complete.merge(df,on=col)
    complete = convert_time_string(complete)
    
    t=len(complete)
    # the wednesday rule!!!!
    wed = feed.calendar[(feed.calendar[day_of_the_week]==1)]['service_id'].unique()
    complete=complete[(complete['service_id'].isin(wed))]
    #frequencies if possible
    if feed.frequencies is not None:
        freq=get_frequency_df(feed,start,stop)
        routes = freq['trip_id'].unique()
        complete=complete[complete['trip_id'].isin(routes)]
    else:
        freq=None
    return complete,freq,feed    

def no_oulier_mean(x):
    x=np.array(x)
    std=np.std(x)
    mean=np.mean(x)
    x=x[(x>=mean-1.5*std)&
        (x<=mean+1.5*std)]
    filt_mean=(np.mean(x) if len(x)>0 else mean)
    return filt_mean

def get_frequency_df(feed,start,stop):
    freqs = convert_time_string(feed.frequencies,cols=['start_time','end_time'])
    freqs = freqs[(freqs['start_time']>=start)&
                  (freqs['end_time']<=stop)]
    freqs = freqs.merge(feed.trips,on='trip_id').merge(feed.routes,on='route_id')
    return freqs

def calc_headway(route_id,freqs,op=no_oulier_mean):
    if len(freqs[freqs['route_id']==route_id])==0:
        return None
    else:
        return op(freqs[freqs['route_id']==route_id]['headway_secs'])

def clean_shortcuts(L):
    while True:
        for node in L.nodes:
            ns = list(nx.neighbors(L,node))
            if len(ns)>1:
                tmax=0
                for n in ns:
                    t=L.edges[(node,n,0)]['time_sec']
                    if t>tmax:
                        tmax=t
                        remove=(node,n,0)
                L.remove_edge(*remove)
                break
        else:
            break
    return L

lines=[]

def get_transit_lines_as_graphs(gtfs_path,start=6*3600,stop=9*3600,
                                clean=True,interpolate_stop_times=False,
                                rest_per_cycle=600):
    complete,freq,feed=get_complete_gtfs_database(gtfs_path)
    lines=[]
    for rid in complete['route_id'].unique():
        temp=complete[complete['route_id']==rid]
        name=temp.iloc[0]['route_short_name']
        lname=temp.iloc[0]['route_long_name']
        color=temp.iloc[0]['route_color']
        mode=temp.iloc[0]['route_type']
        if mode==1:
            print(len(temp))
        temp=temp.sort_values(['trip_id','stop_sequence'])
        ns=temp['stop_id'].unique()
        nodes={}
        for sid in ns:
            dat = feed.stops.loc[feed.stops['stop_id']==sid].iloc[0]
            nodes[f'{rid}_{sid}']={'x':dat['stop_lon'],
                        'y':dat['stop_lat'],
                        'name':dat['stop_name']}
        if len(nodes)<2:
            continue
        L = nx.MultiDiGraph()
        L.add_nodes_from(nodes)
        nx.set_node_attributes(L,nodes)

        #do it by trip or sample them?
        edges={}
        cycles={}
        for tid in temp['trip_id'].unique():
            temp2=temp[temp['trip_id']==tid]
            if len(temp2)<2:
                continue
            if temp2.iloc[0]['stop_sequence']==1:
                cycles[temp2.iloc[0]['direction_id']]=(cycles.get(temp2.iloc[0]['direction_id'],[])+
                                                      [(temp2.iloc[-1]['departure_time']-
                                                      temp2.iloc[0]['departure_time'])+rest_per_cycle])
            times = (np.array(temp2['departure_time'])[1:]-
                     np.array(temp2['departure_time'])[:-1])
            if 'shape_dist_traveled' in temp2.columns:
                dists = (np.array(temp2['shape_dist_traveled'])[1:]-
                         np.array(temp2['shape_dist_traveled'])[:-1])
            else:
                dists = [np.nan]*len(times)
            es = zip(temp2['stop_id'].to_list()[:-1],
                     temp2['stop_id'].to_list()[1:],
                     times,dists)
            for e0,e1,t,d in es:
                e0,e1 = f'{rid}_{e0}',f'{rid}_{e1}'
                if (e0,e1,0) in edges:
                    edges[(e0,e1,0)][0].append(t)
                    edges[(e0,e1,0)][1].append(d)
                else:
                    edges[(e0,e1,0)]=([t],[d])

        for e in edges:
            t,d=edges[e]
            edges[e]={'time_sec':np.mean(t),
                      'route_id':rid,
                      'length':np.mean(d),
                      'short_name':name,
                      'long_name':lname,
                      'color':color,
                      'mode':mode}

        L.add_edges_from(edges)
        nx.set_edge_attributes(L,edges)
        
        #clean up?
        if clean:
            clean_shortcuts(L)
        
        #estimate cycle
        c = 0
        for v in cycles.values():
            c+=np.mean(v)
        

        if freq is not None:
            hway=calc_headway(rid,freq)

        else:#try to infer
            times=[]
            for node in L.nodes:
                f=temp[temp['stop_id']==node.replace(f'{rid}_','')].sort_values('departure_time')
                if len(f)>1:
                    times.append(np.mean(np.array(f['departure_time'])[1:]-
                                         np.array(f['departure_time'])[:-1]))
            if len(times)>0:
                hway=no_oulier_mean(times)
            else:
                hway=None
        L.graph={'route_id': rid,
                 'mode':mode,
                 'name':name,
                 'crs': 'epsg:4326',
                 'cycle_time':c,
                 'headway':hway}
        lines.append(L)
    return lines

def get_closest_nodes(G,subs):
    #can two points have the same node? 
    node_names={}
    X,Y=[],[]
    subs_coords=[0]+[len(sub) for sub in subs]
    subs_coords = np.cumsum(subs_coords)
    for sub in subs:
        for node in sub.nodes:
            x0,y0=sub.nodes[node]['x'],sub.nodes[node]['y']
            X.append(x0)
            Y.append(y0)
    ns=ox.get_nearest_nodes(G,np.array(X),np.array(Y),method='balltree')
    for sub,pos0,pos1 in zip(subs,subs_coords[:-1],subs_coords[1:]):
#         print(len(ns[pos0:pos1]),len(sub))
        node_names={node:name for node,name in zip(sub.nodes,ns[pos0:pos1])}
        nx.set_node_attributes(sub,node_names,'attached')
    return None

def add_bus_line(G,L,line_name,headway=None,freq_cap=5,#min
                 buses=None,avg_speed=20,avg_stop_time=0,
                 attach_node_attr='attached',copy=False):
    L = L.copy()
    mode=list(L.edges(data='mode'))[0][2]
    if buses is None and headway is None:
        raise ValueError('buses or headway must be set')
    if headway is None:
        if buses>0:
            T = (sum(l for _,_,l in L.edges(data='time_sec'))+
                     len(L)*avg_stop_time)/buses
            T = T//5*5#floor to multiple of 5
            if T<freq_cap:
                T = freq_cap
        else:
            T = 1e10
        headway=T
    if copy:
        G = G.copy()
#     L = nx.relabel_nodes(L,{k:f'{line_name}_{n}' for n,k in enumerate(L.nodes)})
    G.update(L)
    edges = {}
    for n in L:
        pair = L.nodes[n][attach_node_attr]
        if pair not in (G.nodes):
            continue
        #entering transit system:
        edges[(pair,n,0)]=dict(time_sec=headway/2,length=.01,mode=mode,route_id=line_name,etype='boarding')
#         G.add_edge(pair,n,key=0,time_min=headway/2,length=.01)
        
        #leaving transit system:
        edges[(n,pair,0)]=dict(time_sec=.1,length=.01,mode=mode,route_id=line_name,etype='unboarding')
#         G.add_edge(n,pair,key=0,time_min=.1,length=.01)
    G.add_edges_from(edges)
    nx.set_edge_attributes(G,edges)
    try:
        G.graph['bus_lines'][line_name]={'headway':headway,
                                         'avg_speed':avg_speed,
                                         'avg_stop_time':avg_stop_time}
    except:
        G.graph['bus_lines']={}
        G.graph['bus_lines'][line_name]={'headway':headway,
                                         'avg_speed':avg_speed,
                                         'avg_stop_time':avg_stop_time}
    return G