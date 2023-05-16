import networkx as nx












def up_walk(g, matched_fun_item, matched_func_list):
    
    start_node = matched_fun_item
    # start_node_info = nx.info(g, start_node)
    in_edges = g.in_edges(start_node)
    out_edges = g.out_edges(start_node)
    
    for in_edge in in_edges:
        father_node = in_edge[0]
        father_out_edges = g.out_edges(father_node)
    for father_out_edge in father_out_edges:
        brother_node = father_out_edge[1]
        if start_node != brother_node:
            brother_tree = nx.ego_graph(g, brother_node, radius=10000)
            node_list = brother_tree.nodes()
    
            edge_list = brother_tree.edges()
        
    # test_node_info = nx.info(g, "BZ2_crc32Table")
    print(in_edge)
    #print(out_edge)


def down_walk(g, matched_fun_item, matched_func_list):
    pass


    
  

def get_father_brother_node(start_node, graph):
    father_node_dict = {}
    in_edges = graph.in_edges(start_node)
    for in_edge in in_edges:
        father_node = in_edge[0]
        father_node_dict[father_node] = []
        father_out_edges = graph.out_edges(father_node)
        for father_out_edge in father_out_edges:
            maybe_brother_node = father_out_edge[1]
            if maybe_brother_node != start_node:
                brother_node = maybe_brother_node    
                father_node_dict[father_node].append(brother_node)
    return father_node_dict
    

def get_child_brother_node(start_node, graph):
    child_node_dict = {}

    out_edges = graph.out_edges(start_node)
    for out_edge in out_edges:
        child_node = out_edge[1]
        child_node_dict[child_node] = []
        child_in_edges = graph.in_edges(child_node)
        for child_in_edge in child_in_edges:
            maybe_brother_node = child_in_edge[0]
        if maybe_brother_node != start_node:
            brother_node = maybe_brother_node    
            child_node_dict[child_node].append(brother_node)
    
    return child_node_dict


def get_child_node(start_node, graph):
    child_node_list = []

    out_edges = graph.out_edges(start_node)
    for out_edge in out_edges:
        child_node = out_edge[1]
        child_node_list.append(child_node)

    
    return child_node_list

def get_children_list(start_node, graph, walked_map):
    children_list = []
    if start_node not in walked_map:
        walked_map.add(start_node)
        children_list= get_child_node(start_node, graph)
        for child in children_list:
            child_children_list = get_children_list(child, graph, walked_map)
            children_list.extend(child_children_list)
    return children_list
    



