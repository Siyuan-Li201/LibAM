import itertools
import os, copy
from turtle import Turtle
import difflib
import networkx as nx

from app.explore import *
from tqdm import tqdm
import random
# import pydot
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn.functional as F

def transform_data(data):
    """
    :param data: {'n_num': 10, 'features': [[1, 2, ...]], 'succs': [[], ..]}
    :return:
    """
    feature_dim = len(data['embeddings'][0])
    node_size = data['n_num']
    X = np.zeros((1, node_size, feature_dim))
    mask = np.zeros((1, node_size, node_size))
    for start_node, end_nodes in enumerate(data['succs']):
        X[0, start_node, :] = np.array(data['embeddings'][start_node])
        for node in end_nodes:
            mask[0, start_node, node] = 1
    return X, mask

def embed_by_feat_torch(feat, gnn):
    X, mask = transform_data(feat)
    X = torch.from_numpy(X).to(torch.float32).cuda()
    mask = torch.from_numpy(mask).to(torch.float32).cuda()
    # return gnn.forward_once(X, mask).cpu().detach().numpy()
    return gnn.forward_once(X, mask)


#20221028 修正浅拷贝bug
def get_taint_graph(pydot_graph, taint_func_list):
    # pydot_graph = nx.nx_pydot.to_pydot(g)
    
    # new_pydot_graph = pydot_graph.copy()
    
    for matched_func in taint_func_list:
        try:
            pydot_graph.get_node(matched_func)[0].set_color("blue")
            pydot_graph.get_node(matched_func)[0].set_style("filled")
        except:
            pydot_graph.get_node('"'+matched_func+'"')[0].set_color("blue")
            pydot_graph.get_node('"'+matched_func+'"')[0].set_style("filled")
    
    return pydot_graph
    pydot_graph.write_png('example6.png')
    
    # pos = nx.spring_layout(g)
    # plt.figure(figsize=(100,100))
    # #nx.draw_networkx_nodes(g, pos, nodelist=taint_func_list, node_color='b')
    # nx.draw(g, pos=pos,with_labels=False)
    # plt.savefig('graphe.png')
    # plt.show()


def get_related_func(sim_funcs, matched_func_ingraph_list):
    
    related_func = {}
    
    for obj_func in sim_funcs:
        related_func[obj_func] = []
        for node_pair in matched_func_ingraph_list:
            if node_pair[0] == obj_func and node_pair[1] not in related_func[obj_func]:
                related_func[obj_func].append(node_pair[1])
    
    return related_func


def get_related_func_one(obj_func, matched_func_ingraph_list, all_related_funcs):
    
    related_func = []
    
    if obj_func in all_related_funcs:
        return all_related_funcs[obj_func], all_related_funcs
    else:
        # for obj_func in sim_funcs:
        # related_func[obj_func] = []
        for node_pair in matched_func_ingraph_list:
            if node_pair[0] == obj_func and node_pair[1] not in related_func:
                related_func.append(node_pair[1])
        all_related_funcs[obj_func] = related_func
        
        return related_func, all_related_funcs


def get_afcg_one_annoy(func_pair, sim_funcs, all_afcg):
    
    afcg = []
    afcg_pre = []
    # for entry in sim_funcs:
        # afcg[entry] = []
    if func_pair in all_afcg:
        afcg_pre = all_afcg[func_pair]
        for child_node in afcg_pre:
            if child_node in sim_funcs and child_node != func_pair:
                # if func_pair not in afcg:
                #     afcg[func_pair] = []
                if child_node not in afcg:
                    afcg.append(child_node)
    return afcg
        # all_afcg[func_pair] = afcg
    # else:
    #     return []
        # walked_map_set = set()
        # child_node_list = get_children_list(func_pair, object_graph, walked_map_set)
        
           
    

def get_afcg_one(func_pair, sim_funcs, object_graph, all_afcg):
    
    afcg = []
    
    # for entry in sim_funcs:
        # afcg[entry] = []
    if func_pair in all_afcg:
        afcg = all_afcg[func_pair]
    else:
        walked_map_set = set()
        child_node_list = get_children_list(func_pair, object_graph, walked_map_set)
        for child_node in child_node_list:
            if child_node in sim_funcs and child_node != func_pair:
                # if func_pair not in afcg:
                #     afcg[func_pair] = []
                if child_node not in afcg:
                    afcg.append(child_node)
        all_afcg[func_pair] = afcg
           
    return afcg, all_afcg


def get_afcg(sim_funcs, object_graph):
    
    afcg = {}
    
    for entry in sim_funcs:
        # afcg[entry] = []
        walked_map_set = set()
        child_node_list = get_children_list(entry, object_graph, walked_map_set)
        for child_node in child_node_list:
            if child_node in sim_funcs and child_node != entry:
                if entry not in afcg:
                    afcg[entry] = []
                if child_node not in afcg[entry]:
                    afcg[entry].append(child_node)
           
    return afcg

def afcg_cost(obj_afcg, cdd_afcg, matched_func_ingraph_list):
    matched_func = []
    entries_lib2tar = {}
    entries_tar2lib = {}
    for sim_f_pair in matched_func_ingraph_list:
        if sim_f_pair[1] not in entries_tar2lib:
            entries_tar2lib[sim_f_pair[1]] = []
        entries_tar2lib[sim_f_pair[1]].append(sim_f_pair[0])
        if sim_f_pair[0] not in entries_lib2tar:
            entries_lib2tar[sim_f_pair[0]] = []
        entries_lib2tar[sim_f_pair[0]].append(sim_f_pair[1])

    common = 0
    diff_lib2tar = 0
    diff_tar2lib = 0
    for entry in obj_afcg:
        for child in obj_afcg[entry]:
            entries_tar = entries_lib2tar[entry]
            children_tar = entries_lib2tar[child]
            flag = 0
            for child_tar in children_tar:
                for entry_tar in entries_tar:
                    if child_tar in cdd_afcg[entry_tar]:
                        flag = 1
                        matched_func.append([entry_tar, child_tar])
                        break
                if flag:
                    break
            if flag:
                common += 1
            else:
                diff_lib2tar += 1

    for entry in cdd_afcg:
        for child in cdd_afcg[entry]:
            flag = 0
            entries_lib = entries_tar2lib[entry]
            children_lib = entries_tar2lib[child]
            flag = 0
            for child_lib in children_lib:
                for entry_lib in entries_lib:
                    if child_lib in obj_afcg[entry_lib]:
                        flag = 1
                        break
                if flag:
                    break
            if flag:
                continue
            else:
                diff_tar2lib += 1
    if common == 0:
        return 0, 0, []
    return (common, common / (common + diff_lib2tar + diff_tar2lib), matched_func)

def judge_in_obj_graph(g, matched_func_list):
    in_graph_node = []
    node_list = g.nodes()
    for matched_func in matched_func_list:
        if matched_func[0] in node_list:
            in_graph_node.append(matched_func[0])

    return in_graph_node


def judge_in_graph(object_graph, candidate_graph, matched_func_list):
    in_graph_node = []
    obj_node_list = list(object_graph.nodes())
    cdd_node_list = list(candidate_graph.nodes())
    
    
    for matched_func in matched_func_list:
        if "|||" in matched_func[0]:
            matched_func[0] = matched_func[0].split("|||")[-1]
            matched_func[1] = matched_func[1].split("|||")[-1]
        if matched_func[0] in obj_node_list:
            if matched_func[1] in cdd_node_list:
                in_graph_node.append(matched_func)

    return in_graph_node


def get_fcg(fcg_path, project_name):
    
    if "___" in project_name:
        project_name = project_name.replace("___","_")
    
    if "_datasets_isrd_to_gemini" in project_name:
        project_name = project_name.split("_datasets_isrd_to_gemini")[0]
    if "bzip2_" in project_name and "_O" in project_name:
        project_fcg_name = project_name[:6] + project_name +"_fcg.pkl"
    else:
        project_fcg_name = project_name +"_fcg.pkl"

    object_g = nx.read_gpickle(os.path.join(fcg_path, project_fcg_name.replace("openssl_dataset_", "openssl_dataset")))
    
    return object_g

def get_isrd_fcg(fcg_path, project_name):
    
    if "_datasets_isrd_to_gemini" in project_name:
        project_name = project_name.split("_datasets_isrd_to_gemini")[0]
    
    if "|||" in project_name:
        project_fcg_name = project_name.replace("|||", "||||") +"_fcg.pkl"
    else:
        project_fcg_name = project_name +"_fcg.pkl"

    try:
        object_g = nx.read_gpickle(os.path.join(fcg_path, project_fcg_name.replace("openssl_dataset_", "openssl_dataset")))
    except:
        object_g = nx.read_gpickle(os.path.join(fcg_path, project_fcg_name.replace("||||", "_").replace("openssl_dataset_", "openssl_dataset")))
    
    return object_g


def get_isrd_fcg_new(fcg_path, project_name):
    
    if "_datasets_isrd_to_gemini" in project_name:
        project_name = project_name.split("_datasets_isrd_to_gemini")[0]
    project_fcg_name = project_name.replace("|||", "___") +"_fcg.pkl"
    try:
        object_g = nx.read_gpickle(os.path.join(fcg_path, project_fcg_name.replace("openssl_dataset_", "openssl_dataset")))
    except:
        object_g = nx.read_gpickle(os.path.join(fcg_path, project_fcg_name.replace("||||", "_").replace("openssl_dataset_", "openssl_dataset")))
    
    return object_g

    
def get_libdb_fcg(fcg_path, project_name):
    
    # if "___" in project_name:
    #     project_name = project_name.replace("___","_")
    
    # if "_datasets_isrd_to_gemini" in project_name:
    #     project_name = project_name.split("_datasets_isrd_to_gemini")[0]
    # if "bzip2_" in project_name and "_O" in project_name:
    #     project_fcg_name = project_name[:6] + project_name +"_fcg.pkl"
    # else:
    #     project_fcg_name = project_name +"_fcg.pkl"
    find_fcg = True
    
    if "|||" in project_name:
        arch = project_name.split("|||")[0]
        binary = project_name.split("|||")[1]
        project = project_name.split("|||")[2]
        fcg_name = project+"||||lib||||"+arch+"||||"+binary+"_fcg.pkl"
    else:
        fcg_name = project_name.split("_reuse_fcg_dict_")[0]+"_fcg.pkl"
    
    
    try:
        object_g = nx.read_gpickle(os.path.join(fcg_path, fcg_name))
    except:
        object_g = None
        find_fcg = False
    
    return object_g, find_fcg
    
    
    
def filter_200_lib(object_cdd_func_dict):
    filtered_lib_dict = {}
    filer_lib_dict = {}
    for matched_item in object_cdd_func_dict:
        lib_name = matched_item.split("||||")[1].split("----")[0]
        if lib_name not in filer_lib_dict:
            filer_lib_dict[lib_name] = 0
        filer_lib_dict[lib_name] += 1
    filer_lib_dict_sorted = list(filer_lib_dict.keys())
    filer_lib_dict_sorted.sort( key = filer_lib_dict.__getitem__ , reverse=True)
    # filer_lib_dict_sorted = sorted(filer_lib_dict.items(),  key=lambda d: d[1], reverse=True)[:100]
    
    for matched_item in object_cdd_func_dict:
        lib_name = matched_item.split("||||")[1].split("----")[0]
        if lib_name in filer_lib_dict_sorted[:200]:         #isrd这里为50
            filtered_lib_dict[matched_item] = object_cdd_func_dict[matched_item]
    
    
    return filtered_lib_dict


def filter_100_func(object_cdd_func_dict):
    filtered_func_dict = {}
    filer_func_dict = {}
    for matched_item in object_cdd_func_dict:
        obj_func_name = matched_item.split("||||")[0]
        candidate_item = matched_item.split("||||")[1]
        if obj_func_name not in filer_func_dict:
            filer_func_dict[obj_func_name] = {}
        if candidate_item not in filer_func_dict[obj_func_name]:
            filer_func_dict[obj_func_name][candidate_item] = object_cdd_func_dict[matched_item]
    
    for obj_func_item in filer_func_dict:
        object_cdd_func_list = sorted(filer_func_dict[obj_func_item].items(),  key=lambda d: d[1], reverse=True)[:200]  
        for object_cdd_func_item in object_cdd_func_list:
            filtered_func_dict[obj_func_item+"||||"+object_cdd_func_item[0]] = object_cdd_func_list[1]
    
    
    return filtered_func_dict


def filter_500_anchor(object_cdd_func_dict):
    
    object_cdd_func_list = sorted(object_cdd_func_dict.items(),  key=lambda d: d[1], reverse=True)  #isrd这里为5w
    
    
    
    
    return object_cdd_func_list


# 修复bug：替换后可能只与一边相连
def have_subgraph_edge_v4(func_list_item, new_node_pair, object_graph, candidate_graph):
    connect_flag = [-1, -1]
    re_connect_flag = [-1, -1]
    # connect_flag = 0
    for i in range(0, len(func_list_item)):
        (is_obj_subgraph, distance1) = is_subgraph_edge_v3([func_list_item[len(func_list_item)-i-1][0], new_node_pair[0]], object_graph)
        (is_cdd_subgraph, distance2) = is_subgraph_edge_v3([func_list_item[len(func_list_item)-i-1][1], new_node_pair[1]], candidate_graph)
        if (is_obj_subgraph == True and is_cdd_subgraph == True):
            if connect_flag == [-1, -1] or (distance1+distance2) < connect_flag[1]:
                connect_flag = (len(func_list_item)-i, (distance1+distance2))
        # if [new_node_pair[0], old_func_pair[0]] == ['BZ2_hbAssignCodes', 'BZ2_bzwrite']:
        #     print("warning")
        (is_obj_subgraph_re, distance1) = is_subgraph_edge_v3([new_node_pair[0], func_list_item[i][0]], object_graph)
        (is_cdd_subgraph_re, distance2) = is_subgraph_edge_v3([new_node_pair[1], func_list_item[i][1]], candidate_graph)
        if (is_obj_subgraph_re == True and is_cdd_subgraph_re == True):
            if re_connect_flag == [-1, -1] or (distance1+distance2) < re_connect_flag[1]:
                re_connect_flag = (i, (distance1+distance2))
    
    if connect_flag != [-1, -1] and re_connect_flag != [-1, -1]:
        if connect_flag[1] > re_connect_flag[1]:
            return (1, re_connect_flag[0])
        else:
            return (1, connect_flag[0])
    elif connect_flag != [-1, -1]:
        return (2, connect_flag[0])
    elif re_connect_flag != [-1, -1]:
        return (3, re_connect_flag[0])
    else:
        return 0, 0

    
def get_cdd_func_dict(object_cdd_func_dict):
    cdd_project_dict = {}
    object_cdd_func_dict = filter_200_lib(object_cdd_func_dict)
    # object_cdd_func_dict = filter_100_func(object_cdd_func_dict)
    object_cdd_func_list = filter_500_anchor(object_cdd_func_dict)
    for matched_item in object_cdd_func_list:
        cdd_item = matched_item[0].split("||||")[1].split("----")[0]
        obj_func_item = matched_item[0].split("||||")[0].split("----")[1]
        cdd_func_item = matched_item[0].split("||||")[1].split("----")[1]
        if cdd_item not in cdd_project_dict:
            cdd_project_dict[cdd_item] = []
        #cdd_project_dict[cdd_item].append(["".join(obj_func_item.split("-")[2:]), "".join(cdd_func_item.split("-")[2:])])
        cdd_project_dict[cdd_item].append(["".join(obj_func_item), "".join(cdd_func_item)])
    return cdd_project_dict


#修复bug:可能替换的节点与另一边不连通
def get_subgraph_func_list_v5(matched_func_ingraph_list, object_graph, candidate_graph):
    
    obj_func_set = set()
    cdd_func_set = set()
    func_list = []
    func_list_set = set()
    
    alignment_max = False
    node_i = 0
    
    for node_pair in matched_func_ingraph_list:
        # if "_start" in node_pair:
        #     print("warning")
        node_i += 1
        # if node_i == 23:
        #     print("warning")
        # if "BZ2_bzDecompress" in node_pair or "BZ2_decompress" in node_pair:#if i == 33:
        #     print("debug")
        #print(i)
        
        #combine same graph
        combine_flag = 0
        combine_num = 0
        combine_str_num = 0
        for func_list_item in func_list:
            combine_str_num += 1
            if len(func_list_item) > 1:
                combine_num += 1
                if combine_num > 1000:
                    combine_flag = 1
                    print("----before combine:"+str(len(func_list)))
                    func_list = combine_fcg(func_list, 80)
                    print("----after combine:"+str(len(func_list)))
                    obj_func_set = set()
                    cdd_func_set = set()
                    func_list_set = set()
                    for func_list_item in func_list:
                        func_list_set.add(str(func_list_item))
                        for func_pair in func_list_item:
                            obj_func_set.add(func_pair[0])
                            cdd_func_set.add(func_pair[1])
                    break
            if combine_str_num > 1500:
                combine_num += 1
                print("----before str combine:"+str(len(func_list)))
                func_list = combine_fcg_str(func_list, 0.8)
                print("----after str combine:"+str(len(func_list)))
                obj_func_set = set()
                cdd_func_set = set()
                func_list_set = set()
                for func_list_item in func_list:
                    func_list_set.add(str(func_list_item))
                    for func_pair in func_list_item:
                        obj_func_set.add(func_pair[0])
                        cdd_func_set.add(func_pair[1])
                break
        
        if combine_flag == 1:   
            combine_num = 0
            combine_str_num = 0
            for func_list_item in func_list:
                combine_str_num += 1
                if len(func_list_item) > 1:
                    combine_num += 1
                    if combine_num > 900:
                        combine_flag = 2
                        print("----before big combine:"+str(len(func_list)))
                        func_list = combine_fcg(func_list, 51)
                        print("----after big combine:"+str(len(func_list)))
                        obj_func_set = set()
                        cdd_func_set = set()
                        func_list_set = set()
                        for func_list_item in func_list:
                            func_list_set.add(str(func_list_item))
                            for func_pair in func_list_item:
                                obj_func_set.add(func_pair[0])
                                cdd_func_set.add(func_pair[1])
                        break
                if combine_str_num > 1300:
                    combine_flag = 2
                    # print("----before str big combine:"+str(len(func_list)))
                    func_list = combine_fcg_str(func_list, 0.7)
                    # print("----after str big combine:"+str(len(func_list)))
                    obj_func_set = set()
                    cdd_func_set = set()
                    func_list_set = set()
                    for func_list_item in func_list:
                        func_list_set.add(str(func_list_item))
                        for func_pair in func_list_item:
                            obj_func_set.add(func_pair[0])
                            cdd_func_set.add(func_pair[1])
                    break
        
        if combine_flag == 2:
            combine_num = 0
            for func_list_item in func_list:
                if len(func_list_item) > 1:
                    combine_num += 1
                    if combine_num > 600:
                        break
            
            if combine_num > 600:
                alignment_max = True
                break
        
        add_flag = 0
        func_list_to_add = []
        func_list_to_insert = {}
        for func_list_item in func_list:
            obj_func_set = [i[0] for i in func_list_item]
            cdd_func_set = [i[1] for i in func_list_item]
            func_list_item_to_add = []
            func_list_item_to_insert = []
            if node_pair[0] not in obj_func_set and node_pair[1] not in cdd_func_set:  
                # for func_list_item in func_list:
                (is_connected_flag, insert_index) = have_subgraph_edge_v4(func_list_item, node_pair, object_graph, candidate_graph)
                if is_connected_flag:
                    add_flag = 1
                    # obj_func_set.add(node_pair[0])
                    # cdd_func_set.add(node_pair[1])
                    func_list_item_to_insert = copy.deepcopy(func_list_item)
                    func_list_item_to_insert.insert(insert_index, node_pair)
                    # func_list_item.insert(insert_index, node_pair)
                    if str(func_list_item_to_insert) not in func_list_set:
                        func_list_set.add(str(func_list_item_to_insert))
                        func_list_to_insert[str(func_list_item)] = func_list_item_to_insert
                    # obj_func_set.add(node_pair[0])
                    # cdd_func_set.add(node_pair[1])
                        
            elif node_pair[1] not in cdd_func_set:
                # func_list_tmp = copy.deepcopy(func_list)
                # for func_list_item_tmp in func_list_tmp:
                for func_pair_tmp in func_list_item:
                    if func_pair_tmp[0] == node_pair[0]:
                        func_list_item_to_add = copy.deepcopy(func_list_item)
                        func_list_item_to_add.remove(func_pair_tmp)
                        (is_connected_flag, insert_index) = have_subgraph_edge_v4(func_list_item_to_add, node_pair, object_graph, candidate_graph)
                        if (func_list_item_to_add==[] and add_flag == 0) or is_connected_flag==1:
                            func_list_item_to_add.insert(insert_index, node_pair)
                            if str(func_list_item_to_add) in func_list_set:
                                func_list_item_to_add = []
                            else:
                                add_flag = 1
                                func_list_set.add(str(func_list_item_to_add))
                                func_list_to_add.append(func_list_item_to_add)
                                # cdd_func_set.add(node_pair[1])
                            break
                        elif is_connected_flag==2:
                            list_len = len(func_list_item_to_add)
                            func_list_item_to_add.insert(insert_index, node_pair)
                            for i in range(1, list_len - insert_index):
                                try:
                                    func_list_item_to_add.pop(-1)
                                except:
                                    print("error")
                            if str(func_list_item_to_add) in func_list_set:
                                func_list_item_to_add = []
                            else:
                                add_flag = 1
                                func_list_set.add(str(func_list_item_to_add))
                                func_list_to_add.append(func_list_item_to_add)
                                # cdd_func_set.add(node_pair[1])
                            break
                        elif is_connected_flag==3:
                            list_len = len(func_list_item_to_add)
                            func_list_item_to_add.insert(insert_index, node_pair)
                            for i in range(0, insert_index):
                                try:
                                    func_list_item_to_add.pop(0)
                                except:
                                    print("error")
                            if str(func_list_item_to_add) in func_list_set:
                                func_list_item_to_add = []
                            else:
                                add_flag = 1
                                func_list_set.add(str(func_list_item_to_add))
                                func_list_to_add.append(func_list_item_to_add)
                                # cdd_func_set.add(node_pair[1])
                            break
                        else:
                            func_list_item_to_add = []
            elif node_pair[0] not in obj_func_set:
                for func_pair_tmp in func_list_item:
                    if func_pair_tmp[1] == node_pair[1]:
                        func_list_item_to_add = copy.deepcopy(func_list_item)
                        func_list_item_to_add.remove(func_pair_tmp)
                        (is_connected_flag, insert_index) = have_subgraph_edge_v4(func_list_item_to_add, node_pair, object_graph, candidate_graph)
                        if (func_list_item_to_add==[] and add_flag == 0) or is_connected_flag==1:
                            func_list_item_to_add.insert(insert_index, node_pair)
                            if str(func_list_item_to_add) in func_list_set:
                                func_list_item_to_add = []
                            else:
                                add_flag = 1
                                func_list_set.add(str(func_list_item_to_add))
                                func_list_to_add.append(func_list_item_to_add)
                                # cdd_func_set.add(node_pair[1])
                            break
                        elif is_connected_flag==2:
                            list_len = len(func_list_item_to_add)
                            func_list_item_to_add.insert(insert_index, node_pair)
                            for i in range(1, list_len - insert_index):
                                try:
                                    func_list_item_to_add.pop(-1)
                                except:
                                    print("error")
                            if str(func_list_item_to_add) in func_list_set:
                                func_list_item_to_add = []
                            else:
                                add_flag = 1
                                func_list_set.add(str(func_list_item_to_add))
                                func_list_to_add.append(func_list_item_to_add)
                                # cdd_func_set.add(node_pair[1])
                            break
                        elif is_connected_flag==3:
                            list_len = len(func_list_item_to_add)
                            func_list_item_to_add.insert(insert_index, node_pair)
                            for i in range(0, insert_index):
                                try:
                                    func_list_item_to_add.pop(0)
                                except:
                                    print("error")
                            if str(func_list_item_to_add) in func_list_set:
                                func_list_item_to_add = []
                            else:
                                add_flag = 1
                                func_list_set.add(str(func_list_item_to_add))
                                func_list_to_add.append(func_list_item_to_add)
                                # cdd_func_set.add(node_pair[1])
                            break
                        else:
                            func_list_item_to_add = []
            else:
                for func_pair_tmp in func_list_item:
                    if func_pair_tmp[1] == node_pair[1] or func_pair_tmp[0] == node_pair[0]:
                        func_list_item_to_add = copy.deepcopy(func_list_item)
                        func_list_item_to_add.remove(func_pair_tmp)
                        (is_connected_flag, insert_index) = have_subgraph_edge_v4(func_list_item_to_add, node_pair, object_graph, candidate_graph)
                        if (func_list_item_to_add==[] and add_flag == 0) or is_connected_flag==1:
                            func_list_item_to_add.insert(insert_index, node_pair)
                            if str(func_list_item_to_add) in func_list_set:
                                func_list_item_to_add = []
                            else:
                                add_flag = 1
                                func_list_set.add(str(func_list_item_to_add))
                                func_list_to_add.append(func_list_item_to_add)
                                # cdd_func_set.add(node_pair[1])
                            break
                        elif is_connected_flag==2:
                            list_len = len(func_list_item_to_add)
                            func_list_item_to_add.insert(insert_index, node_pair)
                            for i in range(1, list_len - insert_index):
                                try:
                                    func_list_item_to_add.pop(-1)
                                except:
                                    print("error")
                            if str(func_list_item_to_add) in func_list_set:
                                func_list_item_to_add = []
                            else:
                                add_flag = 1
                                func_list_set.add(str(func_list_item_to_add))
                                func_list_to_add.append(func_list_item_to_add)
                                # cdd_func_set.add(node_pair[1])
                            break
                        elif is_connected_flag==3:
                            list_len = len(func_list_item_to_add)
                            func_list_item_to_add.insert(insert_index, node_pair)
                            for i in range(0, insert_index):
                                try:
                                    func_list_item_to_add.pop(0)
                                except:
                                    print("error")
                            if str(func_list_item_to_add) in func_list_set:
                                func_list_item_to_add = []
                            else:
                                add_flag = 1
                                func_list_set.add(str(func_list_item_to_add))
                                func_list_to_add.append(func_list_item_to_add)
                                # cdd_func_set.add(node_pair[1])
                            break
                        else:
                            func_list_item_to_add = []
            # if func_list_item_to_add != []:
            #     add_flag = 1
            #     func_list_to_add.append(func_list_item_to_add)
        if func_list_to_add != []:
            func_list.extend(func_list_to_add)
        if func_list_to_insert != {}:
            for func_list_to_insert_item in func_list_to_insert:
                func_list[func_list.index(eval(func_list_to_insert_item))] = func_list_to_insert[func_list_to_insert_item]
        if add_flag == 0:
            func_list.append([])
            func_list[-1].append(node_pair)
            func_list_set.add(str(func_list[-1]))
        
    # combine
    combine_num = 0
    for func_list_item in func_list:
            if len(func_list_item) > 1:
                combine_num += 1
                if combine_num > 50:
                    func_list = combine_fcg(func_list, 80)
                    break
    
    
    if alignment_max:
        return [], alignment_max
    else:
        return func_list, alignment_max

def Alignment_v2(obj_func, cdd_func, obj_afcg, cdd_afcg, obj_sim_funcs_dict, length, obj_sim_funcs, object_graph, cdd_sim_funcs, candidate_graph, matched_func_ingraph_list, tar_afcg_dict, cdd_afcg_dict):
    N = 0
    if len(length) >= 3:
        return N, length
    # A_tpl_desc = cdd_afcg[cdd_func]
    # if obj_func in obj_afcg and cdd_func in cdd_afcg:
    A_tar_desc = obj_afcg
    a_tar_child = random.sample(A_tar_desc, 1)[0]
    
    # subgraph_scale_max = -1
    a_tpl_child = False
    obj_related_funcs_item = obj_sim_funcs_dict[a_tar_child]#, all_related_funcs = get_related_func_one(a_tar_child, matched_func_ingraph_list, all_related_funcs)
    a_tpl_child_item_list = []
    # a_tpl_child_item = random.sample(obj_related_funcs_item, 1)[0]
    # if a_tpl_child_item in cdd_afcg:
    for a_tpl_child_item in obj_related_funcs_item:
    #     # if a_tpl_child_item in cdd_afcg[cdd_func] and a_tpl_child_item in cdd_afcg and len(cdd_afcg[a_tpl_child_item]) > subgraph_scale_max:
        if a_tpl_child_item in cdd_afcg:
            a_tpl_child_item_list.append(a_tpl_child_item)
    if a_tpl_child_item_list != []:
        a_tpl_child = random.sample(a_tpl_child_item_list, 1)[0]
    #         cdd_afcg_child_item = get_afcg_one_annoy(a_tpl_child_item, cdd_sim_funcs, cdd_afcg_dict)
    #         if len(cdd_afcg_child_item) > subgraph_scale_max:
    #             subgraph_scale_max = len(cdd_afcg_child_item)
    #             a_tpl_child = a_tpl_child_item
    #         elif subgraph_scale_max == -1:
    #             subgraph_scale_max = 0
    #             a_tpl_child = a_tpl_child_item
    if a_tpl_child:
        N += 1
        # if len(length)>=4:
        #     print("warning")
        length.append([a_tar_child, a_tpl_child])
        obj_afcg_child = get_afcg_one_annoy(a_tar_child, obj_sim_funcs, tar_afcg_dict)
        obj_related_funcs_new = obj_sim_funcs_dict[a_tar_child]#, all_related_funcs = get_related_func_one(a_tar_child, matched_func_ingraph_list, all_related_funcs)
        cdd_afcg_child = get_afcg_one_annoy(a_tpl_child, cdd_sim_funcs, cdd_afcg_dict)
        if len(obj_afcg_child) > 0 and len(cdd_afcg_child) > 0 and len(obj_related_funcs_new) > 0:
            l, length =  Alignment_v2(a_tar_child, a_tpl_child, obj_afcg_child, cdd_afcg_child, obj_sim_funcs_dict, length, obj_sim_funcs, object_graph, cdd_sim_funcs, candidate_graph, matched_func_ingraph_list, tar_afcg_dict, cdd_afcg_dict)
            N += l
        return N, length
    else:
        return N, length

# def Alignment(obj_func, cdd_func, obj_afcg, cdd_afcg, obj_sim_funcs_dict, length, obj_sim_funcs, object_graph, cdd_sim_funcs, candidate_graph, matched_func_ingraph_list, tar_afcg_dict, cdd_afcg_dict):
#     N = 0
#     if len(length) >= 3:
#         return N, length
#     # A_tpl_desc = cdd_afcg[cdd_func]
#     # if obj_func in obj_afcg and cdd_func in cdd_afcg:
#     A_tar_desc = obj_afcg
#     a_tar_child = random.sample(A_tar_desc, 1)[0]
    
#     subgraph_scale_max = -1
#     a_tpl_child = False
#     obj_related_funcs_item = obj_sim_funcs_dict[a_tar_child]#, all_related_funcs = get_related_func_one(a_tar_child, matched_func_ingraph_list, all_related_funcs)
#     for a_tpl_child_item in obj_related_funcs_item:
#         # if a_tpl_child_item in cdd_afcg[cdd_func] and a_tpl_child_item in cdd_afcg and len(cdd_afcg[a_tpl_child_item]) > subgraph_scale_max:
#         if a_tpl_child_item in cdd_afcg:
#             cdd_afcg_child_item = get_afcg_one_annoy(a_tpl_child_item, cdd_sim_funcs, cdd_afcg_dict)
#             if len(cdd_afcg_child_item) > subgraph_scale_max:
#                 subgraph_scale_max = len(cdd_afcg_child_item)
#                 a_tpl_child = a_tpl_child_item
#             elif subgraph_scale_max == -1:
#                 subgraph_scale_max = 0
#                 a_tpl_child = a_tpl_child_item
#     if a_tpl_child:
#         N += 1
#         # if len(length)>=4:
#         #     print("warning")
#         length.append([a_tar_child, a_tpl_child])
#         obj_afcg_child = get_afcg_one_annoy(a_tar_child, obj_sim_funcs, tar_afcg_dict)
#         obj_related_funcs_new = obj_sim_funcs_dict[a_tar_child]#, all_related_funcs = get_related_func_one(a_tar_child, matched_func_ingraph_list, all_related_funcs)
#         cdd_afcg_child = get_afcg_one_annoy(a_tpl_child, cdd_sim_funcs, cdd_afcg_dict)
#         if len(obj_afcg_child) > 0 and len(cdd_afcg_child) > 0 and len(obj_related_funcs_new) > 0:
#             l, length =  Alignment(a_tar_child, a_tpl_child, obj_afcg_child, cdd_afcg_child, obj_sim_funcs_dict, length, obj_sim_funcs, object_graph, cdd_sim_funcs, candidate_graph, matched_func_ingraph_list, tar_afcg_dict, cdd_afcg_dict)
#             N += l
#         return N, length
#     else:
#         return N, length
# # else:
# #     return N, length
     
def calculate_final_score(gnn_score, fcg_scale, align_num, fcg_num):
    fcg_factor = fcg_scale / fcg_num
    align_factor = align_num / fcg_num
    return gnn_score * fcg_factor * align_factor
    pass


def RARM_score(alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, align_rate):#, alignment_max, max_fcg):
    # if alignment_num_score > 3:
    #     alignment_num_score = 3
    # alignment_num_score_deal = alignment_num_score/alignment_max
    # node_fcg_scale_score_deal = node_fcg_scale_score/max_fcg
    align_rate_score = 0.3 * align_rate + 0.7
    final_score = node_gnn_score * align_rate_score# * node_fcg_scale_diff_score # *alignment_num_score_deal * # * node_fcg_scale_score_deal

    return final_score



def reuse_area_detection_utils(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, func_embeddings, gnn, fcgs_num):
    reuse_flag = False
    black_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    max_alignment_num = 0
    
    target_reuse_lib_dict = []
    target_reuse_area_dict = {}
    
    alignment_anchor_list = []
    
    obj_sim_funcs = []
    obj_sim_funcs_dict = {}
    cdd_sim_funcs = []
    cdd_sim_funcs_dict = {}
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs:
            obj_sim_funcs.append(func_pair[0])
            obj_sim_funcs_dict[func_pair[0]] = []
        if func_pair[1] not in obj_sim_funcs_dict[func_pair[0]]:
            obj_sim_funcs_dict[func_pair[0]].append(func_pair[1])
        if func_pair[1] not in cdd_sim_funcs:
            cdd_sim_funcs.append(func_pair[1])
            cdd_sim_funcs_dict[func_pair[1]] = []
        if func_pair[0] not in cdd_sim_funcs_dict[func_pair[1]]:
            cdd_sim_funcs_dict[func_pair[1]].append(func_pair[0])
            
    
    
    all_obj_afcg = get_afcg(obj_sim_funcs, object_graph)
    all_related_funcs = get_related_func(obj_sim_funcs, matched_func_ingraph_list)
    all_cdd_afcg = get_afcg(cdd_sim_funcs, candidate_graph)
    # all_obj_afcg = {}
    # all_related_funcs = {}
    # all_cdd_afcg = {}
    # cdd_related_funcs = get_related_func(cdd_sim_funcs, matched_func_ingraph_list)
    
    func_pair_dict = {}
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] in all_obj_afcg and func_pair[1] in all_cdd_afcg:
            afcg_scale = all_obj_afcg[func_pair[0]] + all_cdd_afcg[func_pair[1]]
            func_pair_dict[str(func_pair)] = afcg_scale
    
    func_pair_sorted_list = sorted(func_pair_dict.items(),  key=lambda d: d[1], reverse=True)
    
    for func_pair_tuple in func_pair_sorted_list:
        func_pair = eval(func_pair_tuple[0])
        
        DONE_FLAG = False
        if len(target_reuse_area_dict) > 0:
            for done_func_pair_str in target_reuse_area_dict[candidate_name]:
                done_cdd_funcs = target_reuse_area_dict[candidate_name][done_func_pair_str][0]["cdd_fcg"]["feature"]
                done_obj_funcs = target_reuse_area_dict[candidate_name][done_func_pair_str][0]["obj_fcg"]["feature"]
                if func_pair[0] in done_obj_funcs and func_pair[1] in done_cdd_funcs:
                    DONE_FLAG = True
                    break
            if DONE_FLAG:
                continue
        
        obj_afcg, all_obj_afcg = get_afcg_one(func_pair[0], obj_sim_funcs, object_graph, all_obj_afcg)
        obj_related_funcs, all_related_funcs = get_related_func_one(func_pair[0], matched_func_ingraph_list, all_related_funcs)
        cdd_afcg, all_cdd_afcg = get_afcg_one(func_pair[1], cdd_sim_funcs, candidate_graph, all_cdd_afcg)
        if len(obj_afcg) >= 2 and len(cdd_afcg) >= 2:
        # if func_pair == ['ZSTD_createCCtxParams','ZSTD_createCCtxParams']:
        #     print("warning")
            if func_pair[1] not in black_list:
                l_max = 0
                lenth_max = [func_pair]
                n = 0
                
                
                node_pair_feature = {}
                fcg_results = {}
                node_pair = func_pair
                
                obj_fcg = get_subgraph(node_pair[0], object_graph)
                cdd_fcg = get_subgraph(node_pair[1], candidate_graph)
                
                obj_num = len(set(obj_fcg["feature"]))
                cdd_num = len(set(cdd_fcg["feature"]))
                
                obj_com_num = obj_sim_num = 0
                for obj_func in set(obj_fcg["feature"]):
                    if obj_func in obj_com_funcs:
                        obj_com_num += 1
                        if obj_func in obj_sim_funcs_dict and list(set(obj_sim_funcs_dict[obj_func]).intersection(set(cdd_fcg["feature"]))) != []:
                            obj_sim_num += 1
                cdd_com_num = cdd_sim_num = 0
                for cdd_func in set(cdd_fcg["feature"]):
                    if cdd_func in cdd_com_funcs:
                        cdd_com_num += 1
                        if cdd_func in cdd_sim_funcs_dict and list(set(cdd_sim_funcs_dict[cdd_func]).intersection(set(obj_fcg["feature"]))) != []:
                            cdd_sim_num += 1
                
                # com_funcs_scale = (len(obj_com_funcs) + len(cdd_com_funcs)) / 2
                # sim_funcs_scale = (len(obj_sim_funcs) + len(cdd_sim_funcs)) / 2
                if obj_com_num == 0 or cdd_com_num == 0:
                    align_rate = 0
                    continue
                elif obj_com_num <= cdd_com_num:
                    align_rate = obj_sim_num / obj_com_num
                else:
                    align_rate = cdd_sim_num / cdd_com_num
                
                
                
                true_num = 0
                
                false_num = 0
                node_pair_feature[str(node_pair)] = {}
                
                node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
                node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
                node_pair_feature[str(node_pair)]["alignment_rate"] = align_rate
                node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_num, cdd_num)
                
                
                
                feature = []
                for func in node_pair_feature[str(node_pair)]["obj_fcg"]["feature"]:
                    func_name = object_name + "|||" + func
                    if func_name in func_embeddings:
                        embed = func_embeddings[func_name][0]
                        true_num += 1
                    else:
                        false_num += 1
                        embed = list(np.array([0.001 for i in range(64)]))
                    feature.append(embed)
                obj_fcg = node_pair_feature[str(node_pair)]["obj_fcg"].copy()
                obj_fcg["embeddings"] = feature
                obj_embedding = embed_by_feat_torch(obj_fcg, gnn)
                feature = []
                for func in node_pair_feature[str(node_pair)]["cdd_fcg"]["feature"]:
                    func_name = candidate_name + "|||" + func
                    if func_name in func_embeddings:
                        embed = func_embeddings[func_name][0]
                        true_num += 1
                    else:
                        false_num += 1
                        embed = list(np.array([0.001 for i in range(64)]))
                    feature.append(embed)
                cdd_fcg = node_pair_feature[str(node_pair)]["cdd_fcg"].copy()
                cdd_fcg["embeddings"] = feature
                # start = time.time()
                cdd_embedding = embed_by_feat_torch(cdd_fcg, gnn)
                # print(time.time() -start)
                gnn_score = F.cosine_similarity(obj_embedding, cdd_embedding, eps=1e-10, dim=1)
                gnn_score = (1 + gnn_score.cpu().detach().numpy()[0]) / 2.0
                node_pair_feature[str(node_pair)]["gnn_score"] = str(gnn_score)
                node_pair_feature[str(node_pair)]["obj_full_fcg_num"] = str(fcgs_num[object_name])
                # node_pair_feature[str(node_pair)]["final_score"] = str(calculate_final_score(gnn_score, node_pair_feature[str(node_pair)]["fcg_scale"][0], node_pair_feature[str(node_pair)]["alignment_num"], fcg_nums[detect_bin]))
                fcg_results[str(node_pair)] = node_pair_feature[str(node_pair)]
                
                align_rate_score = 0.3 * align_rate + 0.7
                if gnn_score * align_rate_score < 0.8:
                    continue
                
                while True:
                    length = [func_pair]
                    l, length, all_related_funcs, all_obj_afcg, all_cdd_afcg = Alignment(func_pair[0], func_pair[1], obj_afcg, cdd_afcg, obj_related_funcs, length, obj_sim_funcs, object_graph, cdd_sim_funcs, candidate_graph, matched_func_ingraph_list, all_related_funcs, all_obj_afcg, all_cdd_afcg)
                    # if l == 0:
                    #     n += 10
                        # if n >= 100:
                        #     break
                        # continue
                    if l > l_max:
                        l_max = l
                        lenth_max = length
                        n = 0
                    else:
                        n += 1
                    if n >= 100 or len(lenth_max) >= 3:
                        break
                
                if len(lenth_max) >= 2:
                    # if len(lenth_max) > 5:
                    #     print("get")
                    alignment_anchor_list.append(lenth_max)
                    
                        
                    # if ((obj_num - cdd_num) > cdd_num) or ((cdd_num - obj_num) > obj_num) or abs(obj_num - cdd_num) > 100 or abs(sim_num-com_num) > 50 or align_rate < 0.5:#  or align_rate < 0.8 or obj_align_rate < 0.05 or cdd_align_rate <  0.05 :
                    #     alignment_temp = 0
                    # elif obj_num < anchor_alignment_dict[str(node_pair)] or cdd_num < anchor_alignment_dict[str(node_pair)]:
                    #     alignment_temp = min( obj_num, cdd_num)
                    # else: 
                    alignment_temp = len(lenth_max)
                    if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2*min(obj_num, cdd_num) and max(obj_num, cdd_num) > 100) or (abs(obj_num - cdd_num) > 200):
                        alignment_temp = 0
                    if (obj_fcg["n_num"] >= 3 and cdd_fcg["n_num"] >= 3 and alignment_temp >= 3) or (obj_num <= 10 and cdd_num <= 10 and alignment_temp >= 2):
                        
                        if alignment_temp > max_alignment_num:
                            max_alignment_num = alignment_temp
                        node_pair_feature[str(node_pair)]["alignment_num"] = alignment_temp
                        
                        score_file_dict = node_pair_feature
                        node_pair_str = str(node_pair)
                        node_alignment_num_score = score_file_dict[node_pair_str]["alignment_num"]
                        node_fcg_scale_pair = score_file_dict[node_pair_str]["fcg_scale"]
                        node_gnn_score = float(score_file_dict[node_pair_str]["gnn_score"])
                        # raw_final_score = float(score_file_dict[node_pair_str]["final_score"])
                        align_rate = float(score_file_dict[node_pair_str]["alignment_rate"])
                        node_fcg_scale_score = (node_fcg_scale_pair[0] + node_fcg_scale_pair[1])/2
                        node_fcg_scale_diff_score = 0.3 * min(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) / max(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) + 0.7
                        
                        if node_alignment_num_score > 0 and node_fcg_scale_pair[0] >=2 and node_fcg_scale_pair[1] >= 2:
                            final_score =  RARM_score(node_alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, align_rate)#, alignment_max, max_fcg)
                            score_file_dict[node_pair_str]["final_score"] = final_score
                            if (final_score >= 0.8 and node_alignment_num_score >= 3) or (final_score >= 0.95 and node_alignment_num_score >= 2):
                                if candidate_name not in target_reuse_lib_dict:
                                    target_reuse_lib_dict.append(candidate_name)
                                if candidate_name not in target_reuse_area_dict:
                                    target_reuse_area_dict[candidate_name] = {}
                                if node_pair_str not in target_reuse_area_dict[candidate_name]:
                                    target_reuse_area_dict[candidate_name][node_pair_str] = []
                                target_reuse_area_dict[candidate_name][node_pair_str].append(score_file_dict[node_pair_str])
                                reuse_flag = True
                                # break
                                # print("final_score: {}".format(final_score))
                                # print("raw_final_score: {}".format(raw_final_score))
                            # elif node_alignment_num_score >=3:
                                # print("final_score: {}".format(final_score))# 成功通过gnn矫正对齐结果
                                # print("raw_final_score: {}".format(raw_final_score))
                    
                # return l, lenth_max
        
    
    if reuse_flag:
        return reuse_flag, target_reuse_area_dict#node_pair_feature, reuse_flag, max_alignment_num, obj_sim_funcs_dict, cdd_sim_funcs_dict
    else:
        return reuse_flag, {}


def get_obj_afcg():
    pass

def get_obj_subgraph():
    pass

def get_cdd_afcg():
    pass

def get_cdd_subgraph():
    pass

def reuse_area_detection_utils_annoy(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict):
    reuse_flag = False
    black_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    max_alignment_num = 0
    
    alignment_anchor_list = []
    
    obj_sim_funcs = []
    obj_sim_funcs_dict = {}
    cdd_sim_funcs = []
    cdd_sim_funcs_dict = {}
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs:
            obj_sim_funcs.append(func_pair[0])
            obj_sim_funcs_dict[func_pair[0]] = []
        if func_pair[1] not in obj_sim_funcs_dict[func_pair[0]]:
            obj_sim_funcs_dict[func_pair[0]].append(func_pair[1])
        if func_pair[1] not in cdd_sim_funcs:
            cdd_sim_funcs.append(func_pair[1])
            cdd_sim_funcs_dict[func_pair[1]] = []
        if func_pair[0] not in cdd_sim_funcs_dict[func_pair[1]]:
            cdd_sim_funcs_dict[func_pair[1]].append(func_pair[0])
            
    
    
    # all_obj_afcg = get_afcg(obj_sim_funcs, object_graph)
    # all_obj_related_funcs = get_related_func(obj_sim_funcs, matched_func_ingraph_list)
    # all_cdd_afcg = get_afcg(cdd_sim_funcs, candidate_graph)
    # all_related_funcs = {}
    # cdd_related_funcs = get_related_func(cdd_sim_funcs, matched_func_ingraph_list)
    target_reuse_lib_dict = []
    target_reuse_area_dict = {}
    for func_pair in matched_func_ingraph_list:
        obj_afcg = get_afcg_one_annoy(func_pair[0], obj_sim_funcs, tar_afcg_dict)
        # obj_related_funcs = obj_sim_funcs_dict[func_pair[0]]#, all_related_funcs = get_related_func_one(func_pair[0], matched_func_ingraph_list, all_related_funcs)
        cdd_afcg = get_afcg_one_annoy(func_pair[1], cdd_sim_funcs, cdd_afcg_dict)
        if len(obj_afcg) > 0 and len(cdd_afcg) > 0:
        # if func_pair == ['ZSTD_createCCtxParams','ZSTD_createCCtxParams']:
        #     print("warning")
            if func_pair[1] not in black_list:
                l_max = 0
                lenth_max = [func_pair]
                n = 0
                
                
                node_pair_feature = {}
                fcg_results = {}
                node_pair = func_pair
                
                obj_fcg = tar_subgraph_dict[func_pair[0]]#get_subgraph(node_pair[0], object_graph)
                cdd_fcg = cdd_subgraph_dict[func_pair[1]]#get_subgraph(node_pair[1], candidate_graph)
                
                obj_num = len(set(obj_fcg["feature"]))
                cdd_num = len(set(cdd_fcg["feature"]))
                
                obj_com_num = obj_sim_num = 0
                for obj_func in set(obj_fcg["feature"]):
                    if obj_func in obj_com_funcs:
                        obj_com_num += 1
                        if obj_func in obj_sim_funcs_dict and list(set(obj_sim_funcs_dict[obj_func]).intersection(set(cdd_fcg["feature"]))) != []:
                            obj_sim_num += 1
                cdd_com_num = cdd_sim_num = 0
                for cdd_func in set(cdd_fcg["feature"]):
                    if cdd_func in cdd_com_funcs:
                        cdd_com_num += 1
                        if cdd_func in cdd_sim_funcs_dict and list(set(cdd_sim_funcs_dict[cdd_func]).intersection(set(obj_fcg["feature"]))) != []:
                            cdd_sim_num += 1
                
                # com_funcs_scale = (len(obj_com_funcs) + len(cdd_com_funcs)) / 2
                # sim_funcs_scale = (len(obj_sim_funcs) + len(cdd_sim_funcs)) / 2
                if obj_com_num == 0 or cdd_com_num == 0:
                    align_rate = 0
                    continue
                elif obj_com_num <= cdd_com_num:
                    align_rate = obj_sim_num / obj_com_num
                else:
                    align_rate = cdd_sim_num / cdd_com_num
                
                
                
                true_num = 0
                
                false_num = 0
                node_pair_feature[str(node_pair)] = {}
                
                node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
                node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
                node_pair_feature[str(node_pair)]["alignment_rate"] = align_rate
                node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_num, cdd_num)
                
                
                
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["obj_fcg"]["feature"]:
                #     func_name = object_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # obj_fcg = node_pair_feature[str(node_pair)]["obj_fcg"].copy()
                # obj_fcg["embeddings"] = feature
                obj_embedding = torch.tensor(obj_fcg["embedding"])#embed_by_feat_torch(obj_fcg, gnn)
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["cdd_fcg"]["feature"]:
                #     func_name = candidate_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # cdd_fcg = node_pair_feature[str(node_pair)]["cdd_fcg"].copy()
                # cdd_fcg["embeddings"] = feature
                # start = time.time()
                cdd_embedding = torch.tensor(cdd_fcg["embedding"])#embed_by_feat_torch(cdd_fcg, gnn)
                # print(time.time() -start)
                gnn_score = F.cosine_similarity(obj_embedding, cdd_embedding, eps=1e-10, dim=1)
                gnn_score = (1 + gnn_score.cpu().detach().numpy()[0]) / 2.0
                node_pair_feature[str(node_pair)]["gnn_score"] = str(gnn_score)
                node_pair_feature[str(node_pair)]["obj_full_fcg_num"] = str(fcgs_num[object_name])
                # node_pair_feature[str(node_pair)]["final_score"] = str(calculate_final_score(gnn_score, node_pair_feature[str(node_pair)]["fcg_scale"][0], node_pair_feature[str(node_pair)]["alignment_num"], fcg_nums[detect_bin]))
                fcg_results[str(node_pair)] = node_pair_feature[str(node_pair)]
                
                align_rate_score = 0.3 * align_rate + 0.7
                if gnn_score * align_rate_score < 0.8:
                    continue
                
                while True:
                    length = [func_pair]
                    l, length = Alignment_v2(func_pair[0], func_pair[1], obj_afcg, cdd_afcg, obj_sim_funcs_dict, length, obj_sim_funcs, object_graph, cdd_sim_funcs, candidate_graph, matched_func_ingraph_list, tar_afcg_dict, cdd_afcg_dict)
                    # if l == 0:
                    #     n += 10
                        # if n >= 100:
                        #     break
                        # continue
                    if l > l_max:
                        l_max = l
                        lenth_max = length
                        n = 0
                    else:
                        n += 1
                    if n >= 100 or len(lenth_max) >= 3:
                        break
                
                if len(lenth_max) >= 2:
                    # if len(lenth_max) > 5:
                    #     print("get")
                    alignment_anchor_list.append(lenth_max)
                    
                        
                    # if ((obj_num - cdd_num) > cdd_num) or ((cdd_num - obj_num) > obj_num) or abs(obj_num - cdd_num) > 100 or abs(sim_num-com_num) > 50 or align_rate < 0.5:#  or align_rate < 0.8 or obj_align_rate < 0.05 or cdd_align_rate <  0.05 :
                    #     alignment_temp = 0
                    # elif obj_num < anchor_alignment_dict[str(node_pair)] or cdd_num < anchor_alignment_dict[str(node_pair)]:
                    #     alignment_temp = min( obj_num, cdd_num)
                    # else: 
                    alignment_temp = len(lenth_max)
                    if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2*min(obj_num, cdd_num) and max(obj_num, cdd_num) > 100) or (abs(obj_num - cdd_num) > 200):
                        alignment_temp = 0
                    if (obj_fcg["n_num"] >= 3 and cdd_fcg["n_num"] >= 3 and alignment_temp >= 3) or (obj_num <= 10 and cdd_num <= 10 and alignment_temp >= 2):
                        
                        if alignment_temp > max_alignment_num:
                            max_alignment_num = alignment_temp
                        node_pair_feature[str(node_pair)]["alignment_num"] = alignment_temp
                        
                        score_file_dict = node_pair_feature
                        node_pair_str = str(node_pair)
                        node_alignment_num_score = score_file_dict[node_pair_str]["alignment_num"]
                        node_fcg_scale_pair = score_file_dict[node_pair_str]["fcg_scale"]
                        node_gnn_score = float(score_file_dict[node_pair_str]["gnn_score"])
                        # raw_final_score = float(score_file_dict[node_pair_str]["final_score"])
                        align_rate = float(score_file_dict[node_pair_str]["alignment_rate"])
                        node_fcg_scale_score = (node_fcg_scale_pair[0] + node_fcg_scale_pair[1])/2
                        node_fcg_scale_diff_score = 0.3 * min(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) / max(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) + 0.7
                        
                        if node_alignment_num_score > 0 and node_fcg_scale_pair[0] >=2 and node_fcg_scale_pair[1] >= 2:
                            final_score =  RARM_score(node_alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, align_rate)#, alignment_max, max_fcg)
                            score_file_dict[node_pair_str]["final_score"] = final_score
                            if (final_score >= 0.8 and node_alignment_num_score >= 3) or (final_score >= 0.95 and node_alignment_num_score >= 2):
                                if candidate_name not in target_reuse_lib_dict:
                                    target_reuse_lib_dict.append(candidate_name)
                                if candidate_name not in target_reuse_area_dict:
                                    target_reuse_area_dict[candidate_name] = {}
                                if node_pair_str not in target_reuse_area_dict[candidate_name]:
                                    target_reuse_area_dict[candidate_name][node_pair_str] = []
                                target_reuse_area_dict[candidate_name][node_pair_str].append(score_file_dict[node_pair_str])
                                reuse_flag = True
                                # break
                                # print("final_score: {}".format(final_score))
                                # print("raw_final_score: {}".format(raw_final_score))
                            # elif node_alignment_num_score >=3:
                                # print("final_score: {}".format(final_score))# 成功通过gnn矫正对齐结果
                                # print("raw_final_score: {}".format(raw_final_score))
                    
                # return l, lenth_max
        
    
    if reuse_flag:
        return reuse_flag, target_reuse_area_dict#node_pair_feature, reuse_flag, max_alignment_num, obj_sim_funcs_dict, cdd_sim_funcs_dict
    else:
        return reuse_flag, {}


def tpl_detection_fast_utils_annoy_v1(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict):
    reuse_flag = False
    black_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    max_alignment_num = 0
    
    alignment_anchor_list = []
    
    obj_sim_funcs = []
    obj_sim_funcs_dict = {}
    cdd_sim_funcs = []
    cdd_sim_funcs_dict = {}
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs:
            obj_sim_funcs.append(func_pair[0])
            obj_sim_funcs_dict[func_pair[0]] = []
        if func_pair[1] not in obj_sim_funcs_dict[func_pair[0]]:
            obj_sim_funcs_dict[func_pair[0]].append(func_pair[1])
        if func_pair[1] not in cdd_sim_funcs:
            cdd_sim_funcs.append(func_pair[1])
            cdd_sim_funcs_dict[func_pair[1]] = []
        if func_pair[0] not in cdd_sim_funcs_dict[func_pair[1]]:
            cdd_sim_funcs_dict[func_pair[1]].append(func_pair[0])
            
    
    
    # all_obj_afcg = get_afcg(obj_sim_funcs, object_graph)
    # all_obj_related_funcs = get_related_func(obj_sim_funcs, matched_func_ingraph_list)
    # all_cdd_afcg = get_afcg(cdd_sim_funcs, candidate_graph)
    # all_related_funcs = {}
    # cdd_related_funcs = get_related_func(cdd_sim_funcs, matched_func_ingraph_list)
    
    for func_pair in matched_func_ingraph_list:
        # if func_pair[0] == "qlz_compress3":
        #     print("wrning")
        obj_afcg = get_afcg_one_annoy(func_pair[0], obj_sim_funcs, tar_afcg_dict)
        # obj_related_funcs = obj_sim_funcs_dict[func_pair[0]]#, all_related_funcs = get_related_func_one(func_pair[0], matched_func_ingraph_list, all_related_funcs)
        cdd_afcg = get_afcg_one_annoy(func_pair[1], cdd_sim_funcs, cdd_afcg_dict)
        if len(obj_afcg) > 0 and len(cdd_afcg) > 0:
        # if func_pair == ['ZSTD_createCCtxParams','ZSTD_createCCtxParams']:
        #     print("warning")
            if func_pair[1] not in black_list:
                l_max = 0
                lenth_max = [func_pair]
                n = 0
                
                
                node_pair_feature = {}
                fcg_results = {}
                node_pair = func_pair
                
                obj_fcg = tar_subgraph_dict[func_pair[0]]#get_subgraph(node_pair[0], object_graph)
                cdd_fcg = cdd_subgraph_dict[func_pair[1]]#get_subgraph(node_pair[1], candidate_graph)
                
                obj_embedding = torch.tensor(obj_fcg["embedding"])#embed_by_feat_torch(obj_fcg, gnn)
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["cdd_fcg"]["feature"]:
                #     func_name = candidate_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # cdd_fcg = node_pair_feature[str(node_pair)]["cdd_fcg"].copy()
                # cdd_fcg["embeddings"] = feature
                # start = time.time()
                cdd_embedding = torch.tensor(cdd_fcg["embedding"])#embed_by_feat_torch(cdd_fcg, gnn)
                # print(time.time() -start)
                gnn_score = F.cosine_similarity(obj_embedding, cdd_embedding, eps=1e-10, dim=1)
                gnn_score = (1 + gnn_score.cpu().detach().numpy()[0]) / 2.0
                
                if gnn_score < 0.8:
                    continue
                
                
                obj_num = len(set(obj_fcg["feature"]))
                cdd_num = len(set(cdd_fcg["feature"]))
                
                obj_com_num = obj_sim_num = 0
                for obj_func in set(obj_fcg["feature"]):
                    if obj_func in obj_com_funcs:
                        obj_com_num += 1
                        if obj_func in obj_sim_funcs_dict and list(set(obj_sim_funcs_dict[obj_func]).intersection(set(cdd_fcg["feature"]))) != []:
                            obj_sim_num += 1
                cdd_com_num = cdd_sim_num = 0
                for cdd_func in set(cdd_fcg["feature"]):
                    if cdd_func in cdd_com_funcs:
                        cdd_com_num += 1
                        if cdd_func in cdd_sim_funcs_dict and list(set(cdd_sim_funcs_dict[cdd_func]).intersection(set(obj_fcg["feature"]))) != []:
                            cdd_sim_num += 1
                
                # com_funcs_scale = (len(obj_com_funcs) + len(cdd_com_funcs)) / 2
                # sim_funcs_scale = (len(obj_sim_funcs) + len(cdd_sim_funcs)) / 2
                if obj_com_num == 0 or cdd_com_num == 0:
                    align_rate = 0
                    continue
                else:
                    align_rate = (obj_sim_num / obj_com_num + cdd_sim_num / cdd_com_num) / 2
                # elif obj_com_num <= cdd_com_num:
                #     align_rate = obj_sim_num / obj_com_num
                # else:
                #     align_rate = cdd_sim_num / cdd_com_num
                
                
                
                true_num = 0
                
                false_num = 0
                node_pair_feature[str(node_pair)] = {}
                
                node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
                node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
                node_pair_feature[str(node_pair)]["alignment_rate"] = align_rate
                node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_num, cdd_num)
                
                
                
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["obj_fcg"]["feature"]:
                #     func_name = object_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # obj_fcg = node_pair_feature[str(node_pair)]["obj_fcg"].copy()
                # obj_fcg["embeddings"] = feature
                
                node_pair_feature[str(node_pair)]["gnn_score"] = str(gnn_score)
                node_pair_feature[str(node_pair)]["obj_full_fcg_num"] = str(fcgs_num[object_name])
                # node_pair_feature[str(node_pair)]["final_score"] = str(calculate_final_score(gnn_score, node_pair_feature[str(node_pair)]["fcg_scale"][0], node_pair_feature[str(node_pair)]["alignment_num"], fcg_nums[detect_bin]))
                fcg_results[str(node_pair)] = node_pair_feature[str(node_pair)]
                
                align_rate_score = align_rate#0.3 * align_rate + 0.7
                if gnn_score * align_rate_score < 0.8:
                    continue
                
                while True:
                    length = [func_pair]
                    l, length = Alignment_v2(func_pair[0], func_pair[1], obj_afcg, cdd_afcg, obj_sim_funcs_dict, length, obj_sim_funcs, object_graph, cdd_sim_funcs, candidate_graph, matched_func_ingraph_list, tar_afcg_dict, cdd_afcg_dict)
                    # if l == 0:
                    #     n += 10
                        # if n >= 100:
                        #     break
                        # continue
                    if l > l_max:
                        l_max = l
                        lenth_max = length
                        n = 0
                    else:
                        n += 1
                    if n >= 100 or len(lenth_max) >= 3:
                        break
                
                if len(lenth_max) >= 2:
                    # if len(lenth_max) > 5:
                    #     print("get")
                    alignment_anchor_list.append(lenth_max)
                    
                        
                    # if ((obj_num - cdd_num) > cdd_num) or ((cdd_num - obj_num) > obj_num) or abs(obj_num - cdd_num) > 100 or abs(sim_num-com_num) > 50 or align_rate < 0.5:#  or align_rate < 0.8 or obj_align_rate < 0.05 or cdd_align_rate <  0.05 :
                    #     alignment_temp = 0
                    # elif obj_num < anchor_alignment_dict[str(node_pair)] or cdd_num < anchor_alignment_dict[str(node_pair)]:
                    #     alignment_temp = min( obj_num, cdd_num)
                    # else: 
                    alignment_temp = len(lenth_max)
                    if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2*min(obj_num, cdd_num)) or (abs(obj_num - cdd_num) > 100):
                    # if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2*min(obj_num, cdd_num) and max(obj_num, cdd_num) > 50) or (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 4*min(obj_num, cdd_num)) or (abs(obj_num - cdd_num) > 100):
                        alignment_temp = 0
                    if (obj_fcg["n_num"] >= 3 and cdd_fcg["n_num"] >= 3 and alignment_temp >= 3):# or (obj_num <= 10 and cdd_num <= 10 and alignment_temp >= 2):
                        
                        if alignment_temp > max_alignment_num:
                            max_alignment_num = alignment_temp
                        node_pair_feature[str(node_pair)]["alignment_num"] = alignment_temp
                        target_reuse_lib_dict = []
                        target_reuse_area_dict = {}
                        score_file_dict = node_pair_feature
                        node_pair_str = str(node_pair)
                        node_alignment_num_score = score_file_dict[node_pair_str]["alignment_num"]
                        node_fcg_scale_pair = score_file_dict[node_pair_str]["fcg_scale"]
                        node_gnn_score = float(score_file_dict[node_pair_str]["gnn_score"])
                        # raw_final_score = float(score_file_dict[node_pair_str]["final_score"])
                        align_rate = float(score_file_dict[node_pair_str]["alignment_rate"])
                        node_fcg_scale_score = (node_fcg_scale_pair[0] + node_fcg_scale_pair[1])/2
                        node_fcg_scale_diff_score = 0.3 * min(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) / max(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) + 0.7
                        
                        if node_alignment_num_score > 0 and node_fcg_scale_pair[0] >=2 and node_fcg_scale_pair[1] >= 2:
                            final_score =  RARM_score(node_alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, align_rate)#, alignment_max, max_fcg)
                            score_file_dict[node_pair_str]["final_score"] = final_score
                            if (final_score >= 0.8 and node_alignment_num_score >= 3) or (final_score >= 0.95 and node_alignment_num_score >= 2):
                                if candidate_name not in target_reuse_lib_dict:
                                    target_reuse_lib_dict.append(candidate_name)
                                if candidate_name not in target_reuse_area_dict:
                                    target_reuse_area_dict[candidate_name] = {}
                                if node_pair_str not in target_reuse_area_dict[candidate_name]:
                                    target_reuse_area_dict[candidate_name][node_pair_str] = []
                                target_reuse_area_dict[candidate_name][node_pair_str].append(score_file_dict[node_pair_str])
                                reuse_flag = True
                                break
                                # print("final_score: {}".format(final_score))
                                # print("raw_final_score: {}".format(raw_final_score))
                            # elif node_alignment_num_score >=3:
                                # print("final_score: {}".format(final_score))# 成功通过gnn矫正对齐结果
                                # print("raw_final_score: {}".format(raw_final_score))
                    
                # return l, lenth_max
        
    
    if reuse_flag:
        return reuse_flag, target_reuse_area_dict#node_pair_feature, reuse_flag, max_alignment_num, obj_sim_funcs_dict, cdd_sim_funcs_dict
    else:
        return reuse_flag, {}



def tpl_detection_fast_utils_annoy_without_align(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict):
    reuse_flag = False
    black_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    max_alignment_num = 0
    
    alignment_anchor_list = []
    
    obj_sim_funcs = []
    obj_sim_funcs_dict = {}
    cdd_sim_funcs = []
    cdd_sim_funcs_dict = {}
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs:
            obj_sim_funcs.append(func_pair[0])
            obj_sim_funcs_dict[func_pair[0]] = []
        if func_pair[1] not in obj_sim_funcs_dict[func_pair[0]]:
            obj_sim_funcs_dict[func_pair[0]].append(func_pair[1])
        if func_pair[1] not in cdd_sim_funcs:
            cdd_sim_funcs.append(func_pair[1])
            cdd_sim_funcs_dict[func_pair[1]] = []
        if func_pair[0] not in cdd_sim_funcs_dict[func_pair[1]]:
            cdd_sim_funcs_dict[func_pair[1]].append(func_pair[0])
            
    
    
    # all_obj_afcg = get_afcg(obj_sim_funcs, object_graph)
    # all_obj_related_funcs = get_related_func(obj_sim_funcs, matched_func_ingraph_list)
    # all_cdd_afcg = get_afcg(cdd_sim_funcs, candidate_graph)
    # all_related_funcs = {}
    # cdd_related_funcs = get_related_func(cdd_sim_funcs, matched_func_ingraph_list)
    target_reuse_lib_dict = []
    target_reuse_area_dict = {}
    for func_pair in matched_func_ingraph_list:
        obj_afcg = get_afcg_one_annoy(func_pair[0], obj_sim_funcs, tar_afcg_dict)
        # obj_related_funcs = obj_sim_funcs_dict[func_pair[0]]#, all_related_funcs = get_related_func_one(func_pair[0], matched_func_ingraph_list, all_related_funcs)
        cdd_afcg = get_afcg_one_annoy(func_pair[1], cdd_sim_funcs, cdd_afcg_dict)
        if func_pair == ['BZ2_bzDecompress','ZSTD_compressEnd']:
            print("warning")
        if len(obj_afcg) > 0 and len(cdd_afcg) > 0:
            if func_pair[1] not in black_list:
                
                
                
                l_max = 0
                lenth_max = [func_pair]
                n = 0
                
                
                node_pair_feature = {}
                fcg_results = {}
                node_pair = func_pair
                
                obj_fcg = tar_subgraph_dict[func_pair[0]]#get_subgraph(node_pair[0], object_graph)
                cdd_fcg = cdd_subgraph_dict[func_pair[1]]#get_subgraph(node_pair[1], candidate_graph)
                
                
                obj_embedding = torch.tensor(obj_fcg["embedding"])#embed_by_feat_torch(obj_fcg, gnn)
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["cdd_fcg"]["feature"]:
                #     func_name = candidate_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # cdd_fcg = node_pair_feature[str(node_pair)]["cdd_fcg"].copy()
                # cdd_fcg["embeddings"] = feature
                # start = time.time()
                cdd_embedding = torch.tensor(cdd_fcg["embedding"])#embed_by_feat_torch(cdd_fcg, gnn)
                # print(time.time() -start)
                gnn_score = F.cosine_similarity(obj_embedding, cdd_embedding, eps=1e-10, dim=1)
                gnn_score = (1 + gnn_score.cpu().detach().numpy()[0]) / 2.0
                
                if gnn_score < 0.8:
                    continue
                
                
                obj_num = len(set(obj_fcg["feature"]))
                cdd_num = len(set(cdd_fcg["feature"]))
                
                obj_com_num = obj_sim_num = 0
                for obj_func in set(obj_fcg["feature"]):
                    if obj_func in obj_com_funcs:
                        obj_com_num += 1
                        if obj_func in obj_sim_funcs_dict and list(set(obj_sim_funcs_dict[obj_func]).intersection(set(cdd_fcg["feature"]))) != []:
                            obj_sim_num += 1
                cdd_com_num = cdd_sim_num = 0
                for cdd_func in set(cdd_fcg["feature"]):
                    if cdd_func in cdd_com_funcs:
                        cdd_com_num += 1
                        if cdd_func in cdd_sim_funcs_dict and list(set(cdd_sim_funcs_dict[cdd_func]).intersection(set(obj_fcg["feature"]))) != []:
                            cdd_sim_num += 1
                
                # com_funcs_scale = (len(obj_com_funcs) + len(cdd_com_funcs)) / 2
                # sim_funcs_scale = (len(obj_sim_funcs) + len(cdd_sim_funcs)) / 2
                if obj_com_num == 0 or cdd_com_num == 0:
                    align_rate = 0
                    continue
                elif obj_com_num <= cdd_com_num:
                    align_rate = obj_sim_num / obj_com_num
                else:
                    align_rate = cdd_sim_num / cdd_com_num
                
                align_rate = 1
                
                true_num = 0
                
                false_num = 0
                node_pair_feature[str(node_pair)] = {}
                
                node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
                node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
                node_pair_feature[str(node_pair)]["alignment_rate"] = align_rate
                node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_num, cdd_num)
                
                
                
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["obj_fcg"]["feature"]:
                #     func_name = object_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # obj_fcg = node_pair_feature[str(node_pair)]["obj_fcg"].copy()
                # obj_fcg["embeddings"] = feature
                
                node_pair_feature[str(node_pair)]["gnn_score"] = str(gnn_score)
                node_pair_feature[str(node_pair)]["obj_full_fcg_num"] = str(fcgs_num[object_name])
                # node_pair_feature[str(node_pair)]["final_score"] = str(calculate_final_score(gnn_score, node_pair_feature[str(node_pair)]["fcg_scale"][0], node_pair_feature[str(node_pair)]["alignment_num"], fcg_nums[detect_bin]))
                fcg_results[str(node_pair)] = node_pair_feature[str(node_pair)]
                
                align_rate_score = align_rate#0.3 * align_rate + 0.7
                if gnn_score * align_rate_score < 0.8:
                    continue
                
                # while True:
                #     length = [func_pair]
                #     l, length = Alignment_v2(func_pair[0], func_pair[1], obj_afcg, cdd_afcg, obj_sim_funcs_dict, length, obj_sim_funcs, object_graph, cdd_sim_funcs, candidate_graph, matched_func_ingraph_list, tar_afcg_dict, cdd_afcg_dict)
                #     # if l == 0:
                #     #     n += 10
                #         # if n >= 100:
                #         #     break
                #         # continue
                #     if l > l_max:
                #         l_max = l
                #         lenth_max = length
                #         n = 0
                #     else:
                #         n += 1
                #     if n >= 100 or len(lenth_max) >= 3:
                #         break
                
                # if len(lenth_max) >= 2:
                    # if len(lenth_max) > 5:
                    #     print("get")
                # alignment_anchor_list.append(lenth_max)
                
                    
                # if ((obj_num - cdd_num) > cdd_num) or ((cdd_num - obj_num) > obj_num) or abs(obj_num - cdd_num) > 100 or abs(sim_num-com_num) > 50 or align_rate < 0.5:#  or align_rate < 0.8 or obj_align_rate < 0.05 or cdd_align_rate <  0.05 :
                #     alignment_temp = 0
                # elif obj_num < anchor_alignment_dict[str(node_pair)] or cdd_num < anchor_alignment_dict[str(node_pair)]:
                #     alignment_temp = min( obj_num, cdd_num)
                # else: 
                # alignment_temp = len(lenth_max)
                # if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2*min(obj_num, cdd_num) and max(obj_num, cdd_num) > 100) or (abs(obj_num - cdd_num) > 200):
                    # alignment_temp = 0
                # if (obj_fcg["n_num"] >= 3 and cdd_fcg["n_num"] >= 3 and alignment_temp >= 3) or (obj_num <= 10 and cdd_num <= 10 and alignment_temp >= 2):
                    
                    # if alignment_temp > max_alignment_num:
                        # max_alignment_num = alignment_temp
                    # node_pair_feature[str(node_pair)]["alignment_num"] = alignment_temp
                    
                score_file_dict = node_pair_feature
                node_pair_str = str(node_pair)
                # node_alignment_num_score = score_file_dict[node_pair_str]["alignment_num"]
                node_fcg_scale_pair = score_file_dict[node_pair_str]["fcg_scale"]
                node_gnn_score = float(score_file_dict[node_pair_str]["gnn_score"])
                # raw_final_score = float(score_file_dict[node_pair_str]["final_score"])
                align_rate = float(score_file_dict[node_pair_str]["alignment_rate"])
                node_fcg_scale_score = (node_fcg_scale_pair[0] + node_fcg_scale_pair[1])/2
                node_fcg_scale_diff_score = 0.3 * min(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) / max(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) + 0.7
                node_alignment_num_score = 3
                # if node_alignment_num_score > 0 and node_fcg_scale_pair[0] >=2 and node_fcg_scale_pair[1] >= 2:
                final_score =  RARM_score(node_alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, align_rate)#, alignment_max, max_fcg)
                score_file_dict[node_pair_str]["final_score"] = final_score
                if (final_score >= 0.8 and node_alignment_num_score >= 3) or (final_score >= 0.95 and node_alignment_num_score >= 2):
                    if candidate_name not in target_reuse_lib_dict:
                        target_reuse_lib_dict.append(candidate_name)
                    if candidate_name not in target_reuse_area_dict:
                        target_reuse_area_dict[candidate_name] = {}
                    if node_pair_str not in target_reuse_area_dict[candidate_name]:
                        target_reuse_area_dict[candidate_name][node_pair_str] = []
                    target_reuse_area_dict[candidate_name][node_pair_str].append(score_file_dict[node_pair_str])
                    reuse_flag = True
                    break
                    # print("final_score: {}".format(final_score))
                    # print("raw_final_score: {}".format(raw_final_score))
                    # elif node_alignment_num_score >=3:
                        # print("final_score: {}".format(final_score))# 成功通过gnn矫正对齐结果
                        # print("raw_final_score: {}".format(raw_final_score))
            
            # return l, lenth_max
        
    
    if reuse_flag:
        return reuse_flag, target_reuse_area_dict#node_pair_feature, reuse_flag, max_alignment_num, obj_sim_funcs_dict, cdd_sim_funcs_dict
    else:
        return reuse_flag, {}




def tpl_detection_fast_utils_annoy_without_gnn(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict):
    reuse_flag = False
    black_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    max_alignment_num = 0
    
    alignment_anchor_list = []
    
    obj_sim_funcs = []
    obj_sim_funcs_dict = {}
    cdd_sim_funcs = []
    cdd_sim_funcs_dict = {}
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs:
            obj_sim_funcs.append(func_pair[0])
            obj_sim_funcs_dict[func_pair[0]] = []
        if func_pair[1] not in obj_sim_funcs_dict[func_pair[0]]:
            obj_sim_funcs_dict[func_pair[0]].append(func_pair[1])
        if func_pair[1] not in cdd_sim_funcs:
            cdd_sim_funcs.append(func_pair[1])
            cdd_sim_funcs_dict[func_pair[1]] = []
        if func_pair[0] not in cdd_sim_funcs_dict[func_pair[1]]:
            cdd_sim_funcs_dict[func_pair[1]].append(func_pair[0])
            
    
    
    # all_obj_afcg = get_afcg(obj_sim_funcs, object_graph)
    # all_obj_related_funcs = get_related_func(obj_sim_funcs, matched_func_ingraph_list)
    # all_cdd_afcg = get_afcg(cdd_sim_funcs, candidate_graph)
    # all_related_funcs = {}
    # cdd_related_funcs = get_related_func(cdd_sim_funcs, matched_func_ingraph_list)
    target_reuse_lib_dict = []
    target_reuse_area_dict = {}
    for func_pair in matched_func_ingraph_list:
        obj_afcg = get_afcg_one_annoy(func_pair[0], obj_sim_funcs, tar_afcg_dict)
        # obj_related_funcs = obj_sim_funcs_dict[func_pair[0]]#, all_related_funcs = get_related_func_one(func_pair[0], matched_func_ingraph_list, all_related_funcs)
        cdd_afcg = get_afcg_one_annoy(func_pair[1], cdd_sim_funcs, cdd_afcg_dict)
        # if func_pair == ['BZ2_bzDecompress','ZSTD_compressEnd']:
        #     print("warning")
        if len(obj_afcg) > 0 and len(cdd_afcg) > 0:
            if func_pair[1] not in black_list:
                
                
                
                l_max = 0
                lenth_max = [func_pair]
                n = 0
                
                
                node_pair_feature = {}
                fcg_results = {}
                node_pair = func_pair
                
                obj_fcg = tar_subgraph_dict[func_pair[0]]#get_subgraph(node_pair[0], object_graph)
                cdd_fcg = cdd_subgraph_dict[func_pair[1]]#get_subgraph(node_pair[1], candidate_graph)
                
                
                # obj_embedding = torch.tensor(obj_fcg["embedding"])#embed_by_feat_torch(obj_fcg, gnn)
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["cdd_fcg"]["feature"]:
                #     func_name = candidate_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # cdd_fcg = node_pair_feature[str(node_pair)]["cdd_fcg"].copy()
                # cdd_fcg["embeddings"] = feature
                # start = time.time()
                # cdd_embedding = torch.tensor(cdd_fcg["embedding"])#embed_by_feat_torch(cdd_fcg, gnn)
                # print(time.time() -start)
                # gnn_score = F.cosine_similarity(obj_embedding, cdd_embedding, eps=1e-10, dim=1)
                # gnn_score = (1 + gnn_score.cpu().detach().numpy()[0]) / 2.0
                
                # if gnn_score < 0.8:
                #     continue
                
                
                obj_num = len(set(obj_fcg["feature"]))
                cdd_num = len(set(cdd_fcg["feature"]))
                
                obj_com_num = obj_sim_num = 0
                for obj_func in set(obj_fcg["feature"]):
                    if obj_func in obj_com_funcs:
                        obj_com_num += 1
                        if obj_func in obj_sim_funcs_dict and list(set(obj_sim_funcs_dict[obj_func]).intersection(set(cdd_fcg["feature"]))) != []:
                            obj_sim_num += 1
                cdd_com_num = cdd_sim_num = 0
                for cdd_func in set(cdd_fcg["feature"]):
                    if cdd_func in cdd_com_funcs:
                        cdd_com_num += 1
                        if cdd_func in cdd_sim_funcs_dict and list(set(cdd_sim_funcs_dict[cdd_func]).intersection(set(obj_fcg["feature"]))) != []:
                            cdd_sim_num += 1
                
                # com_funcs_scale = (len(obj_com_funcs) + len(cdd_com_funcs)) / 2
                # sim_funcs_scale = (len(obj_sim_funcs) + len(cdd_sim_funcs)) / 2
                if obj_com_num == 0 or cdd_com_num == 0:
                    align_rate = 0
                    continue
                elif obj_com_num <= cdd_com_num:
                    align_rate = obj_sim_num / obj_com_num
                else:
                    align_rate = cdd_sim_num / cdd_com_num
                
                # align_rate = 1
                
                true_num = 0
                
                false_num = 0
                node_pair_feature[str(node_pair)] = {}
                
                node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
                node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
                node_pair_feature[str(node_pair)]["alignment_rate"] = align_rate
                node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_num, cdd_num)
                
                
                
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["obj_fcg"]["feature"]:
                #     func_name = object_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # obj_fcg = node_pair_feature[str(node_pair)]["obj_fcg"].copy()
                # obj_fcg["embeddings"] = feature
                
                # node_pair_feature[str(node_pair)]["gnn_score"] = str(gnn_score)
                node_pair_feature[str(node_pair)]["obj_full_fcg_num"] = str(fcgs_num[object_name])
                # node_pair_feature[str(node_pair)]["final_score"] = str(calculate_final_score(gnn_score, node_pair_feature[str(node_pair)]["fcg_scale"][0], node_pair_feature[str(node_pair)]["alignment_num"], fcg_nums[detect_bin]))
                fcg_results[str(node_pair)] = node_pair_feature[str(node_pair)]
                
                align_rate_score = 0.3 * align_rate + 0.7
                if align_rate_score < 0.8:
                    continue
                
                while True:
                    length = [func_pair]
                    l, length = Alignment_v2(func_pair[0], func_pair[1], obj_afcg, cdd_afcg, obj_sim_funcs_dict, length, obj_sim_funcs, object_graph, cdd_sim_funcs, candidate_graph, matched_func_ingraph_list, tar_afcg_dict, cdd_afcg_dict)
                    # if l == 0:
                    #     n += 10
                        # if n >= 100:
                        #     break
                        # continue
                    if l > l_max:
                        l_max = l
                        lenth_max = length
                        n = 0
                    else:
                        n += 1
                    if n >= 100 or len(lenth_max) >= 3:
                        break
                
                if len(lenth_max) >= 2:
                    # if len(lenth_max) > 5:
                    #     print("get")
                    alignment_anchor_list.append(lenth_max)
                    
                        
                    # if ((obj_num - cdd_num) > cdd_num) or ((cdd_num - obj_num) > obj_num) or abs(obj_num - cdd_num) > 100 or abs(sim_num-com_num) > 50 or align_rate < 0.5:#  or align_rate < 0.8 or obj_align_rate < 0.05 or cdd_align_rate <  0.05 :
                    #     alignment_temp = 0
                    # elif obj_num < anchor_alignment_dict[str(node_pair)] or cdd_num < anchor_alignment_dict[str(node_pair)]:
                    #     alignment_temp = min( obj_num, cdd_num)
                    # else: 
                    alignment_temp = len(lenth_max)
                    if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2*min(obj_num, cdd_num) and max(obj_num, cdd_num) > 100) or (abs(obj_num - cdd_num) > 200):
                        alignment_temp = 0
                    if (obj_fcg["n_num"] >= 3 and cdd_fcg["n_num"] >= 3 and alignment_temp >= 3) or (obj_num <= 10 and cdd_num <= 10 and alignment_temp >= 2):
                        
                        if alignment_temp > max_alignment_num:
                            max_alignment_num = alignment_temp
                        node_pair_feature[str(node_pair)]["alignment_num"] = alignment_temp
                        
                        score_file_dict = node_pair_feature
                        node_pair_str = str(node_pair)
                        node_alignment_num_score = score_file_dict[node_pair_str]["alignment_num"]
                        node_fcg_scale_pair = score_file_dict[node_pair_str]["fcg_scale"]
                        # node_gnn_score = float(score_file_dict[node_pair_str]["gnn_score"])
                        # raw_final_score = float(score_file_dict[node_pair_str]["final_score"])
                        align_rate = float(score_file_dict[node_pair_str]["alignment_rate"])
                        node_fcg_scale_score = (node_fcg_scale_pair[0] + node_fcg_scale_pair[1])/2
                        node_fcg_scale_diff_score = 0.3 * min(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) / max(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) + 0.7
                        
                        if node_alignment_num_score > 0 and node_fcg_scale_pair[0] >=2 and node_fcg_scale_pair[1] >= 2:
                            # final_score =  RARM_score(node_alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, align_rate)#, alignment_max, max_fcg)
                            # score_file_dict[node_pair_str]["final_score"] = final_score
                            if (node_alignment_num_score >= 3):# or (final_score >= 0.95 and node_alignment_num_score >= 2):
                                if candidate_name not in target_reuse_lib_dict:
                                    target_reuse_lib_dict.append(candidate_name)
                                if candidate_name not in target_reuse_area_dict:
                                    target_reuse_area_dict[candidate_name] = {}
                                if node_pair_str not in target_reuse_area_dict[candidate_name]:
                                    target_reuse_area_dict[candidate_name][node_pair_str] = []
                                target_reuse_area_dict[candidate_name][node_pair_str].append(score_file_dict[node_pair_str])
                                reuse_flag = True
                                break
                                # print("final_score: {}".format(final_score))
                                # print("raw_final_score: {}".format(raw_final_score))
                            # elif node_alignment_num_score >=3:
                                # print("final_score: {}".format(final_score))# 成功通过gnn矫正对齐结果
                                # print("raw_final_score: {}".format(raw_final_score))
                    
                # return l, lenth_max
        
    
    if reuse_flag:
        return reuse_flag, target_reuse_area_dict#node_pair_feature, reuse_flag, max_alignment_num, obj_sim_funcs_dict, cdd_sim_funcs_dict
    else:
        return reuse_flag, {}


def tpl_detection_fast_utils_annoy_v2(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict):
    reuse_flag = False
    black_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    max_alignment_num = 0
    
    alignment_anchor_list = []
    
    alignment_tred = 3
    
    obj_sim_funcs = []
    obj_sim_funcs_dict = {}
    cdd_sim_funcs = []
    cdd_sim_funcs_dict = {}
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs:
            obj_sim_funcs.append(func_pair[0])
            obj_sim_funcs_dict[func_pair[0]] = []
        if func_pair[1] not in obj_sim_funcs_dict[func_pair[0]]:
            obj_sim_funcs_dict[func_pair[0]].append(func_pair[1])
        if func_pair[1] not in cdd_sim_funcs:
            cdd_sim_funcs.append(func_pair[1])
            cdd_sim_funcs_dict[func_pair[1]] = []
        if func_pair[0] not in cdd_sim_funcs_dict[func_pair[1]]:
            cdd_sim_funcs_dict[func_pair[1]].append(func_pair[0])
            
    
    
    # all_obj_afcg = get_afcg(obj_sim_funcs, object_graph)
    # all_obj_related_funcs = get_related_func(obj_sim_funcs, matched_func_ingraph_list)
    # all_cdd_afcg = get_afcg(cdd_sim_funcs, candidate_graph)
    # all_related_funcs = {}
    # cdd_related_funcs = get_related_func(cdd_sim_funcs, matched_func_ingraph_list)
    target_reuse_lib_dict = []
    target_reuse_area_dict = {}
    for func_pair in matched_func_ingraph_list:
        obj_afcg = get_afcg_one_annoy(func_pair[0], obj_sim_funcs, tar_afcg_dict)
        # obj_related_funcs = obj_sim_funcs_dict[func_pair[0]]#, all_related_funcs = get_related_func_one(func_pair[0], matched_func_ingraph_list, all_related_funcs)
        cdd_afcg = get_afcg_one_annoy(func_pair[1], cdd_sim_funcs, cdd_afcg_dict)
        # if func_pair == ['BZ2_bzDecompress','BZ2_bzDecompress']:
        #     print("warning")
        if len(obj_afcg) > 0 and len(cdd_afcg) > 0:
            if func_pair[1] not in black_list:
                
                
                
                l_max = 0
                lenth_max = [func_pair]
                n = 0
                
                
                node_pair_feature = {}
                fcg_results = {}
                node_pair = func_pair
                
                obj_fcg = tar_subgraph_dict[func_pair[0]]#get_subgraph(node_pair[0], object_graph)
                cdd_fcg = cdd_subgraph_dict[func_pair[1]]#get_subgraph(node_pair[1], candidate_graph)
                
                
                obj_embedding = torch.tensor(obj_fcg["embedding"])#embed_by_feat_torch(obj_fcg, gnn)
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["cdd_fcg"]["feature"]:
                #     func_name = candidate_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # cdd_fcg = node_pair_feature[str(node_pair)]["cdd_fcg"].copy()
                # cdd_fcg["embeddings"] = feature
                # start = time.time()
                cdd_embedding = torch.tensor(cdd_fcg["embedding"])#embed_by_feat_torch(cdd_fcg, gnn)
                # print(time.time() -start)
                gnn_score = F.cosine_similarity(obj_embedding, cdd_embedding, eps=1e-10, dim=1)
                gnn_score = (1 + gnn_score.cpu().detach().numpy()[0]) / 2.0
                
                if gnn_score < 0.8:
                    continue
                
                
                obj_num = len(set(obj_fcg["feature"]))
                cdd_num = len(set(cdd_fcg["feature"]))
                
                obj_com_num = obj_sim_num = 0
                for obj_func in set(obj_fcg["feature"]):
                    if obj_func in obj_com_funcs:
                        obj_com_num += 1
                        if obj_func in obj_sim_funcs_dict and list(set(obj_sim_funcs_dict[obj_func]).intersection(set(cdd_fcg["feature"]))) != []:
                            obj_sim_num += 1
                cdd_com_num = cdd_sim_num = 0
                for cdd_func in set(cdd_fcg["feature"]):
                    if cdd_func in cdd_com_funcs:
                        cdd_com_num += 1
                        if cdd_func in cdd_sim_funcs_dict and list(set(cdd_sim_funcs_dict[cdd_func]).intersection(set(obj_fcg["feature"]))) != []:
                            cdd_sim_num += 1
                
                # com_funcs_scale = (len(obj_com_funcs) + len(cdd_com_funcs)) / 2
                # sim_funcs_scale = (len(obj_sim_funcs) + len(cdd_sim_funcs)) / 2
                if obj_com_num == 0 or cdd_com_num == 0:
                    align_rate = 0
                    continue
                elif obj_com_num <= cdd_com_num:
                    align_rate = obj_sim_num / obj_com_num
                else:
                    align_rate = cdd_sim_num / cdd_com_num
                
                # align_rate = 1
                
                true_num = 0
                
                false_num = 0
                node_pair_feature[str(node_pair)] = {}
                
                node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
                node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
                node_pair_feature[str(node_pair)]["alignment_rate"] = align_rate
                node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_num, cdd_num)
                
                
                
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["obj_fcg"]["feature"]:
                #     func_name = object_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # obj_fcg = node_pair_feature[str(node_pair)]["obj_fcg"].copy()
                # obj_fcg["embeddings"] = feature
                
                node_pair_feature[str(node_pair)]["gnn_score"] = str(gnn_score)
                node_pair_feature[str(node_pair)]["obj_full_fcg_num"] = str(fcgs_num[object_name])
                # node_pair_feature[str(node_pair)]["final_score"] = str(calculate_final_score(gnn_score, node_pair_feature[str(node_pair)]["fcg_scale"][0], node_pair_feature[str(node_pair)]["alignment_num"], fcg_nums[detect_bin]))
                fcg_results[str(node_pair)] = node_pair_feature[str(node_pair)]
                
                align_rate_score = 0.3 * align_rate + 0.7
                if gnn_score * align_rate_score < 0.8:
                    continue
                
                while True:
                    length = [func_pair]
                    l, length = Alignment_v2(func_pair[0], func_pair[1], obj_afcg, cdd_afcg, obj_sim_funcs_dict, length, obj_sim_funcs, object_graph, cdd_sim_funcs, candidate_graph, matched_func_ingraph_list, tar_afcg_dict, cdd_afcg_dict)
                    # if l == 0:
                    #     n += 10
                        # if n >= 100:
                        #     break
                        # continue
                    if l > l_max:
                        l_max = l
                        lenth_max = length
                        n = 0
                    else:
                        n += 1
                    if n >= 100 or len(lenth_max) >= alignment_tred:
                        break
                
                if len(lenth_max) >= 2:
                    # if len(lenth_max) > 5:
                    #     print("get")
                    alignment_anchor_list.append(lenth_max)
                    
                        
                    # if ((obj_num - cdd_num) > cdd_num) or ((cdd_num - obj_num) > obj_num) or abs(obj_num - cdd_num) > 100 or abs(sim_num-com_num) > 50 or align_rate < 0.5:#  or align_rate < 0.8 or obj_align_rate < 0.05 or cdd_align_rate <  0.05 :
                    #     alignment_temp = 0
                    # elif obj_num < anchor_alignment_dict[str(node_pair)] or cdd_num < anchor_alignment_dict[str(node_pair)]:
                    #     alignment_temp = min( obj_num, cdd_num)
                    # else: 
                    alignment_temp = len(lenth_max)
                    if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2*min(obj_num, cdd_num) and max(obj_num, cdd_num) > 100) or (abs(obj_num - cdd_num) > 200):
                        alignment_temp = 0
                    if (obj_fcg["n_num"] >= 3 and cdd_fcg["n_num"] >= 3 and alignment_temp >= alignment_tred) or (obj_num <= 10 and cdd_num <= 10 and alignment_temp >= 2):
                        
                        if alignment_temp > max_alignment_num:
                            max_alignment_num = alignment_temp
                        node_pair_feature[str(node_pair)]["alignment_num"] = alignment_temp
                        
                        score_file_dict = node_pair_feature
                        node_pair_str = str(node_pair)
                        node_alignment_num_score = score_file_dict[node_pair_str]["alignment_num"]
                        node_fcg_scale_pair = score_file_dict[node_pair_str]["fcg_scale"]
                        node_gnn_score = float(score_file_dict[node_pair_str]["gnn_score"])
                        # raw_final_score = float(score_file_dict[node_pair_str]["final_score"])
                        align_rate = float(score_file_dict[node_pair_str]["alignment_rate"])
                        node_fcg_scale_score = (node_fcg_scale_pair[0] + node_fcg_scale_pair[1])/2
                        node_fcg_scale_diff_score = 0.3 * min(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) / max(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) + 0.7
                        
                        if node_alignment_num_score > 0 and node_fcg_scale_pair[0] >=2 and node_fcg_scale_pair[1] >= 2:
                            final_score =  RARM_score(node_alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, align_rate)#, alignment_max, max_fcg)
                            score_file_dict[node_pair_str]["final_score"] = final_score
                            if (final_score >= 0.8 and node_alignment_num_score >= alignment_tred) or (final_score >= 0.95 and node_alignment_num_score >= 2):
                                if candidate_name not in target_reuse_lib_dict:
                                    target_reuse_lib_dict.append(candidate_name)
                                if candidate_name not in target_reuse_area_dict:
                                    target_reuse_area_dict[candidate_name] = {}
                                if node_pair_str not in target_reuse_area_dict[candidate_name]:
                                    target_reuse_area_dict[candidate_name][node_pair_str] = []
                                target_reuse_area_dict[candidate_name][node_pair_str].append(score_file_dict[node_pair_str])
                                reuse_flag = True
                                break
                                # print("final_score: {}".format(final_score))
                                # print("raw_final_score: {}".format(raw_final_score))
                            # elif node_alignment_num_score >=3:
                                # print("final_score: {}".format(final_score))# 成功通过gnn矫正对齐结果
                                # print("raw_final_score: {}".format(raw_final_score))
                    
                # return l, lenth_max
        
    
    if reuse_flag:
        return reuse_flag, target_reuse_area_dict#node_pair_feature, reuse_flag, max_alignment_num, obj_sim_funcs_dict, cdd_sim_funcs_dict
    else:
        return reuse_flag, {}



def tpl_detection_fast_utils_annoy_1_5(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, alignment_tred):
    reuse_flag = False
    black_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    max_alignment_num = 0
    
    alignment_anchor_list = []
    
    # alignment_tred = 3
    
    obj_sim_funcs = []
    obj_sim_funcs_dict = {}
    cdd_sim_funcs = []
    cdd_sim_funcs_dict = {}
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs:
            obj_sim_funcs.append(func_pair[0])
            obj_sim_funcs_dict[func_pair[0]] = []
        if func_pair[1] not in obj_sim_funcs_dict[func_pair[0]]:
            obj_sim_funcs_dict[func_pair[0]].append(func_pair[1])
        if func_pair[1] not in cdd_sim_funcs:
            cdd_sim_funcs.append(func_pair[1])
            cdd_sim_funcs_dict[func_pair[1]] = []
        if func_pair[0] not in cdd_sim_funcs_dict[func_pair[1]]:
            cdd_sim_funcs_dict[func_pair[1]].append(func_pair[0])
            
    
    
    # all_obj_afcg = get_afcg(obj_sim_funcs, object_graph)
    # all_obj_related_funcs = get_related_func(obj_sim_funcs, matched_func_ingraph_list)
    # all_cdd_afcg = get_afcg(cdd_sim_funcs, candidate_graph)
    # all_related_funcs = {}
    # cdd_related_funcs = get_related_func(cdd_sim_funcs, matched_func_ingraph_list)
    target_reuse_lib_dict = []
    target_reuse_area_dict = {}
    for func_pair in matched_func_ingraph_list:
        obj_afcg = get_afcg_one_annoy(func_pair[0], obj_sim_funcs, tar_afcg_dict)
        # obj_related_funcs = obj_sim_funcs_dict[func_pair[0]]#, all_related_funcs = get_related_func_one(func_pair[0], matched_func_ingraph_list, all_related_funcs)
        cdd_afcg = get_afcg_one_annoy(func_pair[1], cdd_sim_funcs, cdd_afcg_dict)
        if func_pair == ['qlz_compress3','qlz_compress']:
            print("warning")
        if len(obj_afcg) > 0 and len(cdd_afcg) > 0:
            if func_pair[1] not in black_list:
                
                
                
                l_max = 0
                lenth_max = [func_pair]
                n = 0
                
                
                node_pair_feature = {}
                fcg_results = {}
                node_pair = func_pair
                
                obj_fcg = tar_subgraph_dict[func_pair[0]]#get_subgraph(node_pair[0], object_graph)
                cdd_fcg = cdd_subgraph_dict[func_pair[1]]#get_subgraph(node_pair[1], candidate_graph)
                
                
                obj_embedding = torch.tensor(obj_fcg["embedding"])#embed_by_feat_torch(obj_fcg, gnn)
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["cdd_fcg"]["feature"]:
                #     func_name = candidate_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # cdd_fcg = node_pair_feature[str(node_pair)]["cdd_fcg"].copy()
                # cdd_fcg["embeddings"] = feature
                # start = time.time()
                cdd_embedding = torch.tensor(cdd_fcg["embedding"])#embed_by_feat_torch(cdd_fcg, gnn)
                # print(time.time() -start)
                gnn_score = F.cosine_similarity(obj_embedding, cdd_embedding, eps=1e-10, dim=1)
                gnn_score = (1 + gnn_score.cpu().detach().numpy()[0]) / 2.0
                
                if gnn_score < 0.8:
                    continue
                
                
                obj_num = len(set(obj_fcg["feature"]))
                cdd_num = len(set(cdd_fcg["feature"]))
                
                obj_com_num = obj_sim_num = 0
                for obj_func in set(obj_fcg["feature"]):
                    if obj_func in obj_com_funcs:
                        obj_com_num += 1
                        if obj_func in obj_sim_funcs_dict and list(set(obj_sim_funcs_dict[obj_func]).intersection(set(cdd_fcg["feature"]))) != []:
                            obj_sim_num += 1
                cdd_com_num = cdd_sim_num = 0
                for cdd_func in set(cdd_fcg["feature"]):
                    if cdd_func in cdd_com_funcs:
                        cdd_com_num += 1
                        if cdd_func in cdd_sim_funcs_dict and list(set(cdd_sim_funcs_dict[cdd_func]).intersection(set(obj_fcg["feature"]))) != []:
                            cdd_sim_num += 1
                
                # com_funcs_scale = (len(obj_com_funcs) + len(cdd_com_funcs)) / 2
                # sim_funcs_scale = (len(obj_sim_funcs) + len(cdd_sim_funcs)) / 2
                if obj_com_num == 0 or cdd_com_num == 0:
                    align_rate = 0
                    continue
                elif obj_com_num <= cdd_com_num:
                    align_rate = obj_sim_num / obj_com_num
                else:
                    align_rate = cdd_sim_num / cdd_com_num
                
                # align_rate = 1
                
                true_num = 0
                
                false_num = 0
                node_pair_feature[str(node_pair)] = {}
                
                node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
                node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
                node_pair_feature[str(node_pair)]["alignment_rate"] = align_rate
                node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_num, cdd_num)
                
                
                
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["obj_fcg"]["feature"]:
                #     func_name = object_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # obj_fcg = node_pair_feature[str(node_pair)]["obj_fcg"].copy()
                # obj_fcg["embeddings"] = feature
                
                node_pair_feature[str(node_pair)]["gnn_score"] = str(gnn_score)
                node_pair_feature[str(node_pair)]["obj_full_fcg_num"] = str(fcgs_num[object_name])
                # node_pair_feature[str(node_pair)]["final_score"] = str(calculate_final_score(gnn_score, node_pair_feature[str(node_pair)]["fcg_scale"][0], node_pair_feature[str(node_pair)]["alignment_num"], fcg_nums[detect_bin]))
                fcg_results[str(node_pair)] = node_pair_feature[str(node_pair)]
                
                align_rate_score = align_rate#0.3 * align_rate + 0.7
                if gnn_score * align_rate_score < 0.8:
                    continue
                
                while True:
                    length = [func_pair]
                    l, length = Alignment_v2(func_pair[0], func_pair[1], obj_afcg, cdd_afcg, obj_sim_funcs_dict, length, obj_sim_funcs, object_graph, cdd_sim_funcs, candidate_graph, matched_func_ingraph_list, tar_afcg_dict, cdd_afcg_dict)
                    # if l == 0:
                    #     n += 10
                        # if n >= 100:
                        #     break
                        # continue
                    if l > l_max:
                        l_max = l
                        lenth_max = length
                        n = 0
                    else:
                        n += 1
                    if n >= 100 or len(lenth_max) >= alignment_tred:
                        break
                
                if len(lenth_max) >= alignment_tred:
                    # if len(lenth_max) > 5:
                    #     print("get")
                    alignment_anchor_list.append(lenth_max)
                    
                        
                    # if ((obj_num - cdd_num) > cdd_num) or ((cdd_num - obj_num) > obj_num) or abs(obj_num - cdd_num) > 100 or abs(sim_num-com_num) > 50 or align_rate < 0.5:#  or align_rate < 0.8 or obj_align_rate < 0.05 or cdd_align_rate <  0.05 :
                    #     alignment_temp = 0
                    # elif obj_num < anchor_alignment_dict[str(node_pair)] or cdd_num < anchor_alignment_dict[str(node_pair)]:
                    #     alignment_temp = min( obj_num, cdd_num)
                    # else: 
                    alignment_temp = len(lenth_max)
                    if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2*min(obj_num, cdd_num) and max(obj_num, cdd_num) > 100) or (abs(obj_num - cdd_num) > 200):
                        alignment_temp = 0
                    if (obj_fcg["n_num"] >= 3 and cdd_fcg["n_num"] >= 3 and alignment_temp >= alignment_tred) or (obj_num <= 10 and cdd_num <= 10 and alignment_temp >= 2):
                        
                        if alignment_temp > max_alignment_num:
                            max_alignment_num = alignment_temp
                        node_pair_feature[str(node_pair)]["alignment_num"] = alignment_temp
                        
                        score_file_dict = node_pair_feature
                        node_pair_str = str(node_pair)
                        node_alignment_num_score = score_file_dict[node_pair_str]["alignment_num"]
                        node_fcg_scale_pair = score_file_dict[node_pair_str]["fcg_scale"]
                        node_gnn_score = float(score_file_dict[node_pair_str]["gnn_score"])
                        # raw_final_score = float(score_file_dict[node_pair_str]["final_score"])
                        align_rate = float(score_file_dict[node_pair_str]["alignment_rate"])
                        node_fcg_scale_score = (node_fcg_scale_pair[0] + node_fcg_scale_pair[1])/2
                        node_fcg_scale_diff_score = 0.3 * min(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) / max(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) + 0.7
                        
                        if node_alignment_num_score > 0 and node_fcg_scale_pair[0] >=2 and node_fcg_scale_pair[1] >= 2:
                            final_score =  RARM_score(node_alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, align_rate)#, alignment_max, max_fcg)
                            score_file_dict[node_pair_str]["final_score"] = final_score
                            if (final_score >= 0.8 and node_alignment_num_score >= alignment_tred) or (final_score >= 0.95 and node_alignment_num_score >= 2):
                                if candidate_name not in target_reuse_lib_dict:
                                    target_reuse_lib_dict.append(candidate_name)
                                if candidate_name not in target_reuse_area_dict:
                                    target_reuse_area_dict[candidate_name] = {}
                                if node_pair_str not in target_reuse_area_dict[candidate_name]:
                                    target_reuse_area_dict[candidate_name][node_pair_str] = []
                                target_reuse_area_dict[candidate_name][node_pair_str].append(score_file_dict[node_pair_str])
                                reuse_flag = True
                                break
                                # print("final_score: {}".format(final_score))
                                # print("raw_final_score: {}".format(raw_final_score))
                            # elif node_alignment_num_score >=3:
                                # print("final_score: {}".format(final_score))# 成功通过gnn矫正对齐结果
                                # print("raw_final_score: {}".format(raw_final_score))
                    
                # return l, lenth_max
        
    
    if reuse_flag:
        return reuse_flag, target_reuse_area_dict#node_pair_feature, reuse_flag, max_alignment_num, obj_sim_funcs_dict, cdd_sim_funcs_dict
    else:
        return reuse_flag, {}



def tpl_detection_fast_utils_annoy(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict):
    reuse_flag = False
    black_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    max_alignment_num = 0
    
    alignment_anchor_list = []
    
    obj_sim_funcs = []
    obj_sim_funcs_dict = {}
    cdd_sim_funcs = []
    cdd_sim_funcs_dict = {}
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs:
            obj_sim_funcs.append(func_pair[0])
            obj_sim_funcs_dict[func_pair[0]] = []
        if func_pair[1] not in obj_sim_funcs_dict[func_pair[0]]:
            obj_sim_funcs_dict[func_pair[0]].append(func_pair[1])
        if func_pair[1] not in cdd_sim_funcs:
            cdd_sim_funcs.append(func_pair[1])
            cdd_sim_funcs_dict[func_pair[1]] = []
        if func_pair[0] not in cdd_sim_funcs_dict[func_pair[1]]:
            cdd_sim_funcs_dict[func_pair[1]].append(func_pair[0])
            
    
    
    # all_obj_afcg = get_afcg(obj_sim_funcs, object_graph)
    # all_obj_related_funcs = get_related_func(obj_sim_funcs, matched_func_ingraph_list)
    # all_cdd_afcg = get_afcg(cdd_sim_funcs, candidate_graph)
    # all_related_funcs = {}
    # cdd_related_funcs = get_related_func(cdd_sim_funcs, matched_func_ingraph_list)
    target_reuse_lib_dict = []
    target_reuse_area_dict = {}
    for func_pair in matched_func_ingraph_list:
        obj_afcg = get_afcg_one_annoy(func_pair[0], obj_sim_funcs, tar_afcg_dict)
        # obj_related_funcs = obj_sim_funcs_dict[func_pair[0]]#, all_related_funcs = get_related_func_one(func_pair[0], matched_func_ingraph_list, all_related_funcs)
        cdd_afcg = get_afcg_one_annoy(func_pair[1], cdd_sim_funcs, cdd_afcg_dict)
        if func_pair == ['BZ2_bzDecompress','ZSTD_compressEnd']:
            print("warning")
        if len(obj_afcg) > 0 and len(cdd_afcg) > 0:
            if func_pair[1] not in black_list:
                
                
                
                l_max = 0
                lenth_max = [func_pair]
                n = 0
                
                
                node_pair_feature = {}
                fcg_results = {}
                node_pair = func_pair
                
                obj_fcg = tar_subgraph_dict[func_pair[0]]#get_subgraph(node_pair[0], object_graph)
                cdd_fcg = cdd_subgraph_dict[func_pair[1]]#get_subgraph(node_pair[1], candidate_graph)
                
                
                obj_embedding = torch.tensor(obj_fcg["embedding"])#embed_by_feat_torch(obj_fcg, gnn)
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["cdd_fcg"]["feature"]:
                #     func_name = candidate_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # cdd_fcg = node_pair_feature[str(node_pair)]["cdd_fcg"].copy()
                # cdd_fcg["embeddings"] = feature
                # start = time.time()
                cdd_embedding = torch.tensor(cdd_fcg["embedding"])#embed_by_feat_torch(cdd_fcg, gnn)
                # print(time.time() -start)
                gnn_score = F.cosine_similarity(obj_embedding, cdd_embedding, eps=1e-10, dim=1)
                gnn_score = (1 + gnn_score.cpu().detach().numpy()[0]) / 2.0
                
                if gnn_score < 0.8:
                    continue
                
                
                obj_num = len(set(obj_fcg["feature"]))
                cdd_num = len(set(cdd_fcg["feature"]))
                
                obj_com_num = obj_sim_num = 0
                for obj_func in set(obj_fcg["feature"]):
                    if obj_func in obj_com_funcs:
                        obj_com_num += 1
                        if obj_func in obj_sim_funcs_dict and list(set(obj_sim_funcs_dict[obj_func]).intersection(set(cdd_fcg["feature"]))) != []:
                            obj_sim_num += 1
                cdd_com_num = cdd_sim_num = 0
                for cdd_func in set(cdd_fcg["feature"]):
                    if cdd_func in cdd_com_funcs:
                        cdd_com_num += 1
                        if cdd_func in cdd_sim_funcs_dict and list(set(cdd_sim_funcs_dict[cdd_func]).intersection(set(obj_fcg["feature"]))) != []:
                            cdd_sim_num += 1
                
                # com_funcs_scale = (len(obj_com_funcs) + len(cdd_com_funcs)) / 2
                # sim_funcs_scale = (len(obj_sim_funcs) + len(cdd_sim_funcs)) / 2
                if obj_com_num == 0 or cdd_com_num == 0:
                    align_rate = 0
                    continue
                elif obj_com_num <= cdd_com_num:
                    align_rate = obj_sim_num / obj_com_num
                else:
                    align_rate = cdd_sim_num / cdd_com_num
                
                # align_rate = 1
                
                true_num = 0
                
                false_num = 0
                node_pair_feature[str(node_pair)] = {}
                
                node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
                node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
                node_pair_feature[str(node_pair)]["alignment_rate"] = align_rate
                node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_num, cdd_num)
                
                
                
                # feature = []
                # for func in node_pair_feature[str(node_pair)]["obj_fcg"]["feature"]:
                #     func_name = object_name + "|||" + func
                #     if func_name in func_embeddings:
                #         embed = func_embeddings[func_name][0]
                #         true_num += 1
                #     else:
                #         false_num += 1
                #         embed = list(np.array([0.001 for i in range(64)]))
                #     feature.append(embed)
                # obj_fcg = node_pair_feature[str(node_pair)]["obj_fcg"].copy()
                # obj_fcg["embeddings"] = feature
                
                node_pair_feature[str(node_pair)]["gnn_score"] = str(gnn_score)
                node_pair_feature[str(node_pair)]["obj_full_fcg_num"] = str(fcgs_num[object_name])
                # node_pair_feature[str(node_pair)]["final_score"] = str(calculate_final_score(gnn_score, node_pair_feature[str(node_pair)]["fcg_scale"][0], node_pair_feature[str(node_pair)]["alignment_num"], fcg_nums[detect_bin]))
                fcg_results[str(node_pair)] = node_pair_feature[str(node_pair)]
                
                align_rate_score = 0.3 * align_rate + 0.7
                if gnn_score * align_rate_score < 0.8:
                    continue
                
                while True:
                    length = [func_pair]
                    l, length = Alignment_v2(func_pair[0], func_pair[1], obj_afcg, cdd_afcg, obj_sim_funcs_dict, length, obj_sim_funcs, object_graph, cdd_sim_funcs, candidate_graph, matched_func_ingraph_list, tar_afcg_dict, cdd_afcg_dict)
                    # if l == 0:
                    #     n += 10
                        # if n >= 100:
                        #     break
                        # continue
                    if l > l_max:
                        l_max = l
                        lenth_max = length
                        n = 0
                    else:
                        n += 1
                    if n >= 100 or len(lenth_max) >= 3:
                        break
                
                if len(lenth_max) >= 2:
                    # if len(lenth_max) > 5:
                    #     print("get")
                    alignment_anchor_list.append(lenth_max)
                    
                        
                    # if ((obj_num - cdd_num) > cdd_num) or ((cdd_num - obj_num) > obj_num) or abs(obj_num - cdd_num) > 100 or abs(sim_num-com_num) > 50 or align_rate < 0.5:#  or align_rate < 0.8 or obj_align_rate < 0.05 or cdd_align_rate <  0.05 :
                    #     alignment_temp = 0
                    # elif obj_num < anchor_alignment_dict[str(node_pair)] or cdd_num < anchor_alignment_dict[str(node_pair)]:
                    #     alignment_temp = min( obj_num, cdd_num)
                    # else: 
                    alignment_temp = len(lenth_max)
                    if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2*min(obj_num, cdd_num) and max(obj_num, cdd_num) > 100) or (abs(obj_num - cdd_num) > 200):
                        alignment_temp = 0
                    if (obj_fcg["n_num"] >= 3 and cdd_fcg["n_num"] >= 3 and alignment_temp >= 3) or (obj_num <= 10 and cdd_num <= 10 and alignment_temp >= 2):
                        
                        if alignment_temp > max_alignment_num:
                            max_alignment_num = alignment_temp
                        node_pair_feature[str(node_pair)]["alignment_num"] = alignment_temp
                        
                        score_file_dict = node_pair_feature
                        node_pair_str = str(node_pair)
                        node_alignment_num_score = score_file_dict[node_pair_str]["alignment_num"]
                        node_fcg_scale_pair = score_file_dict[node_pair_str]["fcg_scale"]
                        node_gnn_score = float(score_file_dict[node_pair_str]["gnn_score"])
                        # raw_final_score = float(score_file_dict[node_pair_str]["final_score"])
                        align_rate = float(score_file_dict[node_pair_str]["alignment_rate"])
                        node_fcg_scale_score = (node_fcg_scale_pair[0] + node_fcg_scale_pair[1])/2
                        node_fcg_scale_diff_score = 0.3 * min(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) / max(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) + 0.7
                        
                        if node_alignment_num_score > 0 and node_fcg_scale_pair[0] >=2 and node_fcg_scale_pair[1] >= 2:
                            final_score =  RARM_score(node_alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, align_rate)#, alignment_max, max_fcg)
                            score_file_dict[node_pair_str]["final_score"] = final_score
                            if (final_score >= 0.8 and node_alignment_num_score >= 3) or (final_score >= 0.95 and node_alignment_num_score >= 2):
                                if candidate_name not in target_reuse_lib_dict:
                                    target_reuse_lib_dict.append(candidate_name)
                                if candidate_name not in target_reuse_area_dict:
                                    target_reuse_area_dict[candidate_name] = {}
                                if node_pair_str not in target_reuse_area_dict[candidate_name]:
                                    target_reuse_area_dict[candidate_name][node_pair_str] = []
                                target_reuse_area_dict[candidate_name][node_pair_str].append(score_file_dict[node_pair_str])
                                reuse_flag = True
                                break
                                # print("final_score: {}".format(final_score))
                                # print("raw_final_score: {}".format(raw_final_score))
                            # elif node_alignment_num_score >=3:
                                # print("final_score: {}".format(final_score))# 成功通过gnn矫正对齐结果
                                # print("raw_final_score: {}".format(raw_final_score))
                    
                # return l, lenth_max
        
    
    if reuse_flag:
        return reuse_flag, target_reuse_area_dict#node_pair_feature, reuse_flag, max_alignment_num, obj_sim_funcs_dict, cdd_sim_funcs_dict
    else:
        return reuse_flag, {}


def tpl_detection_fast_utils(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, func_embeddings, gnn, fcgs_num):
    reuse_flag = False
    black_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    max_alignment_num = 0
    
    alignment_anchor_list = []
    
    obj_sim_funcs = []
    obj_sim_funcs_dict = {}
    cdd_sim_funcs = []
    cdd_sim_funcs_dict = {}
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs:
            obj_sim_funcs.append(func_pair[0])
            obj_sim_funcs_dict[func_pair[0]] = []
        if func_pair[1] not in obj_sim_funcs_dict[func_pair[0]]:
            obj_sim_funcs_dict[func_pair[0]].append(func_pair[1])
        if func_pair[1] not in cdd_sim_funcs:
            cdd_sim_funcs.append(func_pair[1])
            cdd_sim_funcs_dict[func_pair[1]] = []
        if func_pair[0] not in cdd_sim_funcs_dict[func_pair[1]]:
            cdd_sim_funcs_dict[func_pair[1]].append(func_pair[0])
            
    
    
    # all_obj_afcg = get_afcg(obj_sim_funcs, object_graph)
    # all_obj_related_funcs = get_related_func(obj_sim_funcs, matched_func_ingraph_list)
    # all_cdd_afcg = get_afcg(cdd_sim_funcs, candidate_graph)
    all_obj_afcg = {}
    all_related_funcs = {}
    all_cdd_afcg = {}
    # cdd_related_funcs = get_related_func(cdd_sim_funcs, matched_func_ingraph_list)
    
    for func_pair in matched_func_ingraph_list:
        obj_afcg, all_obj_afcg = get_afcg_one(func_pair[0], obj_sim_funcs, object_graph, all_obj_afcg)
        obj_related_funcs, all_related_funcs = get_related_func_one(func_pair[0], matched_func_ingraph_list, all_related_funcs)
        cdd_afcg, all_cdd_afcg = get_afcg_one(func_pair[1], cdd_sim_funcs, candidate_graph, all_cdd_afcg)
        if len(obj_afcg) >= 2 and len(cdd_afcg) >= 2:
        # if func_pair == ['ZSTD_createCCtxParams','ZSTD_createCCtxParams']:
        #     print("warning")
            if func_pair[1] not in black_list:
                l_max = 0
                lenth_max = [func_pair]
                n = 0
                
                
                node_pair_feature = {}
                fcg_results = {}
                node_pair = func_pair
                
                obj_fcg = get_subgraph(node_pair[0], object_graph)
                cdd_fcg = get_subgraph(node_pair[1], candidate_graph)
                
                obj_num = len(set(obj_fcg["feature"]))
                cdd_num = len(set(cdd_fcg["feature"]))
                
                obj_com_num = obj_sim_num = 0
                for obj_func in set(obj_fcg["feature"]):
                    if obj_func in obj_com_funcs:
                        obj_com_num += 1
                        if obj_func in obj_sim_funcs_dict and list(set(obj_sim_funcs_dict[obj_func]).intersection(set(cdd_fcg["feature"]))) != []:
                            obj_sim_num += 1
                cdd_com_num = cdd_sim_num = 0
                for cdd_func in set(cdd_fcg["feature"]):
                    if cdd_func in cdd_com_funcs:
                        cdd_com_num += 1
                        if cdd_func in cdd_sim_funcs_dict and list(set(cdd_sim_funcs_dict[cdd_func]).intersection(set(obj_fcg["feature"]))) != []:
                            cdd_sim_num += 1
                
                # com_funcs_scale = (len(obj_com_funcs) + len(cdd_com_funcs)) / 2
                # sim_funcs_scale = (len(obj_sim_funcs) + len(cdd_sim_funcs)) / 2
                if obj_com_num == 0 or cdd_com_num == 0:
                    align_rate = 0
                    continue
                elif obj_com_num <= cdd_com_num:
                    align_rate = obj_sim_num / obj_com_num
                else:
                    align_rate = cdd_sim_num / cdd_com_num
                
                
                
                true_num = 0
                
                false_num = 0
                node_pair_feature[str(node_pair)] = {}
                
                node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
                node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
                node_pair_feature[str(node_pair)]["alignment_rate"] = align_rate
                node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_num, cdd_num)
                
                
                
                feature = []
                for func in node_pair_feature[str(node_pair)]["obj_fcg"]["feature"]:
                    func_name = object_name + "|||" + func
                    if func_name in func_embeddings:
                        embed = func_embeddings[func_name][0]
                        true_num += 1
                    else:
                        false_num += 1
                        embed = list(np.array([0.001 for i in range(64)]))
                    feature.append(embed)
                obj_fcg = node_pair_feature[str(node_pair)]["obj_fcg"].copy()
                obj_fcg["embeddings"] = feature
                obj_embedding = embed_by_feat_torch(obj_fcg, gnn)
                feature = []
                for func in node_pair_feature[str(node_pair)]["cdd_fcg"]["feature"]:
                    func_name = candidate_name + "|||" + func
                    if func_name in func_embeddings:
                        embed = func_embeddings[func_name][0]
                        true_num += 1
                    else:
                        false_num += 1
                        embed = list(np.array([0.001 for i in range(64)]))
                    feature.append(embed)
                cdd_fcg = node_pair_feature[str(node_pair)]["cdd_fcg"].copy()
                cdd_fcg["embeddings"] = feature
                # start = time.time()
                cdd_embedding = embed_by_feat_torch(cdd_fcg, gnn)
                # print(time.time() -start)
                gnn_score = F.cosine_similarity(obj_embedding, cdd_embedding, eps=1e-10, dim=1)
                gnn_score = (1 + gnn_score.cpu().detach().numpy()[0]) / 2.0
                node_pair_feature[str(node_pair)]["gnn_score"] = str(gnn_score)
                node_pair_feature[str(node_pair)]["obj_full_fcg_num"] = str(fcgs_num[object_name])
                # node_pair_feature[str(node_pair)]["final_score"] = str(calculate_final_score(gnn_score, node_pair_feature[str(node_pair)]["fcg_scale"][0], node_pair_feature[str(node_pair)]["alignment_num"], fcg_nums[detect_bin]))
                fcg_results[str(node_pair)] = node_pair_feature[str(node_pair)]
                
                align_rate_score = 0.3 * align_rate + 0.7
                if gnn_score * align_rate_score < 0.8:
                    continue
                
                while True:
                    length = [func_pair]
                    l, length, all_related_funcs, all_obj_afcg, all_cdd_afcg = Alignment(func_pair[0], func_pair[1], obj_afcg, cdd_afcg, obj_related_funcs, length, obj_sim_funcs, object_graph, cdd_sim_funcs, candidate_graph, matched_func_ingraph_list, all_related_funcs, all_obj_afcg, all_cdd_afcg)
                    # if l == 0:
                    #     n += 10
                        # if n >= 100:
                        #     break
                        # continue
                    if l > l_max:
                        l_max = l
                        lenth_max = length
                        n = 0
                    else:
                        n += 1
                    if n >= 100 or len(lenth_max) >= 3:
                        break
                
                if len(lenth_max) >= 2:
                    # if len(lenth_max) > 5:
                    #     print("get")
                    alignment_anchor_list.append(lenth_max)
                    
                        
                    # if ((obj_num - cdd_num) > cdd_num) or ((cdd_num - obj_num) > obj_num) or abs(obj_num - cdd_num) > 100 or abs(sim_num-com_num) > 50 or align_rate < 0.5:#  or align_rate < 0.8 or obj_align_rate < 0.05 or cdd_align_rate <  0.05 :
                    #     alignment_temp = 0
                    # elif obj_num < anchor_alignment_dict[str(node_pair)] or cdd_num < anchor_alignment_dict[str(node_pair)]:
                    #     alignment_temp = min( obj_num, cdd_num)
                    # else: 
                    alignment_temp = len(lenth_max)
                    if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2*min(obj_num, cdd_num) and max(obj_num, cdd_num) > 100) or (abs(obj_num - cdd_num) > 200):
                        alignment_temp = 0
                    if (obj_fcg["n_num"] >= 3 and cdd_fcg["n_num"] >= 3 and alignment_temp >= 3) or (obj_num <= 10 and cdd_num <= 10 and alignment_temp >= 2):
                        
                        if alignment_temp > max_alignment_num:
                            max_alignment_num = alignment_temp
                        node_pair_feature[str(node_pair)]["alignment_num"] = alignment_temp
                        target_reuse_lib_dict = []
                        target_reuse_area_dict = {}
                        score_file_dict = node_pair_feature
                        node_pair_str = str(node_pair)
                        node_alignment_num_score = score_file_dict[node_pair_str]["alignment_num"]
                        node_fcg_scale_pair = score_file_dict[node_pair_str]["fcg_scale"]
                        node_gnn_score = float(score_file_dict[node_pair_str]["gnn_score"])
                        # raw_final_score = float(score_file_dict[node_pair_str]["final_score"])
                        align_rate = float(score_file_dict[node_pair_str]["alignment_rate"])
                        node_fcg_scale_score = (node_fcg_scale_pair[0] + node_fcg_scale_pair[1])/2
                        node_fcg_scale_diff_score = 0.3 * min(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) / max(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) + 0.7
                        
                        if node_alignment_num_score > 0 and node_fcg_scale_pair[0] >=2 and node_fcg_scale_pair[1] >= 2:
                            final_score =  RARM_score(node_alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, align_rate)#, alignment_max, max_fcg)
                            score_file_dict[node_pair_str]["final_score"] = final_score
                            if (final_score >= 0.8 and node_alignment_num_score >= 3) or (final_score >= 0.95 and node_alignment_num_score >= 2):
                                if candidate_name not in target_reuse_lib_dict:
                                    target_reuse_lib_dict.append(candidate_name)
                                if candidate_name not in target_reuse_area_dict:
                                    target_reuse_area_dict[candidate_name] = {}
                                if node_pair_str not in target_reuse_area_dict[candidate_name]:
                                    target_reuse_area_dict[candidate_name][node_pair_str] = []
                                target_reuse_area_dict[candidate_name][node_pair_str].append(score_file_dict[node_pair_str])
                                reuse_flag = True
                                break
                                # print("final_score: {}".format(final_score))
                                # print("raw_final_score: {}".format(raw_final_score))
                            # elif node_alignment_num_score >=3:
                                # print("final_score: {}".format(final_score))# 成功通过gnn矫正对齐结果
                                # print("raw_final_score: {}".format(raw_final_score))
                    
                # return l, lenth_max
        
    
    if reuse_flag:
        return reuse_flag, target_reuse_area_dict#node_pair_feature, reuse_flag, max_alignment_num, obj_sim_funcs_dict, cdd_sim_funcs_dict
    else:
        return reuse_flag, {}

def anchor_alignment_ransac_v1_0(matched_func_ingraph_list, object_graph, candidate_graph):
    
    black_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]

    
    alignment_anchor_list = []
    
    obj_sim_funcs = []
    obj_sim_funcs_dict = {}
    cdd_sim_funcs = []
    cdd_sim_funcs_dict = {}
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs:
            obj_sim_funcs.append(func_pair[0])
            obj_sim_funcs_dict[func_pair[0]] = []
        if func_pair[1] not in obj_sim_funcs_dict[func_pair[0]]:
            obj_sim_funcs_dict[func_pair[0]].append(func_pair[1])
        if func_pair[1] not in cdd_sim_funcs:
            cdd_sim_funcs.append(func_pair[1])
            cdd_sim_funcs_dict[func_pair[1]] = []
        if func_pair[0] not in cdd_sim_funcs_dict[func_pair[1]]:
            cdd_sim_funcs_dict[func_pair[1]].append(func_pair[0])
            
    
    
    obj_afcg = get_afcg(obj_sim_funcs, object_graph)
    obj_related_funcs = get_related_func(obj_sim_funcs, matched_func_ingraph_list)
    cdd_afcg = get_afcg(cdd_sim_funcs, candidate_graph)
    # cdd_related_funcs = get_related_func(cdd_sim_funcs, matched_func_ingraph_list)
    
    for func_pair in matched_func_ingraph_list:
        # if func_pair == ['ZSTD_createCCtxParams','ZSTD_createCCtxParams']:
        #     print("warning")
        if func_pair[1] not in black_list:
            l_max = 0
            lenth_max = [func_pair]
            n = 0
            
            while True:
                length = [func_pair]
                l, length = Alignment(func_pair[0], func_pair[1], obj_afcg, cdd_afcg, obj_related_funcs, length)
                # if l == 0:
                #     n += 10
                    # if n >= 100:
                    #     break
                    # continue
                if l > l_max:
                    l_max = l
                    lenth_max = length
                    n = 0
                else:
                    n += 1
                if n >= 100:
                    break
            
            if len(lenth_max) >= 2:
                # if len(lenth_max) > 5:
                #     print("get")
                alignment_anchor_list.append(lenth_max)
            # return l, lenth_max
    
    
    
    return alignment_anchor_list, obj_sim_funcs_dict, cdd_sim_funcs_dict


def anchor_alignment_v2_0(matched_func_ingraph_list, object_graph, candidate_graph):
    cdd_sim_funcs = []
    obj_sim_funcs = []
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs:
            obj_sim_funcs.append(func_pair[0])
        if func_pair[1] not in cdd_sim_funcs:
            cdd_sim_funcs.append(func_pair[1])
    
    obj_afcg = get_afcg(obj_sim_funcs, object_graph)
    cdd_afcg = get_afcg(cdd_sim_funcs, candidate_graph)
    
    alignment_anchor_list = []
    alignment_anchor_list_set = set()
    
    for node_pair in matched_func_ingraph_list:
        afcg_matched_list = []
        
        obj_func_list = obj_afcg[node_pair[0]]
        cdd_func_list = cdd_afcg[node_pair[1]]
        if obj_func_list != [] and cdd_func_list != []:
            for sub_node_pair in matched_func_ingraph_list:
                if sub_node_pair[0] in obj_func_list and sub_node_pair[1] in cdd_func_list:
                    afcg_matched_list.append(sub_node_pair)
            
            func_list, alignment_max = get_subgraph_func_list_v5(afcg_matched_list, object_graph, candidate_graph)
            if alignment_max == False and len(func_list) > 0:
                i_len = len(func_list)
                for i in range(i_len):
                    new_list = [node_pair]
                    new_list.extend(func_list[i])
                    if str(new_list) not in alignment_anchor_list_set:
                        alignment_anchor_list_set.add(str(new_list))
                        alignment_anchor_list.append(new_list)
    if alignment_anchor_list != []:
        return alignment_anchor_list, False
    else:
        return alignment_anchor_list, True
                
    



def filter_alignment_result(sub_graph_list, object_graph, candidate_graph):
    
    
    
    
    return sub_graph_list







def get_up_linked_node(ite_num , obj_walked_node_set, cdd_walked_node_set, obj_node_list, cdd_node_list, matched_func_ingraph_list, tainted_object_graph, tainted_candidate_graph):
    ite_num += 1
    # print(ite_num)
    if ite_num > 50:
        print("error")
    up_path_node_num = 0
    matched_node_pair_list = []
    
    up_obj_node_dict = {}
    up_cdd_node_dict = {}
    for obj_node in obj_node_list:
        if obj_node not in obj_walked_node_set:
            obj_walked_node_set.add(obj_node)
            father_up_obj_node_dict = get_father_brother_node(obj_node, tainted_object_graph)
            up_obj_node_dict = dict(up_obj_node_dict, **father_up_obj_node_dict)
    for cdd_node in cdd_node_list:
        if cdd_node not in cdd_walked_node_set:
            cdd_walked_node_set.add(cdd_node)
            father_up_cdd_node_dict = get_father_brother_node(cdd_node, tainted_candidate_graph)
            up_cdd_node_dict = dict(up_cdd_node_dict, **father_up_cdd_node_dict)
    
    if up_obj_node_dict != {} and up_cdd_node_dict != {}:
        
        next_up_obj_node_list = list(up_obj_node_dict.keys())
        next_up_cdd_node_list = list(up_cdd_node_dict.keys())
        
        for up_obj_node in up_obj_node_dict:
            for up_cdd_node in up_cdd_node_dict:
                if [up_obj_node, up_cdd_node] in matched_func_ingraph_list:
                    up_path_node_num += 1
                    print("find "+str([up_obj_node, up_cdd_node]))
                    if up_obj_node in next_up_obj_node_list:
                        next_up_obj_node_list.remove(next_up_obj_node_list[next_up_obj_node_list.index(up_obj_node)])
                    if up_cdd_node in next_up_cdd_node_list:
                        next_up_cdd_node_list.remove(next_up_cdd_node_list[next_up_cdd_node_list.index(up_cdd_node)])
                    matched_node_pair_list.append([up_obj_node, up_cdd_node])
                    (next_up_path_node_num, next_matched_node_pair_list, obj_next_walked_down_node_set, cdd_next_walked_down_node_set) = get_down_linked_node(ite_num , obj_walked_node_set, cdd_walked_node_set, [up_obj_node], [up_cdd_node], matched_func_ingraph_list, tainted_object_graph, tainted_candidate_graph)
                    up_path_node_num += next_up_path_node_num
                    matched_node_pair_list.extend(next_matched_node_pair_list)
                    obj_walked_node_set = obj_walked_node_set.union(obj_next_walked_down_node_set)
                    cdd_walked_node_set = cdd_walked_node_set.union(cdd_next_walked_down_node_set)
                    (next_up_path_node_num, next_matched_node_pair_list, obj_next_walked_up_node_set, cdd_next_walked_up_node_set) = get_up_linked_node(ite_num , obj_walked_node_set, cdd_walked_node_set, [up_obj_node], [up_cdd_node], matched_func_ingraph_list, tainted_object_graph, tainted_candidate_graph)
                    up_path_node_num += next_up_path_node_num
                    matched_node_pair_list.extend(next_matched_node_pair_list)
                    obj_walked_node_set = obj_walked_node_set.union(obj_next_walked_up_node_set)
                    cdd_walked_node_set = cdd_walked_node_set.union(cdd_next_walked_up_node_set)
                    up_cdd_node_dict.pop(up_cdd_node)
                    break
                
        (next_up_path_node_num, next_matched_node_pair_list, obj_next_walked_up_node_set, cdd_next_walked_up_node_set) = get_up_linked_node(ite_num , obj_walked_node_set, cdd_walked_node_set, next_up_obj_node_list, next_up_cdd_node_list, matched_func_ingraph_list, tainted_object_graph, tainted_candidate_graph)
        up_path_node_num += next_up_path_node_num
        matched_node_pair_list.extend(next_matched_node_pair_list)
        obj_walked_node_set = obj_walked_node_set.union(obj_next_walked_up_node_set)
        cdd_walked_node_set = cdd_walked_node_set.union(cdd_next_walked_up_node_set)

                
     
    return up_path_node_num, matched_node_pair_list, obj_walked_node_set, cdd_walked_node_set


def get_down_linked_node(ite_num, obj_walked_node_set, cdd_walked_node_set, obj_node_list, cdd_node_list, matched_func_ingraph_list, tainted_object_graph, tainted_candidate_graph):
    ite_num += 1
    if ite_num > 50:
        print("error")
    # print(ite_num)
    down_path_node_num = 0
    matched_node_pair_list = []
    
    down_obj_node_dict = {}
    down_cdd_node_dict = {}
    for obj_node in obj_node_list:
        if obj_node not in obj_walked_node_set:
            obj_walked_node_set.add(obj_node)
            down_obj_node_dict = dict(down_obj_node_dict, **get_child_brother_node(obj_node, tainted_object_graph))
    for cdd_node in cdd_node_list:
        if cdd_node not in cdd_walked_node_set:
            cdd_walked_node_set.add(cdd_node)
            down_cdd_node_dict = dict(down_cdd_node_dict, **get_child_brother_node(cdd_node, tainted_candidate_graph))
    
    if down_obj_node_dict != {} and down_cdd_node_dict != {}:
        
        next_down_obj_node_list = list(down_obj_node_dict.keys())
        next_down_cdd_node_list = list(down_cdd_node_dict.keys())
        
        for down_obj_node in down_obj_node_dict:
            for down_cdd_node in down_cdd_node_dict:
                if [down_obj_node, down_cdd_node] in matched_func_ingraph_list:
                    print("find "+str([down_obj_node, down_cdd_node]))
                    down_path_node_num += 1
                    if down_obj_node in next_down_obj_node_list:
                        next_down_obj_node_list.remove(next_down_obj_node_list[next_down_obj_node_list.index(down_obj_node)])
                    if down_cdd_node in next_down_cdd_node_list:
                        next_down_cdd_node_list.remove(next_down_cdd_node_list[next_down_cdd_node_list.index(down_cdd_node)])
                    matched_node_pair_list.append([down_obj_node, down_cdd_node])
                    (next_down_path_node_num, next_matched_node_pair_list, obj_next_walked_up_node_set, cdd_next_walked_up_node_set) = get_up_linked_node(ite_num , obj_walked_node_set, cdd_walked_node_set, [down_obj_node], [down_cdd_node], matched_func_ingraph_list, tainted_object_graph, tainted_candidate_graph)
                    down_path_node_num += next_down_path_node_num
                    matched_node_pair_list.extend(next_matched_node_pair_list)
                    obj_walked_node_set = obj_walked_node_set.union(obj_next_walked_up_node_set)
                    cdd_walked_node_set = cdd_walked_node_set.union(cdd_next_walked_up_node_set)
                    (next_down_path_node_num, next_matched_node_pair_list, obj_next_walked_node_set, cdd_next_walked_node_set) = get_down_linked_node(ite_num , obj_walked_node_set, cdd_walked_node_set, [down_obj_node], [down_cdd_node], matched_func_ingraph_list, tainted_object_graph, tainted_candidate_graph)
                    down_path_node_num += next_down_path_node_num
                    matched_node_pair_list.extend(next_matched_node_pair_list)
                    obj_walked_node_set = obj_walked_node_set.union(obj_next_walked_node_set)
                    cdd_walked_node_set = cdd_walked_node_set.union(cdd_next_walked_node_set)
                    down_cdd_node_dict
                    down_cdd_node_dict.pop(down_cdd_node)
                    break
        
        (next_down_path_node_num, next_matched_node_pair_list, obj_next_walked_node_set, cdd_next_walked_node_set) = get_down_linked_node(ite_num , obj_walked_node_set, cdd_walked_node_set, next_down_obj_node_list, next_down_cdd_node_list, matched_func_ingraph_list, tainted_object_graph, tainted_candidate_graph)
        down_path_node_num += next_down_path_node_num
        matched_node_pair_list.extend(next_matched_node_pair_list)
        obj_walked_node_set = obj_walked_node_set.union(obj_next_walked_node_set)
        cdd_walked_node_set = cdd_walked_node_set.union(cdd_next_walked_node_set)

     
    return down_path_node_num, matched_node_pair_list, obj_walked_node_set, cdd_walked_node_set



def have_subgraph_edge(func_list_item, new_node_pair, object_graph, candidate_graph):
    
    for old_func_pair in func_list_item:
        is_obj_subgraph = is_subgraph_edge([old_func_pair[0], new_node_pair[0]], object_graph)
        is_cdd_subgraph = is_subgraph_edge([old_func_pair[1], new_node_pair[1]], candidate_graph)
        is_obj_subgraph_re = is_subgraph_edge([new_node_pair[0], old_func_pair[0]], object_graph)
        is_cdd_subgraph_re = is_subgraph_edge([new_node_pair[1], old_func_pair[1]], candidate_graph)
    
        if (is_obj_subgraph == True and is_cdd_subgraph == True) or (is_obj_subgraph_re == True and is_cdd_subgraph_re == True):
            return True
    return False


# 存在bug：插入多个后，有的节点跟左右可能都不连接
def have_subgraph_edge_v2(func_list_item, new_node_pair, object_graph, candidate_graph):
    
    for old_func_pair in func_list_item:
        is_obj_subgraph = is_subgraph_edge_v2([old_func_pair[0], new_node_pair[0]], object_graph)
        is_cdd_subgraph = is_subgraph_edge_v2([old_func_pair[1], new_node_pair[1]], candidate_graph)
        # if [new_node_pair[0], old_func_pair[0]] == ['BZ2_hbAssignCodes', 'BZ2_bzwrite']:
        #     print("warning")
        is_obj_subgraph_re = is_subgraph_edge_v2([new_node_pair[0], old_func_pair[0]], object_graph)
        is_cdd_subgraph_re = is_subgraph_edge_v2([new_node_pair[1], old_func_pair[1]], candidate_graph)
    
        if (is_obj_subgraph == True and is_cdd_subgraph == True):
            return True, func_list_item.index(old_func_pair)+1
        elif (is_obj_subgraph_re == True and is_cdd_subgraph_re == True):
            return True, func_list_item.index(old_func_pair)
    return False, 0



# 修复bug：插入多个后，有的节点跟左右可能都不连接
# 遍历所有路径，将新节点连到最近的节点上（前或后）
def have_subgraph_edge_v3(func_list_item, new_node_pair, object_graph, candidate_graph):
    connect_flag = [-1, -1]
    re_connect_flag = [-1, -1]
    for i in range(0, len(func_list_item)):
        (is_obj_subgraph, distance1) = is_subgraph_edge_v3([func_list_item[len(func_list_item)-i-1][0], new_node_pair[0]], object_graph)
        (is_cdd_subgraph, distance2) = is_subgraph_edge_v3([func_list_item[len(func_list_item)-i-1][1], new_node_pair[1]], candidate_graph)
        if (is_obj_subgraph == True and is_cdd_subgraph == True):
            if connect_flag == [-1, -1] or (distance1+distance2) < connect_flag[1]:
                connect_flag = (len(func_list_item)-i-1, (distance1+distance2))
        # if [new_node_pair[0], old_func_pair[0]] == ['BZ2_hbAssignCodes', 'BZ2_bzwrite']:
        #     print("warning")
        (is_obj_subgraph_re, distance1) = is_subgraph_edge_v3([new_node_pair[0], func_list_item[i][0]], object_graph)
        (is_cdd_subgraph_re, distance2) = is_subgraph_edge_v3([new_node_pair[1], func_list_item[i][1]], candidate_graph)
        if (is_obj_subgraph_re == True and is_cdd_subgraph_re == True):
            if re_connect_flag == [-1, -1] or (distance1+distance2) < re_connect_flag[1]:
                re_connect_flag = (i, (distance1+distance2))
    
    if connect_flag != [-1, -1] and re_connect_flag != [-1, -1]:
        if connect_flag[1] > re_connect_flag[1]:
            return (True, re_connect_flag[0])
        else:
            return (True, connect_flag[0])
    elif connect_flag != [-1, -1]:
        return (True, connect_flag[0])
    elif re_connect_flag != [-1, -1]:
        return (True, re_connect_flag[0])
    else:
        return False, 0





# def del_wrong_func_pair(func_list, object_graph, candidate_graph):
#     return_func_list = []
    
#     for func_list_item in func_list:
#         if have_subgraph_edge(func_list_item, object_graph, candidate_graph):
#             return_func_list.append(func_list_item)

#     return return_func_list


def get_subgraph_func_list(matched_func_ingraph_list, object_graph, candidate_graph):
    
    obj_func_set = set()
    cdd_func_set = set()
    func_list = []
    func_list_set = set()
    
    
    i = 0
    
    for node_pair in matched_func_ingraph_list:
        # if "_start" in node_pair:
        #     print("warning")
        i += 1
        print(i)
        
        #combine same graph
        # need2remove = []
        # for func_list_item_1 in func_list:
        #     for func_list_item_2 in func_list:
        #         if 
        
        
        
        if node_pair[0] not in obj_func_set and node_pair[1] not in cdd_func_set:
            add_flag = 0
            for func_list_item in func_list:
                if have_subgraph_edge(func_list_item, node_pair, object_graph, candidate_graph):
                    add_flag = 1
                    obj_func_set.add(node_pair[0])
                    cdd_func_set.add(node_pair[1])
                    func_list_item.append(node_pair)
                    if str(func_list_item) not in func_list_set:
                        func_list_set.add(str(func_list_item))
            if add_flag == 0:
                func_list.append([])
                func_list[-1].append(node_pair)
                func_list_set.add(str(func_list[-1]))
                obj_func_set.add(node_pair[0])
                cdd_func_set.add(node_pair[1])
                    
            # if i != 1:
            #     func_list = del_wrong_func_pair(func_list, object_graph, candidate_graph)
            #     obj_func_set = set()
            #     cdd_func_set = set()
            #     for func_list_item in func_list:
            #         obj_func_set.add(func_list_item[0])
            #     for func_list_item in func_list:
            #         cdd_func_set.add(func_list_item[1])
        elif node_pair[1] not in cdd_func_set:
            func_list_tmp = copy.deepcopy(func_list)
            
            for func_list_item_tmp in func_list_tmp:
                for func_pair_tmp in func_list_item_tmp:
                    if func_pair_tmp[0] == node_pair[0]:
                        func_list.append([])
                        func_list[-1] = copy.deepcopy(func_list_item_tmp)
                        func_list[-1].remove(func_pair_tmp)
                        if func_list[-1]==[] or have_subgraph_edge(func_list[-1], node_pair, object_graph, candidate_graph):
                            func_list[-1].append(node_pair)
                            if str(func_list[-1]) in func_list_set:
                                func_list.remove(func_list[-1])
                            else:
                                func_list_set.add(str(func_list[-1]))
                                cdd_func_set.add(node_pair[1])
                            break
                        else:
                            func_list.remove(func_list[-1])
        elif node_pair[0] not in obj_func_set:
            func_list_tmp = copy.deepcopy(func_list)
            
            for func_list_item_tmp in func_list_tmp:
                for func_pair_tmp in func_list_item_tmp:
                    if func_pair_tmp[1] == node_pair[1]:
                        func_list.append([])
                        func_list[-1] = copy.deepcopy(func_list_item_tmp)
                        func_list[-1].remove(func_pair_tmp)
                        if func_list[-1]==[] or have_subgraph_edge(func_list[-1], node_pair, object_graph, candidate_graph):
                            func_list[-1].append(node_pair)
                            if str(func_list[-1]) in func_list_set:
                                func_list.remove(func_list[-1])
                            else:
                                func_list_set.add(str(func_list[-1]))
                                obj_func_set.add(node_pair[0])
                            break
                        else:
                            func_list.remove(func_list[-1])
        else:
            func_list_tmp = copy.deepcopy(func_list)
            
            for func_list_item_tmp in func_list_tmp:
                for func_pair_tmp in func_list_item_tmp:
                    if func_pair_tmp[1] == node_pair[1] or func_pair_tmp[0] == node_pair[0]:
                        func_list.append([])
                        func_list[-1] = copy.deepcopy(func_list_item_tmp)
                        func_list[-1].remove(func_pair_tmp)
                        if func_list[-1]==[] or have_subgraph_edge(func_list[-1], node_pair, object_graph, candidate_graph):
                            func_list[-1].append(node_pair)
                            if str(func_list[-1]) in func_list_set:
                                func_list.remove(func_list[-1])
                            else:
                                func_list_set.add(str(func_list[-1]))
                        else:
                            func_list.remove(func_list[-1])
    return func_list

def is_str_sim(list1, list2, cnum):
    b = difflib.SequenceMatcher(None, str(list1), str(list2)).quick_ratio()
    
    if b >= cnum:
        return True
    else:
        return False

def is_list_sim(list1, list2, cnum):
    a = [x for x in list1 if x in list2] 
    #b = [y for y in (list1 + list2) if y not in a]
    
    
    if len(a)*200/(len(list1) + len(list2)) >= cnum:
        return True
    else:
        return False



def combine_fcg_str(func_list, cnum):
    
    return_func_list = []
    
    for func_list_item in func_list:
        combine_flag = 0
        for return_func_list_item in return_func_list:
            if is_str_sim(return_func_list_item, func_list_item, cnum):
                combine_flag = 1
                break
                # if len(return_func_list_item) < 10:
                #     for func_pair in func_list_item:
                #         if func_pair not in return_func_list_item:
                #             return_func_list_item.append(func_pair)
        if return_func_list!=[] and combine_flag == 1:
            continue
        return_func_list.append(func_list_item)
        
    
    
    return return_func_list


def combine_fcg(func_list, cnum):
    
    return_func_list = []
    
    for func_list_item in func_list:
        combine_flag = 0
        for return_func_list_item in return_func_list:
            if is_list_sim(return_func_list_item, func_list_item, cnum):
                combine_flag = 1
                break
                # if len(return_func_list_item) < 10:
                #     for func_pair in func_list_item:
                #         if func_pair not in return_func_list_item:
                #             return_func_list_item.append(func_pair)
        if return_func_list!=[] and combine_flag == 1:
            continue
        return_func_list.append(func_list_item)
        
    
    
    return return_func_list


def combine_big_fcg(func_list, cnum):
    
    return_func_list = []
    
    for func_list_item in func_list:
        combine_flag = 0
        if len(func_list_item) < 4:
            for return_func_list_item in return_func_list:
                if is_list_sim(return_func_list_item, func_list_item, cnum):
                    combine_flag = 1
                    break
                    # if len(return_func_list_item) < 10:
                    #     for func_pair in func_list_item:
                    #         if func_pair not in return_func_list_item:
                    #             return_func_list_item.append(func_pair)
            if return_func_list!=[] and combine_flag == 1:
                continue
        return_func_list.append(func_list_item)
        
    
    
    return return_func_list


def add_child(node, g, feature_list, edge_list, walked_map):
    if node not in walked_map:
        walked_map.add(node)
        
        out_edges = list(g.out_edges(node))
        edges = []
        for out_edge in out_edges:
            feature_list.append(out_edge[1])
            edge_list.append([])
            edges.append(feature_list.index(out_edge[1]))
        edge_list[feature_list.index(node)] = edges
        for out_edge in out_edges:
            add_child(out_edge[1], g, feature_list, edge_list, walked_map)


def get_subgraph(node, object_graph):
    # walked_map_set = set()
    # child_node_list = get_children_list(node, object_graph, walked_map_set)
    gemini_feature_dict = {}
    gemini_feature_dict["feature"] = []
    gemini_feature_dict["succs"] = []
    walked_map = set()
    gemini_feature_dict["feature"].append(node)
    gemini_feature_dict["succs"].append([])
    add_child(node, object_graph, gemini_feature_dict["feature"], gemini_feature_dict["succs"], walked_map)
    gemini_feature_dict["n_num"] = len(gemini_feature_dict["feature"])
    
    return gemini_feature_dict

# 加入前向节点，合并算法
def get_subgraph_func_list_v2(matched_func_ingraph_list, object_graph, candidate_graph):
    
    obj_func_set = set()
    cdd_func_set = set()
    func_list = []
    func_list_set = set()
    
    
    i = 0
    
    for node_pair in matched_func_ingraph_list:
        # if "_start" in node_pair:
        #     print("warning")
        i += 1
        print(i)
        
        #combine same graph
        combine_num = 0
        for func_list_item in func_list:
            if len(func_list_item) > 1:
                combine_num += 1
                if combine_num > 50:
                    func_list = combine_fcg(func_list, 80)
                    obj_func_set = set()
                    cdd_func_set = set()
                    func_list_set = set()
                    for func_list_item in func_list:
                        func_list_set.add(str(func_list_item))
                        for func_pair in func_list_item:
                            obj_func_set.add(func_pair[0])
                            cdd_func_set.add(func_pair[1])
                    break
        
        
        
        if node_pair[0] not in obj_func_set and node_pair[1] not in cdd_func_set:
            add_flag = 0
            for func_list_item in func_list:
                if have_subgraph_edge(func_list_item, node_pair, object_graph, candidate_graph):
                    add_flag = 1
                    obj_func_set.add(node_pair[0])
                    cdd_func_set.add(node_pair[1])
                    func_list_item.append(node_pair)
                    if str(func_list_item) not in func_list_set:
                        func_list_set.add(str(func_list_item))
            if add_flag == 0:
                func_list.append([])
                func_list[-1].append(node_pair)
                func_list_set.add(str(func_list[-1]))
                obj_func_set.add(node_pair[0])
                cdd_func_set.add(node_pair[1])
                    
            # if i != 1:
            #     func_list = del_wrong_func_pair(func_list, object_graph, candidate_graph)
            #     obj_func_set = set()
            #     cdd_func_set = set()
            #     for func_list_item in func_list:
            #         obj_func_set.add(func_list_item[0])
            #     for func_list_item in func_list:
            #         cdd_func_set.add(func_list_item[1])
        elif node_pair[1] not in cdd_func_set:
            func_list_tmp = copy.deepcopy(func_list)
            
            for func_list_item_tmp in func_list_tmp:
                for func_pair_tmp in func_list_item_tmp:
                    if func_pair_tmp[0] == node_pair[0]:
                        func_list.append([])
                        func_list[-1] = copy.deepcopy(func_list_item_tmp)
                        func_list[-1].remove(func_pair_tmp)
                        if func_list[-1]==[] or have_subgraph_edge(func_list[-1], node_pair, object_graph, candidate_graph):
                            func_list[-1].append(node_pair)
                            if str(func_list[-1]) in func_list_set:
                                func_list.remove(func_list[-1])
                            else:
                                func_list_set.add(str(func_list[-1]))
                                cdd_func_set.add(node_pair[1])
                            break
                        else:
                            func_list.remove(func_list[-1])
        elif node_pair[0] not in obj_func_set:
            func_list_tmp = copy.deepcopy(func_list)
            
            for func_list_item_tmp in func_list_tmp:
                for func_pair_tmp in func_list_item_tmp:
                    if func_pair_tmp[1] == node_pair[1]:
                        func_list.append([])
                        func_list[-1] = copy.deepcopy(func_list_item_tmp)
                        func_list[-1].remove(func_pair_tmp)
                        if func_list[-1]==[] or have_subgraph_edge(func_list[-1], node_pair, object_graph, candidate_graph):
                            func_list[-1].append(node_pair)
                            if str(func_list[-1]) in func_list_set:
                                func_list.remove(func_list[-1])
                            else:
                                func_list_set.add(str(func_list[-1]))
                                obj_func_set.add(node_pair[0])
                            break
                        else:
                            func_list.remove(func_list[-1])
        else:
            func_list_tmp = copy.deepcopy(func_list)
            
            for func_list_item_tmp in func_list_tmp:
                for func_pair_tmp in func_list_item_tmp:
                    if func_pair_tmp[1] == node_pair[1] or func_pair_tmp[0] == node_pair[0]:
                        func_list.append([])
                        func_list[-1] = copy.deepcopy(func_list_item_tmp)
                        func_list[-1].remove(func_pair_tmp)
                        if func_list[-1]==[] or have_subgraph_edge(func_list[-1], node_pair, object_graph, candidate_graph):
                            func_list[-1].append(node_pair)
                            if str(func_list[-1]) in func_list_set:
                                func_list.remove(func_list[-1])
                            else:
                                func_list_set.add(str(func_list[-1]))
                        else:
                            func_list.remove(func_list[-1])
    
    # combine
    combine_num = 0
    for func_list_item in func_list:
            if len(func_list_item) > 1:
                combine_num += 1
                if combine_num > 50:
                    func_list = combine_fcg(func_list, 80)
                    break
    
    
    
    return func_list




#新增节点时按顺序插入
#20221028 11：20 修复重复节点提前break bug
#20221028 11:39 存在bug：出现重复节点只替换重复节点list，不跟其他的比较是否相连
def get_subgraph_func_list_v3(matched_func_ingraph_list, object_graph, candidate_graph):
    
    obj_func_set = set()
    cdd_func_set = set()
    func_list = []
    func_list_set = set()
    
    
    i = 0
    
    for node_pair in tqdm(matched_func_ingraph_list):
        # if "_start" in node_pair:
        #     print("warning")
        i += 1
        # if "BZ2_bzDecompress" in node_pair or "BZ2_decompress" in node_pair:#if i == 33:
        #     print("debug")
        #print(i)
        
        #combine same graph
        combine_num = 0
        for func_list_item in func_list:
            if len(func_list_item) > 1:
                combine_num += 1
                if combine_num > 50:
                    print("----before combine:"+str(len(func_list)))
                    func_list = combine_fcg(func_list, 80)
                    print("----after combine:"+str(len(func_list)))
                    obj_func_set = set()
                    cdd_func_set = set()
                    func_list_set = set()
                    for func_list_item in func_list:
                        func_list_set.add(str(func_list_item))
                        for func_pair in func_list_item:
                            obj_func_set.add(func_pair[0])
                            cdd_func_set.add(func_pair[1])
                    break
        
        
        
        if node_pair[0] not in obj_func_set and node_pair[1] not in cdd_func_set:
            add_flag = 0
            for func_list_item in func_list:
                (is_connected_flag, insert_index) = have_subgraph_edge_v2(func_list_item, node_pair, object_graph, candidate_graph)
                if is_connected_flag:
                    add_flag = 1
                    obj_func_set.add(node_pair[0])
                    cdd_func_set.add(node_pair[1])
                    func_list_item.insert(insert_index, node_pair)
                    if str(func_list_item) not in func_list_set:
                        func_list_set.add(str(func_list_item))
            if add_flag == 0:
                func_list.append([])
                func_list[-1].append(node_pair)
                func_list_set.add(str(func_list[-1]))
                obj_func_set.add(node_pair[0])
                cdd_func_set.add(node_pair[1])
                    
            # if i != 1:
            #     func_list = del_wrong_func_pair(func_list, object_graph, candidate_graph)
            #     obj_func_set = set()
            #     cdd_func_set = set()
            #     for func_list_item in func_list:
            #         obj_func_set.add(func_list_item[0])
            #     for func_list_item in func_list:
            #         cdd_func_set.add(func_list_item[1])
        elif node_pair[1] not in cdd_func_set:
            func_list_tmp = copy.deepcopy(func_list)
            
            for func_list_item_tmp in func_list_tmp:
                for func_pair_tmp in func_list_item_tmp:
                    if func_pair_tmp[0] == node_pair[0]:
                        func_list.append([])
                        func_list[-1] = copy.deepcopy(func_list_item_tmp)
                        func_list[-1].remove(func_pair_tmp)
                        (is_connected_flag, insert_index) = have_subgraph_edge_v2(func_list[-1], node_pair, object_graph, candidate_graph)
                        if func_list[-1]==[] or is_connected_flag:
                            func_list[-1].insert(insert_index, node_pair)
                            if str(func_list[-1]) in func_list_set:
                                func_list.remove(func_list[-1])
                            else:
                                func_list_set.add(str(func_list[-1]))
                                cdd_func_set.add(node_pair[1])
                            # break
                        else:
                            func_list.remove(func_list[-1])
        elif node_pair[0] not in obj_func_set:
            func_list_tmp = copy.deepcopy(func_list)
            
            for func_list_item_tmp in func_list_tmp:
                for func_pair_tmp in func_list_item_tmp:
                    if func_pair_tmp[1] == node_pair[1]:
                        func_list.append([])
                        func_list[-1] = copy.deepcopy(func_list_item_tmp)
                        func_list[-1].remove(func_pair_tmp)
                        (is_connected_flag, insert_index) = have_subgraph_edge_v2(func_list[-1], node_pair, object_graph, candidate_graph)
                        if func_list[-1]==[] or is_connected_flag:
                            func_list[-1].insert(insert_index, node_pair)
                            if str(func_list[-1]) in func_list_set:
                                func_list.remove(func_list[-1])
                            else:
                                func_list_set.add(str(func_list[-1]))
                                obj_func_set.add(node_pair[0])
                            # break
                        else:
                            func_list.remove(func_list[-1])
        else:
            func_list_tmp = copy.deepcopy(func_list)
            
            for func_list_item_tmp in func_list_tmp:
                for func_pair_tmp in func_list_item_tmp:
                    if func_pair_tmp[1] == node_pair[1] or func_pair_tmp[0] == node_pair[0]:
                        func_list.append([])
                        func_list[-1] = copy.deepcopy(func_list_item_tmp)
                        func_list[-1].remove(func_pair_tmp)
                        (is_connected_flag, insert_index) = have_subgraph_edge_v2(func_list[-1], node_pair, object_graph, candidate_graph)
                        if func_list[-1]==[] or is_connected_flag:
                            func_list[-1].insert(insert_index, node_pair)
                            if str(func_list[-1]) in func_list_set:
                                func_list.remove(func_list[-1])
                            else:
                                func_list_set.add(str(func_list[-1]))
                        else:
                            func_list.remove(func_list[-1])
    
    # combine
    combine_num = 0
    for func_list_item in func_list:
            if len(func_list_item) > 1:
                combine_num += 1
                if combine_num > 50:
                    func_list = combine_fcg(func_list, 80)
                    break
    
    
    
    return func_list

# def use_libdb(matched_func_ingraph_list, object_graph, candidate_graph):
#     # matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    
#     obj_sim_funcs = []
#     cdd_sim_funcs = []
#     for func_pair in matched_func_ingraph_list:
#         if func_pair[0] not in obj_sim_funcs:
#             obj_sim_funcs.append(func_pair[0])
#         if func_pair[1] not in cdd_sim_funcs:
#             cdd_sim_funcs.append(func_pair[1])
    
#     obj_afcg = get_afcg(obj_sim_funcs, object_graph)
#     cdd_afcg = get_afcg(cdd_sim_funcs, candidate_graph)
    
#     alignment_list = []
#     for func_pair in matched_func_ingraph_list:
#         if func_pair[0] in obj_afcg:
#             if func_pair[0] not in alignment_list:
#                 obj_node_children = obj_afcg[func_pair[0]]

def isrd_method(matched_func_list, candidate_fcg):
    matched_len = len(matched_func_list)
    all_func_num = len(candidate_fcg)
    
    if matched_len/all_func_num > 0.2:
        return True
    else:
        return False



def get_gemini_dict(object_cdd_func_dict):
    cdd_project_dict = {}
    for matched_item in object_cdd_func_dict:
        cdd_item = matched_item.split("||||")[1].split("----")[0]
        # obj_func_item = matched_item.split("||||")[0].split("----")[1]
        cdd_func_item = matched_item.split("||||")[1].split("----")[1]
        if cdd_item not in cdd_project_dict:
            cdd_project_dict[cdd_item] = []
        #cdd_project_dict[cdd_item].append(["".join(obj_func_item.split("-")[2:]), "".join(cdd_func_item.split("-")[2:])])
        if cdd_func_item not in cdd_project_dict[cdd_item]:
            cdd_project_dict[cdd_item].append(cdd_func_item)
    return cdd_project_dict

def get_subgraph_func_list_v4(matched_func_ingraph_list, object_graph, candidate_graph):
    
    obj_func_set = set()
    cdd_func_set = set()
    func_list = []
    func_list_set = set()
    
    alignment_max = False
    i = 0
    
    for node_pair in tqdm(matched_func_ingraph_list):
        # if "_start" in node_pair:
        #     print("warning")
        i += 1
        # if "BZ2_bzDecompress" in node_pair or "BZ2_decompress" in node_pair:#if i == 33:
        #     print("debug")
        #print(i)
        
        #combine same graph
        combine_num = 0
        combine_str_num = 0
        for func_list_item in func_list:
            combine_str_num += 1
            if len(func_list_item) > 1:
                combine_num += 1
                if combine_num > 50:
                    print("----before combine:"+str(len(func_list)))
                    func_list = combine_fcg(func_list, 80)
                    print("----after combine:"+str(len(func_list)))
                    obj_func_set = set()
                    cdd_func_set = set()
                    func_list_set = set()
                    for func_list_item in func_list:
                        func_list_set.add(str(func_list_item))
                        for func_pair in func_list_item:
                            obj_func_set.add(func_pair[0])
                            cdd_func_set.add(func_pair[1])
                    break
            # if combine_str_num > 200:
            #     print("----before str combine:"+str(len(func_list)))
            #     func_list = combine_fcg_str(func_list, 0.8)
            #     print("----after str combine:"+str(len(func_list)))
            #     obj_func_set = set()
            #     cdd_func_set = set()
            #     func_list_set = set()
            #     for func_list_item in func_list:
            #         func_list_set.add(str(func_list_item))
            #         for func_pair in func_list_item:
            #             obj_func_set.add(func_pair[0])
            #             cdd_func_set.add(func_pair[1])
            #     break
                
        combine_num = 0
        combine_str_num = 0
        for func_list_item in func_list:
            if len(func_list_item) > 1:
                combine_num += 1
                if combine_num > 50:
                    print("----before combine:"+str(len(func_list)))
                    func_list = combine_fcg(func_list, 51)
                    print("----after combine:"+str(len(func_list)))
                    obj_func_set = set()
                    cdd_func_set = set()
                    func_list_set = set()
                    for func_list_item in func_list:
                        func_list_set.add(str(func_list_item))
                        for func_pair in func_list_item:
                            obj_func_set.add(func_pair[0])
                            cdd_func_set.add(func_pair[1])
                    break
            # if combine_str_num > 200:
            #     print("----before str combine:"+str(len(func_list)))
            #     func_list = combine_fcg_str(func_list, 0.7)
            #     print("----after str combine:"+str(len(func_list)))
            #     obj_func_set = set()
            #     cdd_func_set = set()
            #     func_list_set = set()
            #     for func_list_item in func_list:
            #         func_list_set.add(str(func_list_item))
            #         for func_pair in func_list_item:
            #             obj_func_set.add(func_pair[0])
            #             cdd_func_set.add(func_pair[1])
            #     break
        
        combine_num = 0
        for func_list_item in func_list:
            if len(func_list_item) > 1:
                combine_num += 1
                if combine_num > 50:
                    break
        
        if combine_num > 50:
            alignment_max = True
            break
        
        add_flag = 0
        func_list_to_add = []
        func_list_to_insert = {}
        for func_list_item in func_list:
            obj_func_set = [i[0] for i in func_list_item]
            cdd_func_set = [i[1] for i in func_list_item]
            func_list_item_to_add = []
            func_list_item_to_insert = []
            if node_pair[0] not in obj_func_set and node_pair[1] not in cdd_func_set:  
                # for func_list_item in func_list:
                (is_connected_flag, insert_index) = have_subgraph_edge_v3(func_list_item, node_pair, object_graph, candidate_graph)
                if is_connected_flag:
                    add_flag = 1
                    # obj_func_set.add(node_pair[0])
                    # cdd_func_set.add(node_pair[1])
                    func_list_item_to_insert = copy.deepcopy(func_list_item)
                    func_list_item_to_insert.insert(insert_index, node_pair)
                    # func_list_item.insert(insert_index, node_pair)
                    if str(func_list_item_to_insert) not in func_list_set:
                        func_list_set.add(str(func_list_item_to_insert))
                        func_list_to_insert[str(func_list_item)] = func_list_item_to_insert
                    # obj_func_set.add(node_pair[0])
                    # cdd_func_set.add(node_pair[1])
                        
            elif node_pair[1] not in cdd_func_set:
                # func_list_tmp = copy.deepcopy(func_list)
                # for func_list_item_tmp in func_list_tmp:
                for func_pair_tmp in func_list_item:
                    if func_pair_tmp[0] == node_pair[0]:
                        func_list_item_to_add = copy.deepcopy(func_list_item)
                        func_list_item_to_add.remove(func_pair_tmp)
                        (is_connected_flag, insert_index) = have_subgraph_edge_v3(func_list_item_to_add, node_pair, object_graph, candidate_graph)
                        if (func_list_item_to_add==[] and add_flag == 0) or is_connected_flag:
                            func_list_item_to_add.insert(insert_index, node_pair)
                            if str(func_list_item_to_add) in func_list_set:
                                func_list_item_to_add = []
                            else:
                                add_flag = 1
                                func_list_set.add(str(func_list_item_to_add))
                                func_list_to_add.append(func_list_item_to_add)
                                # cdd_func_set.add(node_pair[1])
                            break
                        else:
                            func_list_item_to_add = []
            elif node_pair[0] not in obj_func_set:
                for func_pair_tmp in func_list_item:
                    if func_pair_tmp[1] == node_pair[1]:
                        func_list_item_to_add = copy.deepcopy(func_list_item)
                        func_list_item_to_add.remove(func_pair_tmp)
                        (is_connected_flag, insert_index) = have_subgraph_edge_v3(func_list_item_to_add, node_pair, object_graph, candidate_graph)
                        if (func_list_item_to_add==[] and add_flag == 0) or is_connected_flag:
                            func_list_item_to_add.insert(insert_index, node_pair)
                            if str(func_list_item_to_add) in func_list_set:
                                func_list_item_to_add = []
                            else:
                                add_flag = 1
                                func_list_set.add(str(func_list_item_to_add))
                                func_list_to_add.append(func_list_item_to_add)
                                # cdd_func_set.add(node_pair[1])
                            break
                        else:
                            func_list_item_to_add = []
            else:
                for func_pair_tmp in func_list_item:
                    if func_pair_tmp[1] == node_pair[1] or func_pair_tmp[0] == node_pair[0]:
                        func_list_item_to_add = copy.deepcopy(func_list_item)
                        func_list_item_to_add.remove(func_pair_tmp)
                        (is_connected_flag, insert_index) = have_subgraph_edge_v3(func_list_item_to_add, node_pair, object_graph, candidate_graph)
                        if (func_list_item_to_add==[] and add_flag == 0) or is_connected_flag:
                            func_list_item_to_add.insert(insert_index, node_pair)
                            if str(func_list_item_to_add) in func_list_set:
                                func_list_item_to_add = []
                            else:
                                add_flag = 1
                                func_list_set.add(str(func_list_item_to_add))
                                func_list_to_add.append(func_list_item_to_add)
                                # cdd_func_set.add(node_pair[1])
                            break
                        else:
                            func_list_item_to_add = []
            # if func_list_item_to_add != []:
            #     add_flag = 1
            #     func_list_to_add.append(func_list_item_to_add)
        if func_list_to_add != []:
            func_list.extend(func_list_to_add)
        if func_list_to_insert != {}:
            for func_list_to_insert_item in func_list_to_insert:
                func_list[func_list.index(eval(func_list_to_insert_item))] = func_list_to_insert[func_list_to_insert_item]
        if add_flag == 0:
            func_list.append([])
            func_list[-1].append(node_pair)
            func_list_set.add(str(func_list[-1]))
        
    # combine
    combine_num = 0
    for func_list_item in func_list:
            if len(func_list_item) > 1:
                combine_num += 1
                if combine_num > 50:
                    func_list = combine_fcg(func_list, 80)
                    break
    
    
    if alignment_max:
        return [], alignment_max
    else:
        return func_list, alignment_max




# 修复列表分组打乱列表顺序的bug
# 改为按列表顺序看是否连接(只检测两点相连，或列表一个接一个连起来(必须是线性连接关系，非线性用其他函数，判断每个节点与前或后有链接即可))
def is_subgraph_edge_v2(sub_graph_func_list, graph):
    #node_group = list(itertools.permutations(sub_graph_func_list, 2))
    for node in sub_graph_func_list:
        if node != sub_graph_func_list[-1]:
            walked_map_set = set()
            if is_connected([node,sub_graph_func_list[sub_graph_func_list.index(node)+1]], graph, walked_map_set):
                return True
    return False


# 添加计算距离
# 通过djtsl算法沿最短路径找
def is_subgraph_edge_v3(sub_graph_func_list, graph):
    #node_group = list(itertools.permutations(sub_graph_func_list, 2))
    if len(sub_graph_func_list) == 2:
        try:
            shortest_path = nx.shortest_path(graph, source=sub_graph_func_list[0],target=sub_graph_func_list[1])
            distance = len(shortest_path)
            return (True, distance)
        except nx.exception.NetworkXNoPath as e:
            return (False, -1)
            # walked_map_set = set()
            # if is_connected([node,sub_graph_func_list[sub_graph_func_list.index(node)+1]], graph, walked_map_set):
            #     return True



def is_subgraph_edge(sub_graph_func_list, graph):
    node_group = list(itertools.permutations(sub_graph_func_list, 2))
    for node_pair in node_group:
        if is_connected(node_pair, graph, set()):
            return True
    return False
    
    
def is_connected(node_pair, graph, walked_map_set):
    if node_pair[0] not in walked_map_set:
        walked_map_set.add(node_pair[0])
        child_node_list = get_child_node(node_pair[0], graph)
        for child_node in child_node_list:
            if child_node == node_pair[1]:
                return True
            else:
                if is_connected([child_node,node_pair[1]], graph, walked_map_set):
                    return True
        return False
    return False
      



def get_common_edge(sub_graph_func_list, obj_subgraph_edge_list, cdd_subgraph_edge_list):
    pass