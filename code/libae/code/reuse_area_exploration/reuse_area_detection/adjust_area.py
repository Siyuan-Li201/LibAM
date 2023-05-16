import networkx as nx
import copy
import torch
import numpy as np
import os, sys
sys.path.append(".")
from settings import DATA_PATH, WORK_PATH
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn.functional as F
from multiprocessing import Process
import pickle

import json

def add_child(node, g, feature_list, edge_list, nnum,walked_map):
    if node not in walked_map:
        walked_map.add(node)
        out_edges = list(g.out_edges(node))
        edges = []
        for out_edge in out_edges:
            if out_edge[1] not in feature_list:
                feature_list.append(out_edge[1])
                nnum += 1
                edge_list.append([])
                edge_list[feature_list.index(node)].append(feature_list.index(out_edge[1]))
        #         edges.append(feature_list.index(out_edge[1]))
        # edge_list[feature_list.index(node)] = edges
        # for out_edge in out_edges:
                add_child(out_edge[1], g, feature_list, edge_list, nnum, walked_map)

#修复bug: 父节点需要插入到列表头，其余连边加1
#存在bias：多个父亲时将同时插入，并把最后一个父亲当作新的头节点，其他父亲没有继续探索，且两个父亲可能不匹配，之后考虑对多个父亲分开考虑
def up_get_fcg_in9(bround_point_list, old_minifcg, binary_fcg:nx.DiGraph, in9_func_list):
    minifcg = copy.deepcopy(old_minifcg)
    new_bround_point = []
    # minifcg = old_minifcg
    has_diff_father = 0
    for bround_point in bround_point_list:
        in_edges = binary_fcg.in_edges(minifcg["feature"][bround_point])
        # has_diff_father = 0
        if len(in_edges) != 0:
            for in_edge in in_edges:
                if in_edge[0] not in in9_func_list and in_edge[0] not in minifcg["feature"]:
                    has_diff_father = 1
                    minifcg["feature"].append(in_edge[0])
                    minifcg["n_num"] += 1
                    new_point = len(minifcg["feature"]) - 1
                    minifcg["succs"].append([bround_point])
                    new_bround_point.append(new_point)
                    # if has_diff_father == 1:
                    # nodes = [in_edge[0]]
                    # succs = [[]]
                    add_child(in_edge[0], binary_fcg, minifcg["feature"], minifcg["succs"], minifcg["n_num"], set())
                    # for node in nodes:
                    #     if node not in minifcg["feature"]:
                    #         minifcg["feature"].append(node)
                    #         minifcg["n_num"] += 1
                    #         minifcg["succs"].append([])
                    # for node in nodes:
                    #     node_index = nodes.index(node)
                    #     node_index_new = minifcg["feature"].index(node)
                    #     # if node_index_new not in new_bround_point:
                    #     #     new_bround_point.append(node_index_new)
                    #     succ = succs[node_index]
                    #     for s in succ:
                    #         new_node = nodes[s]
                    #         new_node_index = minifcg["feature"].index(new_node)
                    #         if new_node_index not in minifcg["succs"][node_index_new]:
                    #             minifcg["succs"][node_index_new].append(new_node_index)
                minifcg["n_num"] = len(minifcg["feature"])
    return has_diff_father, minifcg, new_bround_point
    # return has_diff_father, minifcg, new_bround_point

def down_get_fcg(old_minifcg):
    minifcg = copy.deepcopy(old_minifcg)
    if minifcg["n_num"] > 1:
        minifcg["feature"] = minifcg["feature"][1:minifcg["n_num"]]
        minifcg["succs"] = minifcg["succs"][1:minifcg["n_num"]]
        minifcg["n_num"] -= 1
        succs = []
        for i in range(len(minifcg["succs"])):
            succ = []
            for j in range(len(minifcg["succs"][i])):
                node = minifcg["succs"][i][j] - 1
                if node > 0:
                    succ.append(node)
            succs.append(succ)
        minifcg["succs"] = succs
        return 1, minifcg
    return 0, minifcg
   
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

def calculate_gnn_score(tar_fcg, cdd_fcg, func_embeddings, gnn):
    feature = []
    for func in tar_fcg["feature"]:
        func_name = tar_fcg["file_name"] + "|||" + func
        if func_name in func_embeddings:
            embed = func_embeddings[func_name][0]
        else:
            embed = list(np.array([0.001 for i in range(64)]))
        feature.append(embed)
        tar_fcg_temp = copy.deepcopy(tar_fcg)
        tar_fcg_temp["embeddings"] = feature
    obj_embedding = embed_by_feat_torch(tar_fcg_temp, gnn)

    feature = []
    for func in cdd_fcg["feature"]:
        func_name = cdd_fcg["file_name"] + "|||" + func
        if func_name in func_embeddings:
            embed = func_embeddings[func_name][0]
        else:
            embed = list(np.array([0.001 for i in range(64)]))
        feature.append(embed)
        cdd_fcg_temp = copy.deepcopy(cdd_fcg)
        cdd_fcg_temp["embeddings"] = feature
    cdd_embedding = embed_by_feat_torch(cdd_fcg_temp, gnn)
    gnn_score = F.cosine_similarity(obj_embedding, cdd_embedding, eps=1e-10, dim=1)
    gnn_score = (1 + gnn_score.cpu().detach().numpy()[0]) / 2.0
    return gnn_score



def calculate_rarm_score_combine(align_num, tar_fcg, cdd_fcg, func_embeddings, gnn):
    gnn_score = calculate_gnn_score(tar_fcg, cdd_fcg, func_embeddings, gnn)
    if gnn_score <= 0.7:
        return 0.0, gnn_score
    tar_fcg_num = tar_fcg["n_num"]
    cdd_fcg_num = cdd_fcg["n_num"]
    fcg_scale = (tar_fcg_num+cdd_fcg_num)/2
    fcg_scale_diff = min(tar_fcg_num, cdd_fcg_num)/max(tar_fcg_num, cdd_fcg_num)

    return RARM_score_combine(align_num, gnn_score, fcg_scale, 10.0, 50.0, fcg_scale_diff), gnn_score

def RARM_score_combine(alignment_num_score, node_gnn_score, fcg_scale, alignment_max, fcg_scale_max, fcg_scale_diff):
    if alignment_num_score > 10:
        alignment_num_score = 10
    if fcg_scale > 50:
        fcg_scale = 50
    alignment_num_score_deal = alignment_num_score/alignment_max
    fcg_scale_deal = fcg_scale/fcg_scale_max
    final_score = alignment_num_score_deal * node_gnn_score * fcg_scale_diff * fcg_scale_deal
    return final_score

def calculate_rarm_score(align_rate, tar_fcg, cdd_fcg, target_binary_fcg, candidate_binary_fcg, func_embeddings, gnn, matched_func_ingraph_list):
    
    obj_sim_funcs_dict = {}
    cdd_sim_funcs_dict = {}
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs_dict:
            obj_sim_funcs_dict[func_pair[0]] = []
        if func_pair[1] not in obj_sim_funcs_dict[func_pair[0]]:
            obj_sim_funcs_dict[func_pair[0]].append(func_pair[1])
        if func_pair[1] not in cdd_sim_funcs_dict:
            cdd_sim_funcs_dict[func_pair[1]] = []
        if func_pair[0] not in cdd_sim_funcs_dict[func_pair[1]]:
            cdd_sim_funcs_dict[func_pair[1]].append(func_pair[0])
    
    obj_com_num = obj_sim_num = 0
    for obj_func in set(tar_fcg["feature"]):
        if obj_func in obj_sim_funcs_dict:
            obj_com_num += 1
            if obj_func in obj_sim_funcs_dict and list(set(obj_sim_funcs_dict[obj_func]).intersection(set(cdd_fcg["feature"]))) != []:
                obj_sim_num += 1
    cdd_com_num = cdd_sim_num = 0
    for cdd_func in set(cdd_fcg["feature"]):
        if cdd_func in cdd_sim_funcs_dict:
            cdd_com_num += 1
            if cdd_func in cdd_sim_funcs_dict and list(set(cdd_sim_funcs_dict[cdd_func]).intersection(set(tar_fcg["feature"]))) != []:
                cdd_sim_num += 1
    
    # com_funcs_scale = (len(obj_com_funcs) + len(cdd_com_funcs)) / 2
    # sim_funcs_scale = (len(obj_sim_funcs) + len(cdd_sim_funcs)) / 2
    if obj_com_num <= cdd_com_num:
        align_rate = obj_sim_num / obj_com_num
    else:
        align_rate = cdd_sim_num / cdd_com_num
    
    gnn_score = calculate_gnn_score(tar_fcg, cdd_fcg, func_embeddings, gnn)
    if gnn_score <= 0.8:
        return 0.0, gnn_score
    raw_tar_fcg_num = target_binary_fcg
    raw_cdd_fcg_num = candidate_binary_fcg
    raw_fcg_scale = (raw_tar_fcg_num + raw_cdd_fcg_num)/2
    tar_fcg_num = tar_fcg["n_num"]
    cdd_fcg_num = cdd_fcg["n_num"]
    align_num_score = 0.3 * align_rate + 0.7
    fcg_scale = (tar_fcg_num+cdd_fcg_num)/2
    fcg_scale_diff = 0.3 * min(tar_fcg_num, cdd_fcg_num)/max(tar_fcg_num, cdd_fcg_num) + 0.7
    if fcg_scale >= raw_fcg_scale:
        fcg_scale_score = 1+((fcg_scale-raw_fcg_scale)/fcg_scale)/2
    else:
        fcg_scale_score = 1+((fcg_scale-raw_fcg_scale)/raw_fcg_scale)/2
    return RARM_score(align_num_score, gnn_score, fcg_scale, 6.0, 50.0, fcg_scale_diff, fcg_scale_score), gnn_score

def RARM_score(alignment_num_score, node_gnn_score, fcg_scale, alignment_max, fcg_scale_max, fcg_scale_diff, fcg_scale_score):
    # if alignment_num_score > 6:
    #     alignment_num_score = 6
    # if fcg_scale > 50:
    #     fcg_scale = 50
    # alignment_num_score_deal = alignment_num_score/alignment_max
    # fcg_scale_deal = fcg_scale/fcg_scale_max
    final_score = alignment_num_score * node_gnn_score# * fcg_scale_diff * fcg_scale_score# * fcg_scale_deal
    return final_score

def calculate_score_in9(bround_point_list, t_d, c_d, reuse_area, target_binary_fcg:nx.DiGraph, candidate_binary_fcg:nx.DiGraph, func_embeddings, gnn, matched_func_ingraph_list, tar_in9_func_list, cdd_in9_func_list):
    # new_bround_point = []
    t_status = 1
    c_status = 1
    raw_tar_fcg_num = reuse_area["obj_fcg"]["n_num"]
    raw_cdd_fcg_num = reuse_area["cdd_fcg"]["n_num"]
    if t_d == "up":
        t_status, tar_fcg, tar_new_bround_point = up_get_fcg_in9(bround_point_list[0], reuse_area["obj_fcg"], target_binary_fcg, tar_in9_func_list)
    elif t_d == "down":
        t_status, tar_fcg = down_get_fcg(reuse_area["obj_fcg"])
    else:
        tar_fcg = reuse_area["obj_fcg"]
        tar_new_bround_point = []
    if c_d == "up":
        c_status, cdd_fcg, cdd_new_bround_point = up_get_fcg_in9(bround_point_list[1], reuse_area["cdd_fcg"], candidate_binary_fcg, cdd_in9_func_list)
    elif c_d == "down":
        c_status, cdd_fcg = down_get_fcg(reuse_area["cdd_fcg"])
    else:
        cdd_fcg = reuse_area["cdd_fcg"]
        cdd_new_bround_point = []
    if t_status != 0 and c_status != 0:
        score, gnn_score = calculate_rarm_score(reuse_area["alignment_rate"], tar_fcg, cdd_fcg, raw_tar_fcg_num, raw_cdd_fcg_num, func_embeddings, gnn, matched_func_ingraph_list)
        status = 1
    else:
        status = 0
        score = -1
        gnn_score = -1
    if tar_new_bround_point == [] and cdd_new_bround_point == []:
        return [status, t_status, c_status], score, tar_fcg, cdd_fcg, gnn_score, []
    else:
        return [status, t_status, c_status], score, tar_fcg, cdd_fcg, gnn_score, [[tar_new_bround_point, cdd_new_bround_point]]


# def combine_reuse_area(reuse_area, target_binary_fcg:nx.DiGraph, candidate_binary_fcg:nx.DiGraph, func_embeddings, gnn):
#     tar_fcg = reuse_area["obj_fcg"]  # 起始函数名称
#     cdd_fcg = reuse_area["cdd_fcg"]
    
#     for i in tar_select:
#         for j in cdd_select:
#             status, score, tf, cf, gnn_score = calculate_score(i, j, reuse_area, target_binary_fcg, candidate_binary_fcg, func_embeddings, gnn)


def get_bround_point(bround_point_list_before, reuse_area, target_binary_fcg, candidate_binary_fcg, align_func_list):
    tar_bround_point_list = []

    if bround_point_list_before != []:
        return bround_point_list_before
    else:
        for node_pair in align_func_list:
            if eval(node_pair)[0] in reuse_area["obj_fcg"]["feature"] and eval(node_pair)[1] in reuse_area["cdd_fcg"]["feature"]:
                tar_flag = cdd_flag = False
                tar_in_edges = target_binary_fcg.in_edges(eval(node_pair)[0])
                
                for tar_in_edge in tar_in_edges:
                    if tar_in_edge[0] not in reuse_area["obj_fcg"]["feature"]:
                        tar_flag = True
                        break
                cdd_in_edges = candidate_binary_fcg.in_edges(eval(node_pair)[1])
                
                for cdd_in_edge in cdd_in_edges:
                    if cdd_in_edge[0] not in reuse_area["cdd_fcg"]["feature"]:
                        cdd_flag = True
                        break
                    
                if cdd_flag and tar_flag:
                    tar_bround_point_list.append([[reuse_area["obj_fcg"]["feature"].index(eval(node_pair)[0])], [reuse_area["cdd_fcg"]["feature"].index(eval(node_pair)[1])]])
        return tar_bround_point_list
        # return [[[0], [0]]]
    
    # for tar_func in reuse_area["obj_fcg"]["feature"]:
    #     in_edges = target_binary_fcg.in_edges(tar_func)
    #     for in_edge in in_edges:
    #         if in_edge[0] not in reuse_area["obj_fcg"]["feature"]:
    #             tar_bround_point_list.append(tar_func)
    #             break
    
    # return tar_bround_point_list


def adjust_reuse_area_iter_in9(bround_point_list, reuse_area, target_binary_fcg:nx.DiGraph, candidate_binary_fcg:nx.DiGraph, func_embeddings, gnn, matched_func_ingraph_list, tar_in9_func_list, cdd_in9_func_list):
    
    # bround_point_list_temp = copy.deepcopy(bround_point_list)
    
    tar_select = ["wait", "up"]
    cdd_select = ["wait", "up"]
    max_score_dire = ("wait", "wait")
    max_score = 0
    max_gnn_score = 0
    max_new_bround_point = []
    
    for i in tar_select:
        for j in cdd_select:
            status, score, tf, cf, gnn_score, new_bround_point = calculate_score_in9(bround_point_list, i, j, reuse_area, target_binary_fcg, candidate_binary_fcg, func_embeddings, gnn, matched_func_ingraph_list, tar_in9_func_list, cdd_in9_func_list)
            # if status[0] == 0:
            #     return reuse_area, []
                # return reuse_area
            if status[1] == 0:
                tar_select.remove(i)
                i = "wait"
            if status[2] == 0:
                cdd_select.remove(j)
                j = "wait"
                
            new_scale = (len(set(tf["feature"])) + len(set(tf["feature"]))) / 2
            old_scale = (reuse_area["fcg_scale"][0] + reuse_area["fcg_scale"][1]) / 2
            
            if (abs(score - max_score) < 0.01 and new_scale > old_scale) or score - max_score >= 0.5:
                max_score_dire = (i, j)
                max_score = score
                max_gnn_score = gnn_score
                tar_fcg = tf
                cdd_fcg = cf
                max_new_bround_point = new_bround_point
    if max_score_dire == ("wait", "wait"):
        return reuse_area, []
    # if max_score_dire[0] == "up" and "down" in tar_select:
    #     tar_select.remove("down")
    # if max_score_dire[0] == "down" and "up" in tar_select:
    #     tar_select.remove("up")
    # if max_score_dire[1] == "up" and "down" in cdd_select:
    #     cdd_select.remove("down")
    # if max_score_dire[1] == "down" and "up" in cdd_select:
    #     cdd_select.remove("up")
    else:
        reuse_area["obj_fcg"] = tar_fcg
        reuse_area["cdd_fcg"] = cdd_fcg
        reuse_area["gnn_score"] = max_gnn_score
        reuse_area["final_score"] = max_score
        reuse_area["fcg_scale"] = [len(set(tar_fcg["feature"])), len(set(cdd_fcg["feature"]))]
        # for new_point in max_new_bround_point:
        #     if new_point not in bround_point_list_temp:
        #         bround_point_list_temp.append(new_point)
        # reuse_area = adjust_reuse_area_iter(reuse_area, target_binary_fcg, candidate_binary_fcg, func_embeddings, gnn)
    return reuse_area, max_new_bround_point

def adjust_reuse_area_in9(bround_point_list_before, reuse_area, target_binary_fcg:nx.DiGraph, candidate_binary_fcg:nx.DiGraph, func_embeddings, gnn, matched_func_ingraph_list, align_func_list, tar_in9_func_list, cdd_in9_func_list):
    # tar_fcg = reuse_area["obj_fcg"]  # 起始函数名称
    # cdd_fcg = reuse_area["cdd_fcg"]
    
    bround_point_list_list = get_bround_point(bround_point_list_before, reuse_area, target_binary_fcg, candidate_binary_fcg, align_func_list, tar_in9_func_list, cdd_in9_func_list)
    # if len(bround_point_list_list) > 1:
    #     print("warning")
    for bround_point_list in bround_point_list_list:
        reuse_area, bround_point_list_after = adjust_reuse_area_iter_in9(bround_point_list, reuse_area, target_binary_fcg, candidate_binary_fcg, func_embeddings, gnn, matched_func_ingraph_list, tar_in9_func_list, cdd_in9_func_list)
        
        if bround_point_list_after == []:
            continue
        else:
            reuse_area = adjust_reuse_area_in9(bround_point_list_after, reuse_area, target_binary_fcg, candidate_binary_fcg, func_embeddings, gnn, matched_func_ingraph_list, align_func_list, tar_in9_func_list, cdd_in9_func_list)
            continue
    return reuse_area
    
    
    
    
    
def combine_reuse_areas(rein_files, rein_path, fcgs, save_path, func_embeddings, gnn):
    for file in tqdm.tqdm(rein_files, desc="it is combining reuse area..."):
        rein_file = os.path.join(rein_path, file)
        detect_bin = file.split("_reuse_area")[0]
        save_file = os.path.join(save_path, file)
        
        if os.path.exists(save_file):
            continue
        
        with open(rein_file, "r") as f:
            re_results = json.load(f)
        combine_results = {}
        
        
        
        
        
        
        for candidate_, reuse_areas_ in tqdm.tqdm(re_results.items()):
            combine_results[candidate_] = {}
            candidate_list = {}
            
            # combine_group = []
            # deal_done_list = []
            candidates_bin = candidate_
            # if candidates_bin=="bzip2" and detect_bin == "precomp":
            for now_node_pair_ in reuse_areas_:
                try:
                    if len(reuse_areas_[now_node_pair_]) > 1:
                        print("not only one area for this anchor pair!!!!")
                except:
                    print("warning")
                # if now_node_pair_ not in deal_done_list:
                    # deal_done_list.append(now_node_pair_)
                # now_obj_func = eval(now_node_pair_)[0]
                # now_cdd_func = eval(now_node_pair_)[1]
                now_reuse_item_ = reuse_areas_[now_node_pair_][0]
                now_reuse_item_["obj_fcg"]["file_name"] = detect_bin
                now_reuse_item_["cdd_fcg"]["file_name"] = candidates_bin
                now_reuse_item_["obj_fcg"]["n_num_set"] = len(set(now_reuse_item_["obj_fcg"]["feature"]))
                now_reuse_item_["cdd_fcg"]["n_num_set"] = len(set(now_reuse_item_["cdd_fcg"]["feature"]))
                # now_score, gnn_score = calculate_rarm_score(now_reuse_item_["alignment_num"], now_reuse_item_["obj_fcg"], now_reuse_item_["cdd_fcg"],now_reuse_item_["obj_fcg"]["n_num_set"], now_reuse_item_["cdd_fcg"]["n_num_set"], func_embeddings, gnn)
                
                
                # now_obj_fcg_funcs = now_reuse_item_["obj_fcg"]["feature"]
                # now_cdd_fcg_funcs = now_reuse_item_["cdd_fcg"]["feature"]
                # now_reuse_item_["score"] = now_score
            
            for now_node_pair_ in tqdm.tqdm(reuse_areas_):  
                # find_it = False
                
                now_reuse_item_ = reuse_areas_[now_node_pair_][0]
                now_obj_func = eval(now_node_pair_)[0]
                now_cdd_func = eval(now_node_pair_)[1]
                now_obj_fcg_funcs = now_reuse_item_["obj_fcg"]["feature"]
                now_cdd_fcg_funcs = now_reuse_item_["cdd_fcg"]["feature"]
                now_align_rate = now_reuse_item_["alignment_rate"]
                tar_fcg_num = now_reuse_item_["obj_fcg"]["n_num_set"]
                cdd_fcg_num = now_reuse_item_["cdd_fcg"]["n_num_set"]
                now_scale = (tar_fcg_num + cdd_fcg_num) / 2
                now_scale_diff = 0.5 * min(tar_fcg_num, cdd_fcg_num)/max(tar_fcg_num, cdd_fcg_num) + 0.5
                now_align_score = 0.3 * now_align_rate + 0.7
                max_align_rate = now_align_rate
                max_scale = now_scale
                now_score = float(now_reuse_item_["gnn_score"]) * now_scale_diff * now_align_score
                max_score = now_score
                max_node_pair_ = now_node_pair_
                max_reuse_item_ = now_reuse_item_
                for other_node_pair_ in reuse_areas_:
                    if other_node_pair_ != now_node_pair_:
                        # if other_node_pair_ not in deal_done_list:
                        other_obj_func = eval(other_node_pair_)[0]
                        other_cdd_func = eval(other_node_pair_)[1]
                        other_reuse_item_ = reuse_areas_[other_node_pair_][0]
                        other_obj_fcg_funcs = other_reuse_item_["obj_fcg"]["feature"]
                        other_cdd_fcg_funcs = other_reuse_item_["cdd_fcg"]["feature"]
                        other_align_rate = other_reuse_item_["alignment_rate"]
                        tar_fcg_num = other_reuse_item_["obj_fcg"]["n_num_set"]
                        cdd_fcg_num = other_reuse_item_["cdd_fcg"]["n_num_set"]
                        other_scale = (tar_fcg_num + cdd_fcg_num) / 2
                        #0.5 * min(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) / max(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) + 0.5
                        other_scale_diff = 0.5 * min(tar_fcg_num, cdd_fcg_num)/max(tar_fcg_num, cdd_fcg_num) + 0.5
                        other_align_score = 0.3 * other_align_rate + 0.7
                        
                        if (now_obj_func == other_obj_func or now_cdd_func == other_cdd_func):
                            other_score = float(other_reuse_item_["gnn_score"]) * other_align_score * other_scale_diff
                            # other_reuse_item_["obj_fcg"]["file_name"] = detect_bin
                            # other_reuse_item_["cdd_fcg"]["file_name"] = candidates_bin
                            # other_reuse_item_["obj_fcg"]["n_num_set"] = len(set(other_reuse_item_["obj_fcg"]["feature"]))
                            # other_reuse_item_["cdd_fcg"]["n_num_set"] = len(set(other_reuse_item_["cdd_fcg"]["feature"]))
                            # other_score, gnn_score = calculate_rarm_score(other_reuse_item_["alignment_num"], other_reuse_item_["obj_fcg"], other_reuse_item_["cdd_fcg"], max_reuse_item_["obj_fcg"]["n_num_set"], max_reuse_item_["cdd_fcg"]["n_num_set"], func_embeddings, gnn)
                            # # if other_node_pair_ not in deal_done_list:
                            # #     deal_done_list.append(other_node_pair_)
                            if other_score > max_score:
                                max_align_rate = other_align_rate
                                max_scale = other_scale
                                max_score = other_score
                                max_node_pair_ = other_node_pair_
                                max_reuse_item_ = other_reuse_item_
                if max_node_pair_ not in candidate_list:
                    candidate_list[max_node_pair_] = max_reuse_item_
                        # if other_scale > max_scale:
                        #     other_scale_score = 1+((other_scale-max_scale)/other_scale)/2
                        # else:
                        #     other_scale_score = 1+((other_scale-max_scale)/max_scale)/2
                            
                        # if other_alignment_num > max_alignment_num:
                        #     other_align_score = 1+((other_alignment_num-max_alignment_num)/other_alignment_num)*2
                        # else:
                        #     other_align_score = 1+((other_alignment_num-max_alignment_num)/max_alignment_num)*2
                        
                        # if other_alignment_num > max_alignment_num:
                        #     max_alignment_num = other_alignment_num
                        # if other_scale > max_scale:
                        #     max_scale = other_scale
            for now_node_pair_ in tqdm.tqdm(candidate_list):  
                # find_it = False
                # candidate_list = []
                now_reuse_item_ = reuse_areas_[now_node_pair_][0]
                now_obj_func = eval(now_node_pair_)[0]
                now_cdd_func = eval(now_node_pair_)[1]
                now_obj_fcg_funcs = now_reuse_item_["obj_fcg"]["feature"]
                now_cdd_fcg_funcs = now_reuse_item_["cdd_fcg"]["feature"]
                now_align_rate = now_reuse_item_["alignment_rate"]
                tar_fcg_num = now_reuse_item_["obj_fcg"]["n_num_set"]
                cdd_fcg_num = now_reuse_item_["cdd_fcg"]["n_num_set"]
                now_scale = (tar_fcg_num + cdd_fcg_num) / 2
                now_scale_diff = 0.5 * min(tar_fcg_num, cdd_fcg_num)/max(tar_fcg_num, cdd_fcg_num) + 0.5
                now_align_score = 0.3 * now_align_rate + 0.7
                max_align_rate = now_align_rate
                max_scale = now_scale
                now_score = float(now_reuse_item_["gnn_score"]) * now_scale_diff * now_align_score
                max_score = now_score
                max_node_pair_ = now_node_pair_
                max_reuse_item_ = now_reuse_item_
                for other_node_pair_ in candidate_list:
                    if other_node_pair_ != now_node_pair_:
                        # if other_node_pair_ not in deal_done_list:
                        other_obj_func = eval(other_node_pair_)[0]
                        other_cdd_func = eval(other_node_pair_)[1]
                        other_reuse_item_ = reuse_areas_[other_node_pair_][0]
                        other_obj_fcg_funcs = other_reuse_item_["obj_fcg"]["feature"]
                        other_cdd_fcg_funcs = other_reuse_item_["cdd_fcg"]["feature"]
                        other_align_rate = other_reuse_item_["alignment_rate"]
                        tar_fcg_num = other_reuse_item_["obj_fcg"]["n_num_set"]
                        cdd_fcg_num = other_reuse_item_["cdd_fcg"]["n_num_set"]
                        other_scale = (tar_fcg_num + cdd_fcg_num) / 2
                        #0.5 * min(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) / max(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) + 0.5
                        other_scale_diff = 0.5 * min(tar_fcg_num, cdd_fcg_num)/max(tar_fcg_num, cdd_fcg_num) + 0.5
                        other_align_score = 0.3 * other_align_rate + 0.7
                        
                        if (now_obj_func in other_obj_fcg_funcs[1:] and now_cdd_func in other_cdd_fcg_funcs[1:]) or (other_obj_func in now_obj_fcg_funcs[1:] and other_cdd_func in now_cdd_fcg_funcs[1:]):
                            other_score = float(other_reuse_item_["gnn_score"]) * other_align_score * other_scale_diff
                            if other_scale > max_scale:
                                max_align_rate = other_align_rate
                                max_scale = other_scale
                                max_score = other_score
                                max_node_pair_ = other_node_pair_
                                max_reuse_item_ = other_reuse_item_
                        elif (now_obj_func in other_obj_fcg_funcs[1:] or now_cdd_func in other_cdd_fcg_funcs[1:] or other_obj_func in now_obj_fcg_funcs[1:] or other_cdd_func in now_cdd_fcg_funcs[1:]) or (now_obj_func == other_obj_func or now_cdd_func == other_cdd_func):
                            other_score = float(other_reuse_item_["gnn_score"]) * other_align_score * other_scale_diff
                            # other_reuse_item_["obj_fcg"]["file_name"] = detect_bin
                            # other_reuse_item_["cdd_fcg"]["file_name"] = candidates_bin
                            # other_reuse_item_["obj_fcg"]["n_num_set"] = len(set(other_reuse_item_["obj_fcg"]["feature"]))
                            # other_reuse_item_["cdd_fcg"]["n_num_set"] = len(set(other_reuse_item_["cdd_fcg"]["feature"]))
                            # other_score, gnn_score = calculate_rarm_score(other_reuse_item_["alignment_num"], other_reuse_item_["obj_fcg"], other_reuse_item_["cdd_fcg"], max_reuse_item_["obj_fcg"]["n_num_set"], max_reuse_item_["cdd_fcg"]["n_num_set"], func_embeddings, gnn)
                            # # if other_node_pair_ not in deal_done_list:
                            # #     deal_done_list.append(other_node_pair_)
                            if other_score > max_score:
                                max_align_rate = other_align_rate
                                max_scale = other_scale
                                max_score = other_score
                                max_node_pair_ = other_node_pair_
                                max_reuse_item_ = other_reuse_item_
                                    # combine_results[candidate_][other_node_pair_] = [other_reuse_item_]
                                        # find_it = True
                    # if False == find_it:
                    #     combine_results[candidate_][now_node_pair_] = [now_reuse_item_]
                    # else:
                if max_node_pair_ not in combine_results[candidate_]:
                    combine_results[candidate_][max_node_pair_] = [max_reuse_item_]
                    
                    
                    
                
                
            
            
                # combine_results[candidate_][node_pair_] = [reuse_item_]
            
            
            # combine_results[candidate_] = {}
            # candidates_bin = candidate_
            # for node_pair_ in reuse_areas_:
            #     node_pair_area_list = []
            #     for reuse_item_ in reuse_areas_[node_pair_]:
            #         reuse_item_["obj_fcg"]["file_name"] = detect_bin
            #         reuse_item_["cdd_fcg"]["file_name"] = candidates_bin
            #         # tar_select = ["wait", "up", "down"]
            #         # cdd_select = ["wait", "up", "down"]
            #         reuse_item_ = combine_reuse_area(reuse_item_, fcgs[detect_bin], fcgs[candidates_bin], func_embeddings, gnn)
            #         node_pair_area_list.append(reuse_item_)
            #     combine_results[candidate_][node_pair_] = node_pair_area_list
        
        with open(save_file, "w") as f:
            json.dump(combine_results, f)
    
    
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
    
def adjust_reuse_areas_in9(rein_files, rein_path, fcgs, save_path, func_embeddings, gnn, sim_func_path, align_func_dict, in9_func_dict):
    
    for file in tqdm.tqdm(rein_files, desc="it is adjusting reuse area..."):
        rein_file = os.path.join(rein_path, file)
        detect_bin = file.split("_reuse_area")[0]
        save_file = os.path.join(save_path, file)
        # if detect_bin == "bzip2":
        if os.path.exists(save_file):
            continue
        
        with open(rein_file, "r") as f:
            re_results = json.load(f)
        adjust_results = {}
        for candidate_, reuse_areas_ in tqdm.tqdm(re_results.items()):
            # if detect_bin == "bzip2" and candidate_ == "lzbench":
            
            sim_func_list = json.load(open(os.path.join(sim_func_path, detect_bin+"___"+candidate_+".json"), "r"))
            
            matched_func_ingraph_list = judge_in_graph(fcgs[detect_bin], fcgs[candidate_], sim_func_list)
            
            adjust_results[candidate_] = {}
            candidates_bin = candidate_
            for node_pair_ in reuse_areas_:
                # if node_pair_ == "['bzopen_or_bzdopen', 'bzopen_or_bzdopen']":
                node_pair_area_list = []
                for reuse_item_ in reuse_areas_[node_pair_]:
                    reuse_item_["obj_fcg"]["file_name"] = detect_bin
                    reuse_item_["cdd_fcg"]["file_name"] = candidates_bin
                    reuse_item_ = adjust_reuse_area([], reuse_item_, fcgs[detect_bin], fcgs[candidates_bin], func_embeddings, gnn, matched_func_ingraph_list, align_func_dict[detect_bin+"___"+candidates_bin], in9_func_dict[detect_bin], in9_func_dict[candidates_bin])
                    node_pair_area_list.append(reuse_item_)
                adjust_results[candidate_][node_pair_] = node_pair_area_list
        
        with open(save_file, "w") as f:
            json.dump(adjust_results, f)
    # pass


def adjust_reuse_areas(rein_files, rein_path, fcgs, save_path, func_embeddings, gnn, sim_func_path, align_func_dict):
    
    for file in tqdm.tqdm(rein_files, desc="it is adjusting reuse area..."):
        rein_file = os.path.join(rein_path, file)
        detect_bin = file.split("_reuse_area")[0]
        save_file = os.path.join(save_path, file)
        # if detect_bin == "bzip2":
        if os.path.exists(save_file):
            continue
        
        with open(rein_file, "r") as f:
            re_results = json.load(f)
        adjust_results = {}
        for candidate_, reuse_areas_ in tqdm.tqdm(re_results.items()):
            # if detect_bin == "bzip2" and candidate_ == "lzbench":
            
            sim_func_list = json.load(open(os.path.join(sim_func_path, detect_bin+"___"+candidate_+".json"), "r"))
            
            matched_func_ingraph_list = judge_in_graph(fcgs[detect_bin], fcgs[candidate_], sim_func_list)
            
            adjust_results[candidate_] = {}
            candidates_bin = candidate_
            for node_pair_ in reuse_areas_:
                # if node_pair_ == "['bzopen_or_bzdopen', 'bzopen_or_bzdopen']":
                node_pair_area_list = []
                for reuse_item_ in reuse_areas_[node_pair_]:
                    reuse_item_["obj_fcg"]["file_name"] = detect_bin
                    reuse_item_["cdd_fcg"]["file_name"] = candidates_bin
                    reuse_item_ = adjust_reuse_area([], reuse_item_, fcgs[detect_bin], fcgs[candidates_bin], func_embeddings, gnn, matched_func_ingraph_list, align_func_dict[detect_bin+"___"+candidates_bin])
                    node_pair_area_list.append(reuse_item_)
                adjust_results[candidate_][node_pair_] = node_pair_area_list
        
        with open(save_file, "w") as f:
            json.dump(adjust_results, f)
    # pass



def adjust_reuse_area(bround_point_list_before, reuse_area, target_binary_fcg:nx.DiGraph, candidate_binary_fcg:nx.DiGraph, func_embeddings, gnn, matched_func_ingraph_list, align_func_list):
    # tar_fcg = reuse_area["obj_fcg"]  # 起始函数名称
    # cdd_fcg = reuse_area["cdd_fcg"]
    
    bround_point_list_list = get_bround_point(bround_point_list_before, reuse_area, target_binary_fcg, candidate_binary_fcg, align_func_list)
    # if len(bround_point_list_list) > 1:
    #     print("warning")
    for bround_point_list in bround_point_list_list:
        reuse_area, bround_point_list_after = adjust_reuse_area_iter(bround_point_list, reuse_area, target_binary_fcg, candidate_binary_fcg, func_embeddings, gnn, matched_func_ingraph_list)
        
        if bround_point_list_after == []:
            continue
        else:
            reuse_area = adjust_reuse_area(bround_point_list_after, reuse_area, target_binary_fcg, candidate_binary_fcg, func_embeddings, gnn, matched_func_ingraph_list, align_func_list)
            continue
    return reuse_area
    

def adjust_reuse_area_iter(bround_point_list, reuse_area, target_binary_fcg:nx.DiGraph, candidate_binary_fcg:nx.DiGraph, func_embeddings, gnn, matched_func_ingraph_list):
    
    # bround_point_list_temp = copy.deepcopy(bround_point_list)
    
    tar_select = ["wait", "up"]
    cdd_select = ["wait", "up"]
    max_score_dire = ("wait", "wait")
    max_score = 0
    max_gnn_score = 0
    max_new_bround_point = []
    
    for i in tar_select:
        for j in cdd_select:
            status, score, tf, cf, gnn_score, new_bround_point = calculate_score(bround_point_list, i, j, reuse_area, target_binary_fcg, candidate_binary_fcg, func_embeddings, gnn, matched_func_ingraph_list)
            # if status[0] == 0:
            #     return reuse_area, []
                # return reuse_area
            if status[1] == 0:
                tar_select.remove(i)
                i = "wait"
            if status[2] == 0:
                cdd_select.remove(j)
                j = "wait"
                
            new_scale = (len(set(tf["feature"])) + len(set(tf["feature"]))) / 2
            old_scale = (reuse_area["fcg_scale"][0] + reuse_area["fcg_scale"][1]) / 2
            
            if (abs(score - max_score) < 0.001 and new_scale > old_scale) or score - max_score >= 0.01:
                max_score_dire = (i, j)
                max_score = score
                max_gnn_score = gnn_score
                tar_fcg = tf
                cdd_fcg = cf
                max_new_bround_point = new_bround_point
    if max_score_dire == ("wait", "wait"):
        return reuse_area, []
    # if max_score_dire[0] == "up" and "down" in tar_select:
    #     tar_select.remove("down")
    # if max_score_dire[0] == "down" and "up" in tar_select:
    #     tar_select.remove("up")
    # if max_score_dire[1] == "up" and "down" in cdd_select:
    #     cdd_select.remove("down")
    # if max_score_dire[1] == "down" and "up" in cdd_select:
    #     cdd_select.remove("up")
    else:
        reuse_area["obj_fcg"] = tar_fcg
        reuse_area["cdd_fcg"] = cdd_fcg
        reuse_area["gnn_score"] = max_gnn_score
        reuse_area["final_score"] = max_score
        reuse_area["fcg_scale"] = [len(set(tar_fcg["feature"])), len(set(cdd_fcg["feature"]))]
        # for new_point in max_new_bround_point:
        #     if new_point not in bround_point_list_temp:
        #         bround_point_list_temp.append(new_point)
        # reuse_area = adjust_reuse_area_iter(reuse_area, target_binary_fcg, candidate_binary_fcg, func_embeddings, gnn)
    return reuse_area, max_new_bround_point



def calculate_score(bround_point_list, t_d, c_d, reuse_area, target_binary_fcg:nx.DiGraph, candidate_binary_fcg:nx.DiGraph, func_embeddings, gnn, matched_func_ingraph_list):
    # new_bround_point = []
    t_status = 1
    c_status = 1
    raw_tar_fcg_num = reuse_area["obj_fcg"]["n_num"]
    raw_cdd_fcg_num = reuse_area["cdd_fcg"]["n_num"]
    if t_d == "up":
        t_status, tar_fcg, tar_new_bround_point = up_get_fcg(bround_point_list[0], reuse_area["obj_fcg"], target_binary_fcg)
    elif t_d == "down":
        t_status, tar_fcg = down_get_fcg(reuse_area["obj_fcg"])
    else:
        tar_fcg = reuse_area["obj_fcg"]
        tar_new_bround_point = []
    if c_d == "up":
        c_status, cdd_fcg, cdd_new_bround_point = up_get_fcg(bround_point_list[1], reuse_area["cdd_fcg"], candidate_binary_fcg)
    elif c_d == "down":
        c_status, cdd_fcg = down_get_fcg(reuse_area["cdd_fcg"])
    else:
        cdd_fcg = reuse_area["cdd_fcg"]
        cdd_new_bround_point = []
    if t_status != 0 and c_status != 0:
        score, gnn_score = calculate_rarm_score(reuse_area["alignment_rate"], tar_fcg, cdd_fcg, raw_tar_fcg_num, raw_cdd_fcg_num, func_embeddings, gnn, matched_func_ingraph_list)
        status = 1
    else:
        status = 0
        score = -1
        gnn_score = -1
    if tar_new_bround_point == [] and cdd_new_bround_point == []:
        return [status, t_status, c_status], score, tar_fcg, cdd_fcg, gnn_score, []
    else:
        return [status, t_status, c_status], score, tar_fcg, cdd_fcg, gnn_score, [[tar_new_bround_point, cdd_new_bround_point]]



def up_get_fcg(bround_point_list, old_minifcg, binary_fcg:nx.DiGraph):
    minifcg = copy.deepcopy(old_minifcg)
    new_bround_point = []
    # minifcg = old_minifcg
    has_diff_father = 0
    for bround_point in bround_point_list:
        in_edges = binary_fcg.in_edges(minifcg["feature"][bround_point])
        # has_diff_father = 0
        if len(in_edges) != 0:
            for in_edge in in_edges:
                if in_edge[0] not in minifcg["feature"]:
                    has_diff_father = 1
                    minifcg["feature"].append(in_edge[0])
                    minifcg["n_num"] += 1
                    new_point = len(minifcg["feature"]) - 1
                    minifcg["succs"].append([bround_point])
                    new_bround_point.append(new_point)
                    # if has_diff_father == 1:
                    # nodes = [in_edge[0]]
                    # succs = [[]]
                    add_child(in_edge[0], binary_fcg, minifcg["feature"], minifcg["succs"], minifcg["n_num"], set())
                    # for node in nodes:
                    #     if node not in minifcg["feature"]:
                    #         minifcg["feature"].append(node)
                    #         minifcg["n_num"] += 1
                    #         minifcg["succs"].append([])
                    # for node in nodes:
                    #     node_index = nodes.index(node)
                    #     node_index_new = minifcg["feature"].index(node)
                    #     # if node_index_new not in new_bround_point:
                    #     #     new_bround_point.append(node_index_new)
                    #     succ = succs[node_index]
                    #     for s in succ:
                    #         new_node = nodes[s]
                    #         new_node_index = minifcg["feature"].index(new_node)
                    #         if new_node_index not in minifcg["succs"][node_index_new]:
                    #             minifcg["succs"][node_index_new].append(new_node_index)
                minifcg["n_num"] = len(minifcg["feature"])
    return has_diff_father, minifcg, new_bround_point
    # return has_diff_father, minifcg, new_bround_point


def combine_area(obj_func_embeddings_path, cdd_func_embeddings_path, target_fcgs_path, candidate_fcgs_path, save_path, reinforment_path, gnn_model_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    gnn = False#torch.load(gnn_model_path)
    # gnn.cuda()
    fcgs = {}
    # torch.multiprocessing.set_start_method('spawn', force=True)
    for fcg_p in os.listdir(target_fcgs_path):
        with open(os.path.join(target_fcgs_path, fcg_p), "rb") as f:
            tar_fcg = pickle.load(f)
        fcgs[fcg_p.split("_fcg.pkl")[0]] = tar_fcg
    for fcg_p in os.listdir(candidate_fcgs_path):
        with open(os.path.join(candidate_fcgs_path, fcg_p), "rb") as f:
            cdd_fcg = pickle.load(f)
        fcgs[fcg_p.split("_fcg.pkl")[0]] = cdd_fcg
    with open(obj_func_embeddings_path, "r") as f:
        obj_func_embeddings = json.load(f)
    with open(cdd_func_embeddings_path, "r") as f:
        cdd_func_embeddings = json.load(f)
    for func, embed in obj_func_embeddings.items():
        if func not in cdd_func_embeddings:
            cdd_func_embeddings[func] = embed
        # else:
        #     pass#print("[ERROR]candidates and target have the same func name---{0}...".format(func))

    rein_file = list(os.listdir(reinforment_path))
    rein_files = []
    for f in rein_file:
        if not os.path.exists(save_path + f):
            rein_files.append(f)
    process_num = min(30, len(rein_files))
    p_list = []
    for i in range(process_num):
        files = rein_files[int((i) / process_num * len(rein_files)): int((i + 1) / process_num * len(rein_files))]
        # combine_reuse_areas(files, reinforment_path, fcgs, save_path, cdd_func_embeddings, gnn)
        p = Process(target=combine_reuse_areas, args=(files, reinforment_path, fcgs, save_path, cdd_func_embeddings, gnn))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()
    
    

def adjust_in9(obj_func_embeddings_path, cdd_func_embeddings_path, target_fcgs_path, candidate_fcgs_path, save_path, reinforment_path, gnn_model_path, sim_func_path, align_func_path, tar_in9_func_path, cdd_in9_func_path):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    gnn = torch.load(gnn_model_path)
    gnn.cuda()
    fcgs = {}
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    align_func_dict = {}
    for area_file in os.listdir(align_func_path):
        tar_area_item  = json.load(open(os.path.join(align_func_path, area_file), "r"))
        for candidate_item in tar_area_item:
            alignment_points = list(tar_area_item[candidate_item].keys())
            align_func_dict[area_file.split("_reuse_area")[0]+"___"+candidate_item] = alignment_points
    
    in9_func_dict = {}
    tar_in9_func_list = list(json.load(open(tar_in9_func_path, "r")).keys())
    for tar_in9_func in tar_in9_func_list:
        bin_name = tar_in9_func.split("|||")[0]
        func_name = tar_in9_func.split("|||")[1]
        if bin_name not in in9_func_dict:
            in9_func_dict[bin_name] = []
        if func_name not in in9_func_dict[bin_name]:
            in9_func_dict[bin_name].append(func_name)
    cdd_in9_func_list = list(json.load(open(cdd_in9_func_path, "r")).keys())
    for cdd_in9_func in cdd_in9_func_list:
        bin_name = cdd_in9_func.split("|||")[0]
        func_name = cdd_in9_func.split("|||")[1]
        if bin_name not in in9_func_dict:
            in9_func_dict[bin_name] = []
        if func_name not in in9_func_dict[bin_name]:
            in9_func_dict[bin_name].append(func_name)
        
    for fcg_p in os.listdir(target_fcgs_path):
        with open(os.path.join(target_fcgs_path, fcg_p), "rb") as f:
            tar_fcg = pickle.load(f)
        fcgs[fcg_p.split("_fcg.pkl")[0]] = tar_fcg
    for fcg_p in os.listdir(candidate_fcgs_path):
        with open(os.path.join(candidate_fcgs_path, fcg_p), "rb") as f:
            cdd_fcg = pickle.load(f)
        fcgs[fcg_p.split("_fcg.pkl")[0]] = cdd_fcg
    with open(obj_func_embeddings_path, "r") as f:
        obj_func_embeddings = json.load(f)
    with open(cdd_func_embeddings_path, "r") as f:
        cdd_func_embeddings = json.load(f)
    for func, embed in obj_func_embeddings.items():
        if func not in cdd_func_embeddings:
            cdd_func_embeddings[func] = embed
        # else:
        #     pass#print("[ERROR]candidates and target have the same func name---{0}...".format(func))

    rein_file = list(os.listdir(reinforment_path))
    rein_files = []
    for f in rein_file:
        if not os.path.exists(save_path + f):
            rein_files.append(f)
    process_num = min(35, len(rein_files))
    p_list = []
    for i in range(process_num):
        files = rein_files[int((i) / process_num * len(rein_files)): int((i + 1) / process_num * len(rein_files))]
        # adjust_reuse_areas(files, reinforment_path, fcgs, save_path, cdd_func_embeddings, gnn, sim_func_path, align_func_dict, in9_func_path)
        p = Process(target=adjust_reuse_areas_in9, args=(files, reinforment_path, fcgs, save_path, cdd_func_embeddings, gnn, sim_func_path, align_func_dict, in9_func_dict))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()



def main(obj_func_embeddings_path, cdd_func_embeddings_path, target_fcgs_path, candidate_fcgs_path, save_path, reinforment_path, gnn_model_path, sim_func_path, align_func_path):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    gnn = torch.load(gnn_model_path)
    gnn.cuda()
    fcgs = {}
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    align_func_dict = {}
    for area_file in os.listdir(align_func_path):
        tar_area_item  = json.load(open(os.path.join(align_func_path, area_file), "r"))
        for candidate_item in tar_area_item:
            alignment_points = list(tar_area_item[candidate_item].keys())
            align_func_dict[area_file.split("_reuse_area")[0]+"___"+candidate_item] = alignment_points
    
    
    
    for fcg_p in os.listdir(target_fcgs_path):
        with open(os.path.join(target_fcgs_path, fcg_p), "rb") as f:
            tar_fcg = pickle.load(f)
        fcgs[fcg_p.split("_fcg.pkl")[0]] = tar_fcg
    for fcg_p in os.listdir(candidate_fcgs_path):
        with open(os.path.join(candidate_fcgs_path, fcg_p), "rb") as f:
            cdd_fcg = pickle.load(f)
        fcgs[fcg_p.split("_fcg.pkl")[0]] = cdd_fcg
    with open(obj_func_embeddings_path, "r") as f:
        obj_func_embeddings = json.load(f)
    with open(cdd_func_embeddings_path, "r") as f:
        cdd_func_embeddings = json.load(f)
    for func, embed in obj_func_embeddings.items():
        if func not in cdd_func_embeddings:
            cdd_func_embeddings[func] = embed
        # else:
        #     pass#print("[ERROR]candidates and target have the same func name---{0}...".format(func))

    rein_file = list(os.listdir(reinforment_path))
    rein_files = []
    for f in rein_file:
        if not os.path.exists(save_path + f):
            rein_files.append(f)
    process_num = min(10, len(rein_files))
    p_list = []
    for i in range(process_num):
        files = rein_files[int((i) / process_num * len(rein_files)): int((i + 1) / process_num * len(rein_files))]
        # adjust_reuse_areas(files, reinforment_path, fcgs, save_path, cdd_func_embeddings, gnn, sim_func_path, align_func_dict)
        p = Process(target=adjust_reuse_areas, args=(files, reinforment_path, fcgs, save_path, cdd_func_embeddings, gnn, sim_func_path, align_func_dict))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()




if __name__ == '__main__':
#     obj_func_embeddings_path = "/data/wangyongpan/paper/reuse_detection/datasets/paper_datasets/embeddings/isrd_target_embeddings_torch_best.json"
#     cdd_func_embeddings_path = "/data/wangyongpan/paper/reuse_detection/datasets/paper_datasets/embeddings/isrd_candidates_embeddings_torch_best.json"
#     target_fcgs_path = "/data/wangyongpan/paper/reuse_detection/datasets/paper_datasets/isrd_target_fcg/"
#     candidate_fcgs_path = "/data/wangyongpan/paper/reuse_detection/datasets/paper_datasets/isrd_target_fcg/"
#     save_path = "/data/wangyongpan/adjust/reuse_area_0.8_adjust/"
#     reinforment_path = "/data/wangyongpan/adjust/reuse_area_0.8/"
#     gnn_model_path = "/data/wangyongpan/paper/reuse_detection/code/libdb/saved_model/fcg_analog_gnn-best-0.01.pt"
    main(os.path.join(DATA_PATH, "4_embedding/target_func_embedding"), 
         os.path.join(DATA_PATH, "4_embedding/candidate_func_embedding"), 
         os.path.join(DATA_PATH, "2_target/fcg"), 
         os.path.join(DATA_PATH, "3_candidate/fcg"), 
         os.path.join(DATA_PATH, "9_area_adjustment_result/reuse_area_7_adjust"), 
         os.path.join(DATA_PATH, "7_gnn_result/after_gnn_result/reuse_area_7"), 
         os.path.join(WORK_PATH, "code/reuse_area_Exploration/Embeded-GNN/fcg_gnn-best-0.01.pt"))
    
    

