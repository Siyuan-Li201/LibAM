import json
import os
import pickle

from app.utils import *


def taint_old(object_graph, matched_func_list):
    matched_func_ingraph_list = judge_in_obj_graph(object_graph, matched_func_list)
    tainted_graph = get_taint_graph(object_graph, matched_func_ingraph_list)
                
    return tainted_graph          


    
        
def anchor_align_v1(object_graph, matched_func_list, candidate_graph):
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    obj_walked_node_set = set()
    cdd_walked_node_set  = set()
    
    tainted_object_graph = object_graph
    tainted_candidate_graph = candidate_graph
    
    taint_flag = False
    ite_num = 0
    
    obj_tainted_func_set = set()
    cdd_tainted_func_set = set()
    
    for node_pair in matched_func_ingraph_list:
        if node_pair[0] not in obj_walked_node_set and node_pair[1] not in cdd_walked_node_set:
            obj_find_func_set = set()
            cdd_find_func_set = set()
            print("---------------------------------------")
            print("new explore:"+str(node_pair))
            (up_edge_num, up_matched_node_pair_list, obj_up_walked_node_set, cdd_up_walked_node_set) = get_up_linked_node(ite_num, obj_walked_node_set, cdd_walked_node_set, [node_pair[0]], [node_pair[1]], matched_func_ingraph_list, object_graph, candidate_graph)
            (down_edge_num, down_matched_node_pair_list, obj_down_walked_node_set, cdd_down_walked_node_set) = get_down_linked_node(ite_num ,obj_walked_node_set, cdd_walked_node_set, [node_pair[0]], [node_pair[1]], matched_func_ingraph_list, object_graph, candidate_graph)
            obj_walked_node_set = obj_walked_node_set.union(obj_up_walked_node_set)
            obj_walked_node_set = obj_walked_node_set.union(obj_down_walked_node_set)
            cdd_walked_node_set = cdd_walked_node_set.union(cdd_up_walked_node_set)
            cdd_walked_node_set = cdd_walked_node_set.union(cdd_down_walked_node_set)
            for node_pair in up_matched_node_pair_list:
                obj_find_func_set.add(node_pair[0])
                cdd_find_func_set.add(node_pair[1])
            for node_pair in down_matched_node_pair_list:
                obj_find_func_set.add(node_pair[0])
                cdd_find_func_set.add(node_pair[1])
        if up_edge_num + down_edge_num >= 3 and len(cdd_find_func_set) >= 3 and len(obj_find_func_set) >= 3:
            for node_pair in up_matched_node_pair_list:
                obj_tainted_func_set.add(node_pair[0])
                cdd_tainted_func_set.add(node_pair[1])
            for node_pair in down_matched_node_pair_list:
                obj_tainted_func_set.add(node_pair[0])
                cdd_tainted_func_set.add(node_pair[1])
            # up_matched_node_pair_list.extend(down_matched_node_pair_list)
            # tainted_object_graph = get_taint_graph(tainted_object_graph, up_matched_node_pair_list, 0)
            # tainted_candidate_graph = get_taint_graph(tainted_candidate_graph, up_matched_node_pair_list, 1)
            taint_flag = True
            
                
    return (tainted_object_graph, tainted_candidate_graph, taint_flag, obj_tainted_func_set, cdd_tainted_func_set)        

                

def anchor_align_v2(object_graph, matched_func_list, candidate_graph):
    
    tainted_graph_list = []
    
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    sub_graph_list = get_subgraph_func_list_v2(matched_func_ingraph_list, object_graph, candidate_graph)
    
    all_obj_taint_node_list = []
    all_cdd_taint_node_list = []
    for sub_graph_func_list in sub_graph_list:
        # obj_subgraph_edge_list = get_subgraph_edge(sub_graph_func_list[0], object_graph)
        # cdd_subgraph_edge_list = get_subgraph_edge(sub_graph_func_list[1], candidate_graph)
    
        # common_edge_num = get_common_edge(sub_graph_func_list, obj_subgraph_edge_list, cdd_subgraph_edge_list)
        
        if len(sub_graph_func_list) >= 3:
            object_pydot_graph = nx.nx_pydot.to_pydot(object_graph)
            candidate_pydot_graph = nx.nx_pydot.to_pydot(candidate_graph)
            obj_taint_node_list = []
            cdd_taint_node_list = []
            for node_pair in sub_graph_func_list:
                obj_taint_node_list.append(node_pair[0])
                cdd_taint_node_list.append(node_pair[1])
            all_obj_taint_node_list.extend(obj_taint_node_list)
            all_cdd_taint_node_list.extend(cdd_taint_node_list)
            object_tainted_pydot_graph = get_taint_graph(object_pydot_graph, obj_taint_node_list)
            candidate_tainted_pydot_graph = get_taint_graph(candidate_pydot_graph, cdd_taint_node_list)
            
            tainted_graph_list.append([object_tainted_pydot_graph, candidate_tainted_pydot_graph, len(sub_graph_func_list)])
    if all_obj_taint_node_list != [] and all_cdd_taint_node_list != []:
        object_tainted_pydot_graph = get_taint_graph(object_pydot_graph, all_obj_taint_node_list)
        candidate_tainted_pydot_graph = get_taint_graph(candidate_pydot_graph, all_cdd_taint_node_list)
        
        tainted_graph_list.append([object_tainted_pydot_graph, candidate_tainted_pydot_graph, 0])
            
    return tainted_graph_list    

#20221108修复cdd fcg bug
def reuse_area_exploration(object_graph, matched_func_list, candidate_graph):
    reuse_flag = False
    tainted_graph_list = []
    
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    # print("-- area explore......")
    sub_graph_list, alignment_max = get_subgraph_func_list_v5(matched_func_ingraph_list, object_graph, candidate_graph)
    # print("-- area explore done!")
    
    if alignment_max:
        reuse_flag = True
        node_pair_feature = {}
        (common, afcg_rate, matched_func) = libdb_fcg_filter(object_graph, matched_func_list, candidate_graph)
        if afcg_rate > 0.1:
            max_alignment_num = -common
            
            
def reuse_area_detection_core(object_name, candidate_name, object_graph, matched_func_list, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num):
    # reuse_flag = False
    
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    
    return reuse_area_detection_utils(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num)
    
    
def reuse_area_detection_core_annoy(object_name, candidate_name, object_graph, matched_func_list, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict):
    # reuse_flag = False
    
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    
    return reuse_area_detection_utils_annoy(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict)
    
def tpl_detection_fast_core_annoy_without_align(object_name, candidate_name, object_graph, matched_func_list, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict):
    # reuse_flag = False
    
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    
    return tpl_detection_fast_utils_annoy_without_align(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict)
            
    


def tpl_detection_fast_core_annoy_without_gnn(object_name, candidate_name, object_graph, matched_func_list, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict):
    # reuse_flag = False
    
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    
    return tpl_detection_fast_utils_annoy_without_gnn(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict)
            

def tpl_detection_fast_core_annoy_1_5(object_name, candidate_name, object_graph, matched_func_list, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, alignment_tred):
    # reuse_flag = False
    
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    
    return tpl_detection_fast_utils_annoy_1_5(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, alignment_tred)
    

    
def tpl_detection_fast_core_annoy(object_name, candidate_name, object_graph, matched_func_list, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict):
    # reuse_flag = False
    
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    
    return tpl_detection_fast_utils_annoy_v2(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict)
            
def tpl_detection_fast_core(object_name, candidate_name, object_graph, matched_func_list, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num):
    # reuse_flag = False
    
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    
    return tpl_detection_fast_utils(object_name, candidate_name, matched_func_ingraph_list, object_graph, candidate_graph, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num)
    
    # anchor_alignment_dict = {}
    # node_pair_feature = {}
    # max_alignment_num = 0
    # for sub_graph_func_list in sub_graph_list:
    #     # obj_subgraph_edge_list = get_subgraph_edge(sub_graph_func_list[0], object_graph)
    #     # cdd_subgraph_edge_list = get_subgraph_edge(sub_graph_func_list[1], candidate_graph)
    
    #     # common_edge_num = get_common_edge(sub_graph_func_list, obj_subgraph_edge_list, cdd_subgraph_edge_list)
        
    #     if len(sub_graph_func_list) >= 2:
    #         # for node_pair in sub_graph_func_list:
    #         if str(sub_graph_func_list[0]) not in anchor_alignment_dict:
    #             anchor_alignment_dict[str(sub_graph_func_list[0])] = len(sub_graph_func_list)# - sub_graph_func_list.index(node_pair)
                
    
    # for node_pair in matched_func_ingraph_list:
        
    #     if str(node_pair) in anchor_alignment_dict:
    #         obj_fcg = get_subgraph(node_pair[0], object_graph)
    #         cdd_fcg = get_subgraph(node_pair[1], candidate_graph)
            
    #         obj_num = len(set(obj_fcg["feature"]))
    #         cdd_num = len(set(cdd_fcg["feature"]))
            
    #         obj_com_num = obj_sim_num = 0
    #         for obj_func in set(obj_fcg["feature"]):
    #             if obj_func in obj_com_funcs:
    #                 obj_com_num += 1
    #                 if obj_func in obj_sim_funcs_dict and list(set(obj_sim_funcs_dict[obj_func]).intersection(set(cdd_fcg["feature"]))) != []:
    #                     obj_sim_num += 1
    #         cdd_com_num = cdd_sim_num = 0
    #         for cdd_func in set(cdd_fcg["feature"]):
    #             if cdd_func in cdd_com_funcs:
    #                 cdd_com_num += 1
    #                 if cdd_func in cdd_sim_funcs_dict and list(set(cdd_sim_funcs_dict[cdd_func]).intersection(set(obj_fcg["feature"]))) != []:
    #                     cdd_sim_num += 1
            
    #         # com_funcs_scale = (len(obj_com_funcs) + len(cdd_com_funcs)) / 2
    #         # sim_funcs_scale = (len(obj_sim_funcs) + len(cdd_sim_funcs)) / 2
    #         if obj_com_num <= cdd_com_num:
    #             align_rate = obj_sim_num / obj_com_num
    #         else:
    #             align_rate = cdd_sim_num / cdd_com_num
                
    #         # if ((obj_num - cdd_num) > cdd_num) or ((cdd_num - obj_num) > obj_num) or abs(obj_num - cdd_num) > 100 or abs(sim_num-com_num) > 50 or align_rate < 0.5:#  or align_rate < 0.8 or obj_align_rate < 0.05 or cdd_align_rate <  0.05 :
    #         #     alignment_temp = 0
    #         # elif obj_num < anchor_alignment_dict[str(node_pair)] or cdd_num < anchor_alignment_dict[str(node_pair)]:
    #         #     alignment_temp = min( obj_num, cdd_num)
    #         # else: 
    #         alignment_temp = anchor_alignment_dict[str(node_pair)]
    #         if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2*min(obj_num, cdd_num) and max(obj_num, cdd_num) > 100) or (abs(obj_num - cdd_num) > 200):
    #             alignment_temp = 0
    #         if (obj_fcg["n_num"] >= 3 and cdd_fcg["n_num"] >= 3 and alignment_temp >= 3) or (obj_num <= 10 and cdd_num <= 10 and alignment_temp >= 2):
    #             reuse_flag = True
    #             node_pair_feature[str(node_pair)] = {}
    #             node_pair_feature[str(node_pair)]["alignment_num"] = alignment_temp
    #             node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
    #             node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
    #             node_pair_feature[str(node_pair)]["alignment_rate"] = align_rate
    #             node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_num, cdd_num)
                
                
                
                
    #             if alignment_temp > max_alignment_num:
    #                 max_alignment_num = alignment_temp
                

            
    #     # else:
    #     #     node_pair_feature[str(node_pair)] = {}
    #     #     node_pair_feature[str(node_pair)]["alignment_num"] = 0
    #     #     obj_fcg = get_subgraph(node_pair[0], object_graph)
    #     #     cdd_fcg = get_subgraph(node_pair[1], candidate_graph)
    #     #     node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
    #     #     node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
            
    #     #     node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_fcg["n_num"], cdd_fcg["n_num"])
            
    #     #     reuse_flag = False
            
    # return node_pair_feature, reuse_flag, max_alignment_num, obj_sim_funcs_dict, cdd_sim_funcs_dict
            
def anchor_alignment_ransac(object_graph, matched_func_list, candidate_graph, obj_com_funcs, cdd_com_funcs):
    reuse_flag = False
    
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    
    sub_graph_list, obj_sim_funcs_dict, cdd_sim_funcs_dict = anchor_alignment_ransac_v1_0(matched_func_ingraph_list, object_graph, candidate_graph)
    
    anchor_alignment_dict = {}
    node_pair_feature = {}
    max_alignment_num = 0
    for sub_graph_func_list in sub_graph_list:
        # obj_subgraph_edge_list = get_subgraph_edge(sub_graph_func_list[0], object_graph)
        # cdd_subgraph_edge_list = get_subgraph_edge(sub_graph_func_list[1], candidate_graph)
    
        # common_edge_num = get_common_edge(sub_graph_func_list, obj_subgraph_edge_list, cdd_subgraph_edge_list)
        
        if len(sub_graph_func_list) >= 2:
            # for node_pair in sub_graph_func_list:
            if str(sub_graph_func_list[0]) not in anchor_alignment_dict:
                anchor_alignment_dict[str(sub_graph_func_list[0])] = len(sub_graph_func_list)# - sub_graph_func_list.index(node_pair)
                
    
    for node_pair in matched_func_ingraph_list:
        
        if str(node_pair) in anchor_alignment_dict:
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
            if obj_com_num <= cdd_com_num:
                align_rate = obj_sim_num / obj_com_num
            else:
                align_rate = cdd_sim_num / cdd_com_num
                
            # if ((obj_num - cdd_num) > cdd_num) or ((cdd_num - obj_num) > obj_num) or abs(obj_num - cdd_num) > 100 or abs(sim_num-com_num) > 50 or align_rate < 0.5:#  or align_rate < 0.8 or obj_align_rate < 0.05 or cdd_align_rate <  0.05 :
            #     alignment_temp = 0
            # elif obj_num < anchor_alignment_dict[str(node_pair)] or cdd_num < anchor_alignment_dict[str(node_pair)]:
            #     alignment_temp = min( obj_num, cdd_num)
            # else: 
            alignment_temp = anchor_alignment_dict[str(node_pair)]
            if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2*min(obj_num, cdd_num) and max(obj_num, cdd_num) > 100) or (abs(obj_num - cdd_num) > 200):
                alignment_temp = 0
            if (obj_fcg["n_num"] >= 3 and cdd_fcg["n_num"] >= 3 and alignment_temp >= 3) or (obj_num <= 10 and cdd_num <= 10 and alignment_temp >= 2):
                reuse_flag = True
                node_pair_feature[str(node_pair)] = {}
                node_pair_feature[str(node_pair)]["alignment_num"] = alignment_temp
                node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
                node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
                node_pair_feature[str(node_pair)]["alignment_rate"] = align_rate
                node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_num, cdd_num)
                
                
                
                
                if alignment_temp > max_alignment_num:
                    max_alignment_num = alignment_temp
                

            
        # else:
        #     node_pair_feature[str(node_pair)] = {}
        #     node_pair_feature[str(node_pair)]["alignment_num"] = 0
        #     obj_fcg = get_subgraph(node_pair[0], object_graph)
        #     cdd_fcg = get_subgraph(node_pair[1], candidate_graph)
        #     node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
        #     node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
            
        #     node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_fcg["n_num"], cdd_fcg["n_num"])
            
        #     reuse_flag = False
            
    return node_pair_feature, reuse_flag, max_alignment_num, obj_sim_funcs_dict, cdd_sim_funcs_dict
            
            
#20221108修复cdd fcg bug
def anchor_alignment_area(object_graph, matched_func_list, candidate_graph):
    reuse_flag = False
    tainted_graph_list = []
    
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    # print("-- area explore......")
    sub_graph_list, alignment_max = anchor_alignment_v2_0(matched_func_ingraph_list, object_graph, candidate_graph)
    # print("-- area explore done!")
    
    if alignment_max:
        reuse_flag = True
        node_pair_feature = {}
        (common, afcg_rate, matched_func) = libdb_fcg_filter(object_graph, matched_func_list, candidate_graph)
        if afcg_rate > 0.1:
            max_alignment_num = -common         
            
            
            
            
        else:
            max_alignment_num = 0
    else:
        
        # sub_graph_list = filter_alignment_result(sub_graph_list, object_graph, candidate_graph)
        
        anchor_alignment_dict = {}
        node_pair_feature = {}
        max_alignment_num = 0
        for sub_graph_func_list in sub_graph_list:
            # obj_subgraph_edge_list = get_subgraph_edge(sub_graph_func_list[0], object_graph)
            # cdd_subgraph_edge_list = get_subgraph_edge(sub_graph_func_list[1], candidate_graph)
        
            # common_edge_num = get_common_edge(sub_graph_func_list, obj_subgraph_edge_list, cdd_subgraph_edge_list)
            
            if len(sub_graph_func_list) >= 2:
                for node_pair in sub_graph_func_list:
                    anchor_alignment_dict[str(node_pair)] = len(sub_graph_func_list) - sub_graph_func_list.index(node_pair)
                
        
        for node_pair in matched_func_ingraph_list:
            
            if str(node_pair) in anchor_alignment_dict:
                node_pair_feature[str(node_pair)] = {}
                obj_fcg = get_subgraph(node_pair[0], object_graph)
            
                cdd_fcg = get_subgraph(node_pair[1], candidate_graph)
                node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
                node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
                node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_fcg["n_num"], cdd_fcg["n_num"])
                
                obj_align_rate = anchor_alignment_dict[str(node_pair)] / obj_fcg["n_num"]
                
                cdd_align_rate = anchor_alignment_dict[str(node_pair)] / cdd_fcg["n_num"]
                
                if ((obj_fcg["n_num"] - cdd_fcg["n_num"]) > 2*cdd_fcg["n_num"]) or ((cdd_fcg["n_num"] - obj_fcg["n_num"]) > 2*obj_fcg["n_num"]) or obj_align_rate < 0.05 or cdd_align_rate <  0.05 :
                    node_pair_feature[str(node_pair)]["alignment_num"] = 0
                elif obj_fcg["n_num"] < anchor_alignment_dict[str(node_pair)] or cdd_fcg["n_num"] < anchor_alignment_dict[str(node_pair)]:
                    node_pair_feature[str(node_pair)]["alignment_num"] = min( obj_fcg["n_num"], cdd_fcg["n_num"])
                else: 
                    node_pair_feature[str(node_pair)]["alignment_num"] = anchor_alignment_dict[str(node_pair)]
                    if anchor_alignment_dict[str(node_pair)] > max_alignment_num:
                        max_alignment_num = anchor_alignment_dict[str(node_pair)] 
                

                if obj_fcg["n_num"] >= 5 and cdd_fcg["n_num"] >= 5:
                    reuse_flag = True
            else:
                node_pair_feature[str(node_pair)] = {}
                node_pair_feature[str(node_pair)]["alignment_num"] = 0
                obj_fcg = get_subgraph(node_pair[0], object_graph)
                cdd_fcg = get_subgraph(node_pair[1], candidate_graph)
                node_pair_feature[str(node_pair)]["obj_fcg"] = obj_fcg
                node_pair_feature[str(node_pair)]["cdd_fcg"] = cdd_fcg
                
                node_pair_feature[str(node_pair)]["fcg_scale"] = (obj_fcg["n_num"], cdd_fcg["n_num"])
                
                reuse_flag = False
            
    return node_pair_feature, reuse_flag, max_alignment_num



#20221028：发现bug：以list保存子图，可能替换时出现不连通（暂未解决，可以替换后判断是否联通，不连通的话只保留前面联通部分，散点往里插，插不进去丢掉）（也可以用networkx的图结构来存子图）
def anchor_align_v3(object_graph, matched_func_list, candidate_graph):
    
    tainted_graph_list = []
    
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    print("-- area explore......")
    sub_graph_list, alignment_max = get_subgraph_func_list_v4(matched_func_ingraph_list, object_graph, candidate_graph)
    print("-- area explore done!")
    
    if alignment_max:
        return tainted_graph_list 
    else:
        print("-- area explore done!")
        all_obj_taint_node_list = []
        all_cdd_taint_node_list = []
        print("-- taint graph......")
        for sub_graph_func_list in tqdm(sub_graph_list):
            # obj_subgraph_edge_list = get_subgraph_edge(sub_graph_func_list[0], object_graph)
            # cdd_subgraph_edge_list = get_subgraph_edge(sub_graph_func_list[1], candidate_graph)
        
            # common_edge_num = get_common_edge(sub_graph_func_list, obj_subgraph_edge_list, cdd_subgraph_edge_list)
            
            if len(sub_graph_func_list) >= 3:
                object_graph_tmp = object_graph.copy(as_view=True)
                candidate_graph_tmp = candidate_graph.copy(as_view=True)
                object_pydot_graph = nx.nx_pydot.to_pydot(object_graph_tmp)
                object_pydot_graph.set_rankdir('LR')
                candidate_pydot_graph = nx.nx_pydot.to_pydot(candidate_graph_tmp)
                candidate_pydot_graph.set_rankdir('LR')
                obj_taint_node_list = []
                cdd_taint_node_list = []
                for node_pair in sub_graph_func_list:
                    obj_taint_node_list.append(node_pair[0])
                    cdd_taint_node_list.append(node_pair[1])
                all_obj_taint_node_list.extend(obj_taint_node_list)
                all_cdd_taint_node_list.extend(cdd_taint_node_list)
                object_tainted_pydot_graph = get_taint_graph(object_pydot_graph, obj_taint_node_list)
                candidate_tainted_pydot_graph = get_taint_graph(candidate_pydot_graph, cdd_taint_node_list)
                
                tainted_graph_list.append([object_tainted_pydot_graph, candidate_tainted_pydot_graph, len(sub_graph_func_list)])
        if all_obj_taint_node_list != [] and all_cdd_taint_node_list != []:
            object_graph_tmp = object_graph.copy(as_view=True)
            candidate_graph_tmp = candidate_graph.copy(as_view=True)
            object_pydot_graph = nx.nx_pydot.to_pydot(object_graph_tmp)
            object_pydot_graph.set_rankdir('LR')
            candidate_pydot_graph = nx.nx_pydot.to_pydot(candidate_graph_tmp)
            candidate_pydot_graph.set_rankdir('LR')
            object_tainted_pydot_graph = get_taint_graph(object_pydot_graph, all_obj_taint_node_list)
            candidate_tainted_pydot_graph = get_taint_graph(candidate_pydot_graph, all_cdd_taint_node_list)
            
            tainted_graph_list.append([object_tainted_pydot_graph, candidate_tainted_pydot_graph, 0])
        print("-- taint graph done!")      
        return tainted_graph_list    


def libdb_fcg_filter(object_graph, matched_func_list, candidate_graph):
    
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    
    obj_sim_funcs = []
    cdd_sim_funcs = []
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs:
            obj_sim_funcs.append(func_pair[0])
        if func_pair[1] not in cdd_sim_funcs:
            cdd_sim_funcs.append(func_pair[1])
    
    obj_afcg = get_afcg(obj_sim_funcs, object_graph)
    cdd_afcg = get_afcg(cdd_sim_funcs, candidate_graph)
    
    common, afcg_rate, matched_func = afcg_cost(obj_afcg, cdd_afcg, matched_func_ingraph_list)

    # print("-- taint graph done!")      
    return (common, afcg_rate, matched_func)    

