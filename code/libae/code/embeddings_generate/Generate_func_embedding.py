#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from tqdm import tqdm
import os
import numpy as np
import time, pickle, sys


def transform_data(data):
    """
    :param data: {'n_num': 10, 'features': [[1, 2, ...]], 'succs': [[], ..]}
    :return:
    """
    feature_dim = len(data['features'][0])
    node_size = data['n_num']
    X = np.zeros((1, node_size, feature_dim))
    mask = np.zeros((1, node_size, node_size))
    for start_node, end_nodes in enumerate(data['succs']):
        X[0, start_node, :] = np.array(data['features'][start_node])
        for node in end_nodes:
            mask[0, start_node, node] = 1
    return X, mask

def embed_by_feat_torch(feat, gnn):
    X, mask = transform_data(feat)
    X = torch.from_numpy(X).to(torch.float32).cuda()
    mask = torch.from_numpy(mask).to(torch.float32).cuda()
    return gnn.forward_once(X, mask).cpu().detach().numpy()

def load_model(model_path="saved_model/gnn-best.pt"):
    return torch.load(model_path)



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
    return list(set(children_list))

def generate_afcg(save_path, fcg_path, func_embedding_path, model_path):
    
    if not os.path.exists(save_path):
    # os.rmdir(savePath)
        os.makedirs(save_path)
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # gnn = torch.load(model_path)
    
    
    
    with open(func_embedding_path, "r") as f:
        func_embeddings = json.load(f)
    
    all_acfg = {}
        
        
        
    for bin_func in tqdm(func_embeddings):
        bin_name = bin_func.split("|||")[0]
        func_name = bin_func.split("|||")[1]
        if not os.path.exists(os.path.join(save_path, bin_name+"_afcg.json")):
            with open(os.path.join(fcg_path, bin_name+"_fcg.pkl"), "rb") as f:
                fcg = pickle.load(f)
            
            walked_map_set = set()
            child_node_list = get_children_list(func_name, fcg, walked_map_set)
            
            if child_node_list != []:
                if bin_name not in all_acfg:
                    all_acfg[bin_name] = {}
                all_acfg[bin_name][func_name] = child_node_list
        
    for bin_name in all_acfg:
        json.dump(all_acfg[bin_name], open(os.path.join(save_path, bin_name+"_afcg.json"), "w"))



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


def transform_fcg(data):
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

def embed_fcg(feat, gnn):
    X, mask = transform_fcg(feat)
    X = torch.from_numpy(X).to(torch.float32).cuda()
    mask = torch.from_numpy(mask).to(torch.float32).cuda()
    # return gnn.forward_once(X, mask).cpu().detach().numpy()
    return gnn.forward_once(X, mask)

def generate_subgraph(save_path, fcg_path, func_embedding_path, model_path):
    if not os.path.exists(save_path):
    # os.rmdir(savePath)
        os.makedirs(save_path)
    
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gnn = torch.load(model_path)
        
    # with open(fcg_path, "rb") as f:
    #     fcg = pickle.load(f)
    with open(func_embedding_path, "r") as f:
        func_embeddings = json.load(f)
    
    all_subgraph = {}

    for bin_func in tqdm(func_embeddings):
        # if bin_func == "xz|||lzma_alone_encoder":
        #     print("warning")
        bin_name = bin_func.split("|||")[0]
        func_name = bin_func.split("|||")[1]
        if not os.path.exists(os.path.join(save_path, bin_name+"_subgraph.json")):
            with open(os.path.join(fcg_path, bin_name+"_fcg.pkl"), "rb") as f:
                fcg = pickle.load(f)

            subgraph = {}
            subgraph["feature"] = []
            subgraph["succs"] = []
            walked_map = set()
            subgraph["feature"].append(func_name)
            subgraph["succs"].append([])
            add_child(func_name, fcg, subgraph["feature"], subgraph["succs"], walked_map)
            subgraph["n_num"] = len(subgraph["feature"])
            
            
            feature = []
            for func in subgraph["feature"]:
                bin_func_name = bin_name + "|||" + func
                if bin_func_name in func_embeddings:
                    embed = func_embeddings[bin_func_name][0]
                    # true_num += 1
                else:
                    # false_num += 1
                    embed = list(np.array([0.001 for i in range(64)]))
                feature.append(embed)
            subgraph2emb = subgraph.copy()
            subgraph2emb["embeddings"] = feature
            subgraph_embedding = embed_fcg(subgraph2emb, gnn)
            
            subgraph["embedding"] = subgraph_embedding.tolist()
            
            if bin_name not in all_subgraph:
                all_subgraph[bin_name] = {}
            all_subgraph[bin_name][func_name] = subgraph
            
    for bin_name in all_subgraph:
        json.dump(all_subgraph[bin_name], open(os.path.join(save_path, bin_name+"_subgraph.json"), "w"))

        
        
        

def subfcg_embedding(TIME_PATH, test_gemini_feat_paths, savePath, model_path):
    # test_gemini_feat_paths = '/data/wangyongpan/libdb_dataset_features'
    gnn = load_model(model_path)
    fname_embeddings = {}
    fname_embeddings_in = {}
    
    (path, filename) = os.path.split(savePath)
    if not os.path.exists(path):
        if not os.path.exists(path):
            # os.rmdir(savePath)
            os.makedirs(path)
        if not os.path.exists(TIME_PATH):
            # os.rmdir(savePath)
            os.makedirs(TIME_PATH)
    if not os.path.exists(savePath.replace("_in9_bl5", "_in9")):
        for test_gemini_feat_path in tqdm(os.listdir(test_gemini_feat_paths), desc="it is generating func embeddings..."):
            start = time.time()
            project_name = test_gemini_feat_path.split(".json")[0]
            with open(os.path.join(test_gemini_feat_paths, test_gemini_feat_path), 'r') as f:
                for line in f:
                    feat = json.loads(line)
                    inst_num = 0
                    for block in feat['features']:
                        inst_num += block[4]
                    if inst_num > 9 and feat['n_num'] >= 5:
                        embedding = embed_by_feat_torch(feat, gnn)
                        fname = project_name + "|||" + feat['fname']
                        fname_embeddings_in[fname] = embedding.tolist()
                    if inst_num > 9:
                        embedding = embed_by_feat_torch(feat, gnn)
                        fname = project_name + "|||" + feat['fname']
                        fname_embeddings[fname] = embedding.tolist()
            end = time.time()
            timecost = end - start
            feature_timecost = dict()
            feature_timecost[os.path.basename(test_gemini_feat_path)] = timecost
            json.dump(feature_timecost, open(os.path.join(TIME_PATH, os.path.basename(test_gemini_feat_path)+"_timecost.json"), "w"))
        # with open("/data/wangyongpan/libdataset_in10_nn5_embeddings_torch_best.json", "w") as f:
        with open(savePath, "w") as f:
            json.dump(fname_embeddings_in, f)
        with open(savePath.replace("_in9_bl5", "_in9"), "w") as f:
            json.dump(fname_embeddings, f)
        return fname_embeddings_in
    pass

if __name__ == '__main__':
    subfcg_embedding()

