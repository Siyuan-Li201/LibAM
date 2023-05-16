# -*- encoding: utf-8 -*-
'''
@File    :   get_subfcg.py
@Time    :   2022/11/25 13:35:44
@Author  :   WangYongpan 
'''
from multiprocessing import Process
import networkx as nx
import os, json
import tqdm
import numpy as np
np.random.seed(0)

drop_rate = 0.05
replace_rate = 0.05
PROCESS_NUM = 30
#fname_embeddings = subfcg_embedding()
# with open("embeddings/hxl_embeddings_tf_best.json", "r") as f:
#     fname_embeddings = json.load(f)

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

def analog_add_child(node, g, feature_list, edge_list, walked_map):
    if node not in walked_map:
        walked_map.add(node)
        out_edges = list(g.out_edges(node))
        out_edges = [list(oe) for oe in out_edges]
        edges = []

        # 随机替换和删除
        g_nodes = list(g.nodes())
        child_select_prob = list(np.random.rand(len(out_edges)))
        out_es = [e for oe in out_edges for e in oe]
        for out_edge in out_edges:
            out_index = out_edges.index(out_edge)
            if 0 <= child_select_prob[out_index] < drop_rate:
                out_edges.remove(out_edge)
                continue
            if drop_rate <= child_select_prob[out_index] < (drop_rate + replace_rate):
                flag = 0
                for i in range(10):
                    random_index = np.random.randint(0, len(g_nodes))
                    if random_index not in out_es:
                        flag = 1
                        break
                if flag == 1:
                    out_edges[out_index][1] = g_nodes[random_index]

        for out_edge in out_edges:
            # if out_edge[1][0] != ".":
            feature_list.append(out_edge[1])
            edge_list.append([])
            edges.append(feature_list.index(out_edge[1]))
        edge_list[feature_list.index(node)] = edges
        for out_edge in out_edges:
            add_child(out_edge[1], g, feature_list, edge_list, walked_map)

def get_sub_g(fcg_path, files, feature_path, mode=None):
    tq = tqdm.tqdm(files)
    for fcg_item in tq:
        if ".idb" not in fcg_item and os.path.getsize(os.path.join(fcg_path, fcg_item)) < 600*1024:
            tq.set_description("[" + str(fcg_item) + "] is processing...")
            g = nx.read_gpickle(os.path.join(fcg_path, fcg_item))
            nodes_list = list(g.nodes())
            write_list = []
            save_path = os.path.join(feature_path, fcg_item[:-4]+"_feature.json")
            if os.path.exists(save_path):
                continue
            for node in nodes_list:
                if node[0] == ".":
                    continue
                gemini_feature_dict = {}
                gemini_feature_dict["feature"] = []
                gemini_feature_dict["succs"] = []
                walked_map = set()
                gemini_feature_dict["feature"].append(node)
                gemini_feature_dict["succs"].append([])
                if mode == "analog":
                    analog_add_child(node, g, gemini_feature_dict["feature"], gemini_feature_dict["succs"], walked_map)
                else:
                    add_child(node, g, gemini_feature_dict["feature"], gemini_feature_dict["succs"], walked_map)
                gemini_feature_dict["n_num"] = len(gemini_feature_dict["feature"])
                gemini_feature_dict["fname"] = fcg_item[:-21] + str("||||") + node
                if gemini_feature_dict["n_num"] >=5 and gemini_feature_dict["n_num"] <= 500:
                    write_list.append(gemini_feature_dict)
            if write_list != []:
                with open(save_path, "w") as wf:
                    for line in write_list:
                        wf.write(json.dumps(line) + "\n")

if __name__ == "__main__":
    fcg_path = "/home/wangyongpan/paper/reuse_detection/code/reuse_minifcg/fcg_6.4/"
    feature_path = "/home/wangyongpan/paper/reuse_detection/code/reuse_minifcg/minifcg_no_analog/6.4_1_500/"
    files = os.listdir(fcg_path)
    p_list = []
    for i in range(PROCESS_NUM):
        p = Process(target=get_sub_g, args=(fcg_path, files[int((i/PROCESS_NUM)*len(files)):int(((i+1)/PROCESS_NUM)*len(files))], feature_path, "analo"))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()


