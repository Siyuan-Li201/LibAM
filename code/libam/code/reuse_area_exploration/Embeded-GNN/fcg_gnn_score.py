import json
import os
import pickle
import time
from multiprocessing import Process

import tqdm
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn.functional as F
import copy

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

def calculate_final_score(gnn_score, fcg_scale, align_num, fcg_num):
    fcg_factor = fcg_scale / fcg_num
    align_factor = align_num / fcg_num
    return gnn_score * fcg_factor * align_factor
    pass

def calculate_gnn_score(func_embeddings, fcg_nums, files_rein, save_path, reinforment_path, gnn, timepath):
    true_num = 0
    false_num = 0
    for file in tqdm.tqdm(files_rein, desc="it is calculate gnn score..."):
        
        if not os.path.exists(os.path.join(save_path, file)):
            start = time.time()
            reinforcement_file = os.path.join(reinforment_path, file)
            fcg_results = {}
            detect_bin = file.split("----")[0].replace("|||", "_")
            candidates_bin = file.split("----")[1].split("_feature_result")[0]
            with open(reinforcement_file, "r") as f:
                re_results = json.load(f)
            for func_pair, fcgs in re_results.items():
                feature = []
                # c_feat = copy.deepcopy(fcgs)
                for func in fcgs["obj_fcg"]["feature"]:
                    func_name = detect_bin + "|||" + func
                    if func_name in func_embeddings:
                        embed = func_embeddings[func_name][0]
                        true_num += 1
                    else:
                        false_num += 1
                        embed = list(np.array([0.001 for i in range(64)]))
                    feature.append(embed)
                obj_fcg = fcgs["obj_fcg"].copy()
                obj_fcg["embeddings"] = feature
                obj_embedding = embed_by_feat_torch(obj_fcg, gnn)
                feature = []
                for func in fcgs["cdd_fcg"]["feature"]:
                    func_name = candidates_bin + "|||" + func
                    if func_name in func_embeddings:
                        embed = func_embeddings[func_name][0]
                        true_num += 1
                    else:
                        false_num += 1
                        embed = list(np.array([0.001 for i in range(64)]))
                    feature.append(embed)
                cdd_fcg = fcgs["cdd_fcg"].copy()
                cdd_fcg["embeddings"] = feature
                # start = time.time()
                cdd_embedding = embed_by_feat_torch(cdd_fcg, gnn)
                # print(time.time() -start)
                gnn_score = F.cosine_similarity(obj_embedding, cdd_embedding, eps=1e-10, dim=1)
                gnn_score = (1 + gnn_score.cpu().detach().numpy()[0]) / 2.0
                fcgs["gnn_score"] = str(gnn_score)
                fcgs["obj_full_fcg_num"] = str(fcg_nums[detect_bin])
                fcgs["final_score"] = str(calculate_final_score(gnn_score, fcgs["fcg_scale"][0], fcgs["alignment_num"], fcg_nums[detect_bin]))
                fcg_results[func_pair] = fcgs
            
            end = time.time()
            timecost = end - start
            save_file = os.path.join(save_path, file)
            with open(save_file, "w") as f:
                json.dump(fcg_results, f)
            with open(os.path.join(timepath, file), "w") as ff:
                ff.write(str(timecost))
    print("true_num:" + str(true_num))
    print("false_num:" + str(false_num))
    pass

def main(obj_func_embeddings_path, cdd_func_embeddings_path, tar_fcgs_path, candidate_fcgs_path, save_path, reinforment_path, gnn_model_path, timepath):
    # obj_func_embeddings_path = "/data/wangyongpan/paper/reuse_detection/datasets/paper_datasets/embeddings/isrd_target_embeddings_torch_best.json"
    # cdd_func_embeddings_path = "/data/wangyongpan/paper/reuse_detection/datasets/paper_datasets/embeddings/isrd_candidates_embeddings_torch_best.json"
    # fcgs_path = "/data/wangyongpan/paper/reuse_detection/datasets/paper_datasets/isrd_target_fcg/"
    # save_path = "/data/wangyongpan/paper/reuse_detection/datasets/results/libAE2.0_result/TPL_detection_result/1109_5_libae_paper_top50_gnn_analog_0.001/"
    # reinforment_path = "/data/wangyongpan/paper/reuse_detection/datasets/results/libAE2.0_result/TPL_detection_result/1109_5_libae_paper_top50/"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(timepath):
        os.makedirs(timepath)
    
    gnn = torch.load(gnn_model_path)
    fcgs_num = {}
    torch.multiprocessing.set_start_method('spawn', force=True)
    for fcg_p in os.listdir(tar_fcgs_path):
        with open(os.path.join(tar_fcgs_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
    for fcg_p in os.listdir(candidate_fcgs_path):
        with open(os.path.join(candidate_fcgs_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
    with open(obj_func_embeddings_path, "r") as f:
        obj_func_embeddings = json.load(f)
    with open(cdd_func_embeddings_path, "r") as f:
        cdd_func_embeddings = json.load(f)
    for func, embed in obj_func_embeddings.items():
        if func not in cdd_func_embeddings:
            cdd_func_embeddings[func] = embed
        # else:
        #     print("[ERROR]candidates and target have the same func name---{0}...".format(func))

    rein_file = list(os.listdir(reinforment_path))
    rein_files = []
    for f in rein_file:
        if not os.path.exists(save_path + f):
            rein_files.append(f)
    process_num = 15
    p_list = []
    for i in range(process_num):
        files = rein_files[int((i) / process_num * len(rein_files)): int((i + 1) / process_num * len(rein_files))]
        p = Process(target=calculate_gnn_score, args=(cdd_func_embeddings, fcgs_num, files, save_path, reinforment_path, gnn, timepath))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()
    # calculate_gnn_score(cdd_func_embeddings, fcgs_num)

if __name__ == '__main__':
    main()
