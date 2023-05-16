import json, sys
import os
import pickle

import networkx as nx
import tqdm
sys.path.append(".")
from settings import DATA_PATH

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

def draw_graph(graph, png_path, matched_nodes, grounds_dec, error_nodes):
    nodes = list(graph.nodes())
    try:
        pydot_graph = nx.nx_pydot.to_pydot(graph)
    except:
        print("{} transform networkx to pydot error!".format(png_path.split("----")[0]))
        return
    for mn in matched_nodes:
        try:
            pydot_graph.get_node(mn)[0].set_color("blue")
            pydot_graph.get_node(mn)[0].set_style("filled")

        except:
            if mn in nodes:
                pydot_graph.get_node('"' + mn + '"')[0].set_color("blue")
                pydot_graph.get_node('"' + mn + '"')[0].set_style("filled")
            else:
                print(mn + " node not exist in " + png_path.split("/")[-1].split("----")[0])
    for mn in grounds_dec:
        try:
            pydot_graph.get_node(mn)[0].set_color("red")
            pydot_graph.get_node(mn)[0].set_style("filled")
        except:
            if mn in nodes:
                pydot_graph.get_node('"' + mn + '"')[0].set_color("red")
                pydot_graph.get_node('"' + mn + '"')[0].set_style("filled")
            else:
                print(mn + " node not exist in " + png_path.split("/")[-1].split("----")[0])
    for mn in error_nodes:
        try:
            pydot_graph.get_node(mn)[0].set_color("green")
            pydot_graph.get_node(mn)[0].set_style("filled")
        except:
            if mn in nodes:
                pydot_graph.get_node('"' + mn + '"')[0].set_color("green")
                pydot_graph.get_node('"' + mn + '"')[0].set_style("filled")
            else:
                print(mn + " node not exist in " + png_path.split("/")[-1].split("----")[0])
    pydot_graph.set_rankdir('LR')
    try:
        pydot_graph.write_png(os.path.join(png_path))
    except:
        print("{} save is error!".format(png_path))


def calculate_final_score(gnn_score, fcg_scale, align_num, fcg_num):
    fcg_factor = fcg_scale / fcg_num
    align_factor = align_num / fcg_num
    return gnn_score * fcg_factor * align_factor
    pass

def get_connect_num(node, fcg:nx.DiGraph, nodes):
    if len(nodes) >= 5:
        return 1
    for in_edge in fcg.in_edges(node):
        if in_edge[0] not in nodes:
            nodes.append(in_edge[0])
            status = get_connect_num(in_edge[0], fcg, nodes)
            if status == 1:
                return 1
    for out_edge in fcg.out_edges(node):
        if out_edge[1] not in nodes:
            nodes.append(out_edge[1])
            status = get_connect_num(out_edge[1], fcg, nodes)
            if status == 1:
                return 1
    if len(nodes) < 5:
        return 0
    pass

def get_connect_nodes_in_fcg(fcg:nx.DiGraph):
    init_nodes = fcg.nodes()
    final_nodes = []
    single_nodes = []
    delect_nodes = set()
    for node in init_nodes:
        if node in list(delect_nodes):
            continue
        if len(fcg.in_edges(node)) == 0 and len(fcg.out_edges(node)) == 0:
            single_nodes.append(node)
            continue
        nodes = [node]
        if get_connect_num(node, fcg, nodes) == 0:
            delect_nodes.update(set(nodes))
            continue
        final_nodes.append(node)
    return final_nodes
    pass

def get_ground_truth(detect_reuse_areas, tar_fcg_path, cdd_fcg_path, ground_truth="libae_ground_truth.json", save_path=""):
    tar_fcg_files = list(os.listdir(tar_fcg_path))
    cdd_fcg_files = list(os.listdir(cdd_fcg_path))
    with open(ground_truth, "r") as f:
        cy_ground_truth = json.load(f)
    true_reuse_area = {}
    true_reuse_areas = {}
    for key in tqdm.tqdm(detect_reuse_areas, desc="it is calculating ground truth..."):
        # key = key.split("----")[0]
        key_file = key + "_fcg.pkl"
        if key_file in tar_fcg_files:
            g = nx.read_gpickle(os.path.join(tar_fcg_path, key_file))
            father_list = set(get_connect_nodes_in_fcg(g))
            child_lists = {}
            if key.split("_")[0] in cy_ground_truth.keys():
                child_list = cy_ground_truth[key.split("_")[0]]
                for child in child_list:
                    child_file = child + "_fcg.pkl"
                    if child_file in cdd_fcg_files:
                        child_lists[child] = list(set(get_connect_nodes_in_fcg(nx.read_gpickle(os.path.join(cdd_fcg_path, child_file)))))
                    else:
                        print(child_file + " not exists...child file")
            # child_lists = set(child_lists)
            ans = {}
            for chi, child_l in child_lists.items():
                ans[chi] = list(father_list.intersection(set(child_l)))
            reuse_ans = {}
            for chia, aa in ans.items():
                reuse_ans[chia] = []
                for a in aa:
                    if a == "main" or "_libc_start_main" in a:
                        continue 
                    nodes = [a]
                    succs = [[]]
                    add_child(a, g, nodes, succs, set())
                    reuse_ans[chia].extend(nodes)
                reuse_ans[chia] = list(set(reuse_ans[chia]))
            true_reuse_area[key] = reuse_ans
            true_reuse_areas[key] = (reuse_ans, g)
        else:
            print(key_file + " not exists...key file")
    with open(save_path.split(".")[0] + ".json", "w") as f:
        json.dump(true_reuse_area, f)
    with open(save_path, "wb") as f:
        pickle.dump(true_reuse_areas, f)
    return true_reuse_areas

def area_eval(detect_result, ground_results, save_path):
    all = ["arm_O0", "arm_O1", "arm_O2", "arm_O3", "x86_O0", "x86_O1", "x86_O2", "x86_O3", "x64_O0", "x64_O1", "x64_O2", "x64_O3"]
    archs = ["arm_O2", "x86_O2", "x64_O2"]
    opts = ["x64_O0", "x64_O1", "x64_O2", "x64_O3"]
    mode_list = ["isrd", "x64_O0", "x64_O1", "x64_O2", "x64_O3", "opti_average", "arm_O2", "x86_O2", "x64_O2", "arch_average", "all"]

    arch_results = {}
    opt_results = {}
    for arch in archs:
        arch_results[arch] = {}
    for opt in opts:
        opt_results[opt] = {}
    mix_recall = 0.0
    mix_precision = 0.0
    mix_f1 = 0.0
    mix_num = 0
    for key, value in tqdm.tqdm(detect_result.items()):
        # if "xz" in key or "gipfeli" in key or "libdeflate" in key or "zstd" in key \
        # or "brotli" in key or "lzo" in key or "quicklz" in key or "liblzg" in key:
        #     continue
        # if "xz" in key or "precomp" in key:
        #     continue
        if key in ground_results:
            gv = ground_results[key][0]
            graph = ground_results[key][1]
            ground_value = []
            dec_value = []
            for k, v in value.items():
                # if "xz" in k or "gipfeli" in k or "libdeflate" in k or "zstd" in k \
                # or "brotli" in k or "lzo" in k or "quicklz" in k or "liblzg" in k:
                #     continue
                # if "xz" in k or "precomp" in k:
                #     continue
                if k in gv:
                    ground_value.extend(gv[k])
                dec_value.extend(v)
            if len(ground_value) == 0:
                continue
            ground_value = list(set(ground_value))
            dec_value = list(set(dec_value))
            insect_value = list(set(ground_value).intersection(set(dec_value)))
            ground_dec = set(ground_value) - set(insect_value)
            dec_dec = set(dec_value) - set(insect_value)
            # if "precomp" in key:
            #     draw_graph(graph, "D:\\reuse_detection\\area_dataset\\adjust\\" + key + ".png", list(insect_value), list(ground_dec), list(dec_dec))
            recall = len(insect_value) / len(ground_value)
            try:
                precision = len(insect_value) / len(dec_value)
            except:
                precision = 0.0
            try:
                F1 = 2 * (recall * precision) / (precision + recall)
            except:
                F1 = 0.0
            ans_value = (precision, recall, F1)
            mix_recall += recall
            mix_precision += precision
            mix_num += 1
            mix_f1 += F1
            if "_" in key:
                arch_opt = key.split("_")[1] + "_" + key.split("_")[2]
                if arch_opt in arch_results:
                    arch_results[arch_opt][key] = ans_value
                if arch_opt in opt_results:
                    opt_results[arch_opt][key] = ans_value
    print("mix_recall:", str(mix_recall/mix_num))
    print("mix_precision:", str(mix_precision/mix_num))
    print("mix_f1:", str(mix_f1/mix_num))
    for key, value in tqdm.tqdm(detect_result.items()):
        # if "xz" in key or "gipfeli" in key or "libdeflate" in key or "zstd" in key \
        # or "brotli" in key or "lzo" in key or "quicklz" in key or "liblzg" in key:
        #     continue
        # if "xz" in key or "precomp" in key:
        #     continue
        if key in ground_results:
            gv = ground_results[key][0]
            graph = ground_results[key][1]
            for k, v in value.items():
                # if "xz" in k or "gipfeli" in k or "libdeflate" in k or "zstd" in k \
                # or "brotli" in k or "lzo" in k or "quicklz" in k or "liblzg" in k:
                #     continue
                # if "xz" in k or "precomp" in k:
                #     continue
                if k in gv:
                    ground_value = list(set(gv[k]))
                    if len(ground_value) == 0:
                        continue
                    dec_value = list(set(v))
                    insect_value = list(set(ground_value).intersection(set(dec_value)))
                    ground_dec = set(ground_value) - set(insect_value)
                    dec_dec = set(dec_value) - set(insect_value)
                    # if "precomp" in key:
                    # draw_graph(graph, "D:\\reuse_detection\\area_dataset\\adjust\\" + key + "----" + k + ".png", list(insect_value), list(ground_dec), list(dec_dec))
                    recall = len(insect_value) / len(ground_value)
                    try:
                        precision = len(insect_value) / len(dec_value)
                    except:
                        precision = 0.0
                    try:
                        F1 = 2 * (recall * precision) / (precision + recall)
                    except:
                        F1 = 0.0
                    ans_value = (precision, recall, F1)
                    if "_" in key:
                        arch_opt = key.split("_")[1] + "_" + key.split("_")[2]
                        if arch_opt in arch_results:
                            arch_results[arch_opt][key + "----" + k] = ans_value
                        if arch_opt in opt_results:
                            opt_results[arch_opt][key + "----" + k] = ans_value

    for key, value in arch_results.items():
        # if len(value) == 0:
        #     continue
        precision = 0
        recall = 0
        F1 = 0
        child_num = 0
        tar_num = 0
        aver_precision = 0
        aver_recall = 0
        aver_f1 = 0
        for k, v in value.items():
            if "----" in k:
                precision += v[1]
                recall += v[0]
                F1 += v[2]
                child_num += 1
            else:
                aver_precision += v[1]
                aver_recall += v[0]
                aver_f1 += v[2]
                tar_num += 1
        if child_num > 0:
            arch_results[key]["all_metrics"] = (precision / child_num, recall / child_num, F1 / child_num)
        else:
            arch_results[key]["all_metrics"] = (0, 0, 0)
        if tar_num > 0:
            arch_results[key]["average_metrics"] = (aver_precision / tar_num, aver_recall / tar_num, aver_f1 / tar_num)
        else:
            arch_results[key]["average_metrics"] = (0, 0, 0)
    for key, value in opt_results.items():
        # if len(value) == 0:
        #     continue
        precision = 0
        recall = 0
        F1 = 0
        child_num = 0
        tar_num = 0
        aver_precision = 0
        aver_recall = 0
        aver_f1 = 0
        for k, v in value.items():
            if "----" in k:
                precision += v[1]
                recall += v[0]
                F1 += v[2]
                child_num += 1
            else:
                aver_precision += v[1]
                aver_recall += v[0]
                aver_f1 += v[2]
                tar_num += 1
        if child_num > 0:
            opt_results[key]["all_metrics"] = (precision / child_num, recall / child_num, F1 / child_num)
        else:
            opt_results[key]["all_metrics"] = (0, 0, 0)
        if tar_num > 0:
            opt_results[key]["average_metrics"] = (aver_precision / tar_num, aver_recall / tar_num, aver_f1 / tar_num)
        else:
            opt_results[key]["average_metrics"] = (0, 0, 0)
    # print("archs_results:", str(arch_results))
    # print("opt_results:", str(opt_results))
    results = {}
    final_result = {}
    results["archs_results"] = arch_results
    p = 0
    f = 0
    r = 0
    for key, value in arch_results.items():
        final_result[key] = value["average_metrics"]
        p += value["average_metrics"][0]
        r += value["average_metrics"][1]
        f += value["average_metrics"][2]
        final_result["arch_aver"] = [p / 3.0, r / 3.0, f / 3.0]
        final_result["mix_result"] = [mix_precision, mix_recall, mix_f1]
    p = 0
    f = 0
    r = 0
    for key, value in opt_results.items():
        final_result[key] = value["average_metrics"]
        p += value["average_metrics"][0]
        r += value["average_metrics"][1]
        f += value["average_metrics"][2]
        final_result["opt_aver"] = [p / 4.0, r / 4.0, f / 4.0]
    results["opt_results"] = opt_results
    results["final_results"] = final_result
    with open(save_path, "w") as f:
        json.dump(results ,f)
    pass

def gemini_area_aval(result_path, ground_results, save_path):
    with open(result_path, "r") as f:
        gemini_results = json.load(f)
    gemini = {}
    for g, value in gemini_results.items():
        g = g.replace("___", "_")
        if value != {}:
            gv = {}
            for r, v in value.items():
                r = r.replace("___", "_").split("_datasets_isrd_to_gemini")[0]
                svs = []
                for sv in list(set(v)):
                    sv = sv.split("|||")[-1]
                    svs.append(sv)
                gv[r] = svs
            gemini[g] = gv
    with open(save_path, "w") as f:
        json.dump(gemini, f)
    area_eval(gemini, ground_results, save_path="Gemini_metrics.json")
    pass

def libdb_area_eval(result_path, ground_results, save_path):
    with open(result_path, "r") as f:
        libdb_results = json.load(f)
    libdb = {}
    for g, value in libdb_results.items():
        g = g.replace("___", "_")
        if value != {}:
            gv = {}
            for r, v in value.items():
                r = r.replace("___", "_").split("_datasets_isrd_to_gemini")[0]
                svs = []
                for vv in v:
                    vv = vv[0].split("|||")[-1]
                    svs.append(vv)
                svs = list(set(svs))
                gv[r] = svs
            libdb[g] = gv
    with open(save_path, "w") as f:
        json.dump(libdb, f)
    area_eval(libdb, ground_results, save_path="libDB_metrics.json")
    pass

def libae_area_eval(results_path, save_path, align_thre=0):
    detect_reuse_areas = {}
    for file in tqdm.tqdm(os.listdir(results_path), desc="it is calculating detect results..."):
        align_file = os.path.join(results_path, file)
        detect_reuse_area = {}
        # file = file.replace("___", "_")
        tar_name = file.split("_reuse_area")[0]
        
        # if int(file.split(".json")[0].split("_")[-1]) < align_thre:
        #     continue
        with open(align_file, "r") as f:
            align_result = json.load(f)
        for key, value in align_result.items():
            # if value["alignment_num"] < align_thre:
            #     continue
            cand_name = key
            for node_pair in value:
                for area_item in value[node_pair]:
                    detect_reuse_area[cand_name] = list(set(area_item["obj_fcg"]["feature"]))
        if tar_name in detect_reuse_areas:
            detect_reuse_areas[tar_name].update(detect_reuse_area)
        else:
            detect_reuse_areas[tar_name] = detect_reuse_area
    with open(save_path, "w") as f:
        json.dump(detect_reuse_areas, f)
    return detect_reuse_areas



def get_area_result(area_result_path, save_path, ground_truth_path, tar_fcg_path, cdd_fcg_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # result_path = os.path.join(save_path, "libAE_align_results.json")
    
    area_result_dict = {}
    for area_result_path_item in tqdm.tqdm(os.listdir(area_result_path)):
        area_result = json.load(open(os.path.join(area_result_path, area_result_path_item), "r"))
        tar_bin = area_result_path_item.split("_reuse_area")[0]
        area_result_dict[tar_bin] = []
        for candidate_bin in area_result:
            for node_pair in area_result[candidate_bin]:
                for obj_func in area_result[candidate_bin][node_pair][0]["obj_fcg"]["feature"]:
                    if obj_func not in area_result_dict[tar_bin]:
                        area_result_dict[tar_bin].append(obj_func)
    
    ground_truth = json.load(open(ground_truth_path, "r"))
    ground_truth_dict = {}
    for gt_item in ground_truth:
        tar_bin = gt_item.split("___")[0]
        cdd_bin = gt_item.split("___")[1]
        
        if tar_bin not in ground_truth_dict:
            ground_truth_dict[tar_bin] = []
        if cdd_bin not in ground_truth_dict:
            ground_truth_dict[cdd_bin] = []
        
        for reuse_func in ground_truth[gt_item]:
            if reuse_func not in ground_truth_dict[tar_bin]:
                ground_truth_dict[tar_bin].append(reuse_func)
            if reuse_func not in ground_truth_dict[cdd_bin]:
                ground_truth_dict[cdd_bin].append(reuse_func)
        
    all_score_dict = {}
    all_pre = 0
    all_recall = 0
    all_F1 = 0
    i = 0
    for tar_bin in area_result_dict:
        if tar_bin.split("_")[0] in ground_truth_dict:
            i += 1
            all_score_dict[tar_bin] = {}
            tar_fcg = nx.read_gpickle(os.path.join(tar_fcg_path, tar_bin+"_fcg.pkl"))
            tar_nodes = tar_fcg.nodes()
            tar_reuse_funcs = []
            for cdd_point in area_result_dict[tar_bin]:
                
                if cdd_point in tar_nodes:
                    tar_reuse_funcs.append(cdd_point)
                    desc_funclist = nx.descendants(tar_fcg, cdd_point)
                    for desc_func in desc_funclist:
                        if desc_func in tar_nodes and desc_func not in tar_reuse_funcs:
                            tar_reuse_funcs.append(desc_func)
            tar_ground_truth_reuse_func = []
            for gt_point in ground_truth_dict[tar_bin.split("_")[0]]:
                
                if gt_point in tar_nodes:
                    tar_ground_truth_reuse_func.append(gt_point)
                    desc_funclist = nx.descendants(tar_fcg, gt_point)
                    for desc_func in desc_funclist:
                        if desc_func in tar_nodes and desc_func not in tar_ground_truth_reuse_func:
                            tar_ground_truth_reuse_func.append(desc_func)   
            
            true_positive = list(set(tar_reuse_funcs).intersection(set(tar_ground_truth_reuse_func)))
            false_positive = list(set(tar_reuse_funcs) - set(tar_ground_truth_reuse_func))
            false_negative = list(set(tar_ground_truth_reuse_func) - set(tar_reuse_funcs))
            try:
                Precision = len(true_positive) / len(set(tar_reuse_funcs))
                Recall = len(true_positive) / len(set(tar_ground_truth_reuse_func))
                F1_score = 2*Precision*Recall/(Precision+Recall)
            except:
                Precision = Recall = F1_score = 0
            all_pre += Precision
            all_recall += Recall
            all_F1 += F1_score
            
            print(tar_bin + " Precision: {}  Recall: {}  F1: {}".format(Precision, Recall, F1_score) )
            json.dump(true_positive, open(os.path.join(save_path, tar_bin+"_true_positive.json"), "w"))
            json.dump(false_positive, open(os.path.join(save_path, tar_bin+"_false_positive.json"), "w"))
            json.dump(false_negative, open(os.path.join(save_path, tar_bin+"_false_negative.json"), "w"))
            
            all_score_dict[tar_bin]["Precision"] = Precision
            all_score_dict[tar_bin]["Recall"] = Recall
            all_score_dict[tar_bin]["F1_score"] = F1_score
        
    json.dump(all_score_dict, open(os.path.join(save_path, "all_result.json"), "w"))
    print("All Precision: {}  Recall: {}  F1: {}".format(all_pre/i, all_recall/i, all_F1/i) )


def not_in_target(obj_item, all):
    error_flag = 0
    for all_item in all:
        if all_item in obj_item:
            error_flag = 1
            break
    return error_flag


def is_in_target(obj_item, opti):
    error_flag = 1
    for all_item in opti:
        if all_item in obj_item:
            error_flag = 0
            break
    return error_flag


def is_mode(obj_item, mode_item):
    if mode_item in obj_item:
        return 0
    else:
        return 1


def deal_with_sndfile(libae_result):
    # all = ["arm_O0", "arm_O1", "arm_O2", "arm_O3", "x86_O0", "x86_O1", "x86_O2", "x86_O3", "x64_O0", "x64_O1", "x64_O2", "x64_O3"]
    libae_result_new = copy.deepcopy(libae_result)
    
    
    for arch_opti in all:
        libsndfile_sndfile2k = []
        for obj_item in libae_result:
            if "sndfile" in obj_item and arch_opti in obj_item:
                # for cdd_item in libae_result[obj_item]:
                #     if cdd_item not in libsndfile_sndfile2k:
                #         libsndfile_sndfile2k.append(cdd_item)
                libae_result_new.pop(obj_item)
        
        # libae_result_new["libsndfile-sndfile2k_"+arch_opti] = libsndfile_sndfile2k
    
    # for obj_item in libae_result:
    #     if "sndfile" in obj_item:
    #         find_flag = False
    #         for arch_opti in all:
    #             if arch_opti in obj_item:
    #                 find_flag = True
    #         if False == find_flag:
    #             libsndfile_sndfile2k = []
    #             for cdd_item in libae_result[obj_item]:
    #                 if cdd_item not in libsndfile_sndfile2k:
    #                     libsndfile_sndfile2k.append(cdd_item)
    # libae_result_new["libsndfile-sndfile2k"] = libsndfile_sndfile2k
                        
    
    return libae_result_new



def test_func(match_result, ground_truth):
    # score = {}
    
    # full_P = 0
    # full_R = 0
    # full_F1 = 0
    
    # test_num = 0
    
    # for detected_item in match_result:
    #     if detected_item in ground_truth:
    # test_num += 1
    detected_data = match_result
    groung_truth_data = ground_truth
    
    TP = TN = FP = FN = 0
    TN = 85 - len(detected_data)
    for reuse_item in detected_data:
        reuse_item = reuse_item.split("_datasets_isrd_to_gemini")[0]
        if reuse_item in groung_truth_data:
            TP += 1
        else:
            FP += 1
    for reuse_item in groung_truth_data:
        if reuse_item not in detected_data:
            FN += 1
    
    try:
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1_s = 2*Precision*Recall/(Precision+Recall)
    except Exception as e:
        Precision = 0
        Recall = 0
        F1_s = 0
    
    
    return (Precision, Recall, F1_s)

def cal_score(mode_func, arg2, save_path, mode_item, area_result_dict, ground_truth_dict, tar_fcg_path):
    # with open(os.path.join(save_path, "TPL_result_"+mode_item), "w") as result_f:
    #     full_P = 0
    #     full_R = 0
    #     full_F1 = 0
    #     candidate_num = 0
        
    #     libae_result_deal = deal_with_sndfile(area_result_dict)
        
    #     for obj_item in libae_result_deal:
    #         if mode_func(obj_item, arg2) == 0:
    #             candidate_num += 1
    #             (Precision, Recall, F1_s) = test_func(libae_result_deal[obj_item], ground_truth_dict[obj_item.split("_")[0]])
            
    #             print(obj_item+" score: P:{} R:{} F1:{}".format(Precision, Recall, F1_s))
    #             result_f.write(obj_item+" score: P:{} R:{} F1:{}\n".format(Precision, Recall, F1_s))
    #             full_P += Precision
    #             full_R += Recall
    #             full_F1 += F1_s
    #     if candidate_num > 0:
    #         print("Full Precision: {}  Recall: {}  F1: {}".format(full_P/candidate_num, full_R/candidate_num, full_F1/candidate_num) )
    #         result_f.write("Full Precision: {}  Recall: {}  F1: {}\n".format(full_P/candidate_num, full_R/candidate_num, full_F1/candidate_num) )
    #     else:
    #         print("Full Precision: {}  Recall: {}  F1: {}".format(full_P, full_R, full_F1) )
    #         result_f.write("Full Precision: {}  Recall: {}  F1: {}\n".format(full_P, full_R, full_F1) )

    all_score_dict = {}
    result_list = []
    all_pre = 0
    all_recall = 0
    all_F1 = 0
    i = 0
    # all_score_dict = {}
    for tar_bin in area_result_dict:
        if mode_func(tar_bin, arg2) == 0 and tar_bin.split("_")[0] in ground_truth_dict:
            i += 1
            all_score_dict[tar_bin] = {}
            tar_fcg = nx.read_gpickle(os.path.join(tar_fcg_path, tar_bin+"_fcg.pkl"))
            tar_nodes = tar_fcg.nodes()
            tar_reuse_funcs = []
            for cdd_point in area_result_dict[tar_bin]:
                tar_reuse_funcs.append(cdd_point)
                if cdd_point in tar_nodes:
                    desc_funclist = nx.descendants(tar_fcg, cdd_point)
                    for desc_func in desc_funclist:
                        if desc_func not in tar_reuse_funcs:
                            tar_reuse_funcs.append(desc_func)
            tar_ground_truth_reuse_func = []
            for gt_point in ground_truth_dict[tar_bin.split("_")[0]]:
                tar_ground_truth_reuse_func.append(gt_point)
                if gt_point in tar_nodes:
                    desc_funclist = nx.descendants(tar_fcg, gt_point)
                    for desc_func in desc_funclist:
                        if desc_func not in tar_ground_truth_reuse_func:
                            tar_ground_truth_reuse_func.append(desc_func)   
            
            true_positive = list(set(tar_reuse_funcs).intersection(set(tar_ground_truth_reuse_func)))
            false_positive = list(set(tar_reuse_funcs) - set(tar_ground_truth_reuse_func))
            false_negative = list(set(tar_ground_truth_reuse_func) - set(tar_reuse_funcs))
            try:
                Precision = len(true_positive) / len(set(tar_reuse_funcs))
                Recall = len(true_positive) / len(set(tar_ground_truth_reuse_func))
                F1_score = 2*Precision*Recall/(Precision+Recall)
            except:
                Precision = Recall = F1_score = 0
            all_pre += Precision
            all_recall += Recall
            all_F1 += F1_score
            
            print(tar_bin + " Precision: {}  Recall: {}  F1: {}".format(Precision, Recall, F1_score) )
            result_list.append(tar_bin + " Precision: {}  Recall: {}  F1: {}".format(Precision, Recall, F1_score))
            json.dump(true_positive, open(os.path.join(save_path, tar_bin+"_true_positive.json"), "w"))
            json.dump(false_positive, open(os.path.join(save_path, tar_bin+"_false_positive.json"), "w"))
            json.dump(false_negative, open(os.path.join(save_path, tar_bin+"_false_negative.json"), "w"))
            
            all_score_dict[tar_bin]["Precision"] = Precision
            all_score_dict[tar_bin]["Recall"] = Recall
            all_score_dict[tar_bin]["F1_score"] = F1_score
        
    if i != 0:
        json.dump(all_score_dict, open(os.path.join(save_path, "all_result.json"), "w"))
        print("All Precision: {}  Recall: {}  F1: {}".format(all_pre/i, all_recall/i, all_F1/i) )
        result_list.append("All Precision: {}  Recall: {}  F1: {}".format(all_pre/i, all_recall/i, all_F1/i))
        with open(os.path.join(save_path, mode_item+"_all_result.json"), "w") as wf:
            for wf_line in result_list:
                wf.write(wf_line+"\n")


def get_area_result_several(area_result_path, save_path, ground_truth_path, tar_fcg_path, cdd_fcg_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # result_path = os.path.join(save_path, "libAE_align_results.json")
    
    # area_result_func_dict = {}
    # area_result_dict = {}
    # for area_result_path_item in tqdm.tqdm(os.listdir(area_result_path)):
    #     area_result = json.load(open(os.path.join(area_result_path, area_result_path_item), "r"))
    #     tar_bin = area_result_path_item.split("----")[0]
    #     if tar_bin not in area_result_dict:
    #         area_result_dict[tar_bin] = area_result
    #     else:
    #         area_result_dict[tar_bin] = dict(area_result_dict[tar_bin], **area_result)
    # for tar_bin in area_result_dict:
    #     area_result_func_dict[tar_bin] = []
    #     for candidate_bin in area_result_dict[tar_bin]:
    #         for node_pair in area_result_dict[tar_bin][candidate_bin]:
    #             for obj_func in area_result_dict[tar_bin][candidate_bin][node_pair][0]["obj_fcg"]["feature"]:
    #                 if obj_func not in area_result_func_dict[tar_bin]:
    #                     area_result_func_dict[tar_bin].append(obj_func)
    
    
    area_result_func_dict = {}
    for area_result_path_item in tqdm.tqdm(os.listdir(area_result_path)):
        area_result = json.load(open(os.path.join(area_result_path, area_result_path_item), "r"))
        tar_bin = area_result_path_item.split("_reuse_area")[0]
        area_result_func_dict[tar_bin] = []
        for candidate_bin in area_result:
            if candidate_bin not in area_result_path_item:
                for node_pair in area_result[candidate_bin]:
                    for obj_func in area_result[candidate_bin][node_pair][0]["obj_fcg"]["feature"]:
                        if obj_func not in area_result_func_dict[tar_bin]:
                            area_result_func_dict[tar_bin].append(obj_func)
    
    ground_truth = json.load(open(ground_truth_path, "r"))
    ground_truth_dict = {}
    for gt_item in ground_truth:
        tar_bin = gt_item.split("___")[0]
        cdd_bin = gt_item.split("___")[1]
        
        if tar_bin not in ground_truth_dict:
            ground_truth_dict[tar_bin] = []
        if cdd_bin not in ground_truth_dict:
            ground_truth_dict[cdd_bin] = []
        
        for reuse_func in ground_truth[gt_item]:
            if reuse_func not in ground_truth_dict[tar_bin]:
                ground_truth_dict[tar_bin].append(reuse_func)
            if reuse_func not in ground_truth_dict[cdd_bin]:
                ground_truth_dict[cdd_bin].append(reuse_func)
        

    
    all = ["arm_O0", "arm_O1", "arm_O2", "arm_O3", "x86_O0", "x86_O1", "x86_O2", "x86_O3", "x64_O0", "x64_O1", "x64_O2", "x64_O3"]
    arch_average = ["arm_O2", "x86_O2", "x64_O2"]
    opti_average = ["x64_O0", "x64_O1", "x64_O2", "x64_O3"]
    
    mode_list = ["isrd", "x64_O0", "x64_O1", "x64_O2", "x64_O3", "opti_average", "arm_O2", "x86_O2", "x64_O2", "arch_average", "all"]
    
    for mode_item in mode_list:
        if mode_item == "isrd":
            cal_score(not_in_target, all, save_path, mode_item, area_result_func_dict, ground_truth_dict, tar_fcg_path)
        if mode_item == "opti_average":
            cal_score(is_in_target, opti_average, save_path, mode_item, area_result_func_dict, ground_truth_dict, tar_fcg_path)
        elif mode_item == "arch_average":
            cal_score(is_in_target, arch_average, save_path, mode_item, area_result_func_dict, ground_truth_dict, tar_fcg_path)
        elif mode_item == "all":
            cal_score(is_in_target, all, save_path, mode_item, area_result_func_dict, ground_truth_dict, tar_fcg_path)
        else:
            cal_score(is_mode, mode_item, save_path, mode_item, area_result_func_dict, ground_truth_dict, tar_fcg_path)
    
 
def preprocess(g):
    nodes = g.nodes()
    
    del_nodes = []
    for node in nodes:
        # if node == "BZ2_bzBuffToBuffCompress":
        #     print("BZ2_bzBuffToBuffCompress")
        if g.degree(node) == 0:
            del_nodes.append(node)
    for del_node in del_nodes:
        g.remove_node(del_node)
            
    return g   
    
def get_area_scale(area_result_path, save_path, tar_fcg_path, cdd_fcg_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tar_area_result_dict = {}
    cdd_area_result_dict = {}
    for area_result_path_item in tqdm.tqdm(os.listdir(area_result_path)):
        area_result = json.load(open(os.path.join(area_result_path, area_result_path_item), "r"))
        tar_bin = area_result_path_item.split("_reuse_area")[0]
        tar_area_result_dict[tar_bin] = {}
        cdd_area_result_dict[tar_bin] = {}
        for candidate_bin in area_result:
            tar_area_result_dict[tar_bin][candidate_bin] = []
            cdd_area_result_dict[tar_bin][candidate_bin] = []
            for node_pair in area_result[candidate_bin]:
                for obj_func in area_result[candidate_bin][node_pair][0]["obj_fcg"]["feature"]:
                    if obj_func not in tar_area_result_dict[tar_bin][candidate_bin]:
                        tar_area_result_dict[tar_bin][candidate_bin].append(obj_func)    
                for cdd_func in area_result[candidate_bin][node_pair][0]["cdd_fcg"]["feature"]:
                    if cdd_func not in cdd_area_result_dict[tar_bin][candidate_bin]:
                        cdd_area_result_dict[tar_bin][candidate_bin].append(cdd_func)    
    
    reuse_scale_dict = {}    
    for tar_bin in cdd_area_result_dict:
        for cdd_bin in cdd_area_result_dict[tar_bin]:
            
            tar_fcg = os.path.join(tar_fcg_path, tar_bin + "_fcg.pkl")
            cdd_fcg = os.path.join(cdd_fcg_path, cdd_bin + "_fcg.pkl")
            
            tar_g = preprocess(nx.read_gpickle(tar_fcg))
            cdd_g = preprocess(nx.read_gpickle(cdd_fcg))
            
            tar_funcs = list(tar_g.nodes())
            cdd_funcs = list(cdd_g.nodes())
            
            cdd_reuse_funcs_item = cdd_area_result_dict[tar_bin][cdd_bin]
            cdd_reuse_funcs = []
            for anchor_point in cdd_reuse_funcs_item:
                if anchor_point in cdd_funcs and anchor_point not in cdd_reuse_funcs:
                    cdd_reuse_funcs.append(anchor_point)
                    desc_funclist = nx.descendants(cdd_g, anchor_point)
                    for desc_func in desc_funclist:
                        if desc_func not in cdd_reuse_funcs:
                            cdd_reuse_funcs.append(desc_func)           
            
            tar_reuse_funcs_item = tar_area_result_dict[tar_bin][cdd_bin]
            tar_reuse_funcs = []
            for anchor_point in tar_reuse_funcs_item:
                if anchor_point in tar_funcs and anchor_point not in tar_reuse_funcs:
                    tar_reuse_funcs.append(anchor_point)
                    desc_funclist = nx.descendants(tar_g, anchor_point)
                    for desc_func in desc_funclist:
                        if desc_func not in tar_reuse_funcs:
                            tar_reuse_funcs.append(desc_func)  
            
            tar_func_num = len(tar_funcs)
            cdd_func_num = len(cdd_funcs)
            tar_reuse_num = len(tar_reuse_funcs)
            cdd_reuse_num = len(cdd_reuse_funcs)
            
            tar_reuse_scale = tar_reuse_num / tar_func_num
            cdd_reuse_scale = cdd_reuse_num / cdd_func_num
                        
            reuse_scale_dict[tar_bin+"___"+cdd_bin] = {}
            reuse_scale_dict[tar_bin+"___"+cdd_bin]["tar_reuse_scale"] = tar_reuse_scale
            reuse_scale_dict[tar_bin+"___"+cdd_bin]["cdd_reuse_scale"] = cdd_reuse_scale
            reuse_scale_dict[tar_bin+"___"+cdd_bin]["tar_reuse_num"] = tar_reuse_num
            reuse_scale_dict[tar_bin+"___"+cdd_bin]["tar_func_num"] = tar_func_num
            reuse_scale_dict[tar_bin+"___"+cdd_bin]["cdd_reuse_num"] = cdd_reuse_num
            reuse_scale_dict[tar_bin+"___"+cdd_bin]["cdd_func_num"] = cdd_func_num
                        
    json.dump(reuse_scale_dict, open(os.path.join(save_path, "reuse_scale.json"), "w"))                 
    # return
            
            


def get_area_result_for_each(area_result_path, save_path, ground_truth_path, tar_fcg_path, cdd_fcg_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # result_path = os.path.join(save_path, "libAE_align_results.json")
    
    area_result_dict = {}
    for area_result_path_item in tqdm.tqdm(os.listdir(area_result_path)):
        area_result = json.load(open(os.path.join(area_result_path, area_result_path_item), "r"))
        tar_bin = area_result_path_item.split("_reuse_area")[0]
        area_result_dict[tar_bin] = {}
        for candidate_bin in area_result:
            area_result_dict[tar_bin][candidate_bin] = []
            for node_pair in area_result[candidate_bin]:
                for obj_func in area_result[candidate_bin][node_pair][0]["obj_fcg"]["feature"]:
                    if obj_func not in area_result_dict[tar_bin][candidate_bin]:
                        area_result_dict[tar_bin][candidate_bin].append(obj_func)
    
    ground_truth = json.load(open(ground_truth_path, "r"))
    ground_truth_dict = {}
    for gt_item in ground_truth:
        tar_bin = gt_item.split("___")[0]
        cdd_bin = gt_item.split("___")[1]
        # if tar_bin in area_result_dict and cdd_bin in area_result_dict[tar_bin]:
        if tar_bin not in ground_truth_dict:
            ground_truth_dict[tar_bin] = {}
        if cdd_bin not in ground_truth_dict[tar_bin]:
            ground_truth_dict[tar_bin][cdd_bin] = []
        
        for reuse_func in ground_truth[gt_item]:
            # if reuse_func not in ground_truth_dict[tar_bin]:
            #     ground_truth_dict[tar_bin].append(reuse_func)
            if reuse_func not in ground_truth_dict[tar_bin][cdd_bin]:
                ground_truth_dict[tar_bin][cdd_bin].append(reuse_func)
        # if cdd_bin in area_result_dict and tar_bin in area_result_dict[cdd_bin]:
        if cdd_bin not in ground_truth_dict:
            ground_truth_dict[cdd_bin] = {}
        if tar_bin not in ground_truth_dict[cdd_bin]:
            ground_truth_dict[cdd_bin][tar_bin] = []
        
        for reuse_func in ground_truth[gt_item]:
            # if reuse_func not in ground_truth_dict[tar_bin]:
            #     ground_truth_dict[cdd_bin].append(reuse_func)
            if reuse_func not in ground_truth_dict[cdd_bin][tar_bin]:
                ground_truth_dict[cdd_bin][tar_bin].append(reuse_func)
        
    all_score_dict = {}
    all_pre = 0
    all_recall = 0
    all_F1 = 0
    i = 0
    for tar_bin in area_result_dict:
        for cdd_bin in area_result_dict[tar_bin]:
            # if tar_bin=="bzip2" and cdd_bin== "lzbench":
            if tar_bin.split("_")[0] in ground_truth_dict and cdd_bin in ground_truth_dict[tar_bin.split("_")[0]]:
                i += 1
                all_score_dict[tar_bin] = {}
                tar_fcg = nx.read_gpickle(os.path.join(tar_fcg_path, tar_bin+"_fcg.pkl"))
                tar_nodes = tar_fcg.nodes()
                tar_reuse_funcs = []
                for cdd_point in area_result_dict[tar_bin][cdd_bin]:
                    if cdd_point in tar_nodes:
                        tar_reuse_funcs.append(cdd_point)
                        # if cdd_point in tar_nodes:
                        desc_funclist = nx.descendants(tar_fcg, cdd_point)
                        for desc_func in desc_funclist:
                            if desc_func in tar_nodes and desc_func not in tar_reuse_funcs:
                                tar_reuse_funcs.append(desc_func)
                tar_ground_truth_reuse_func = []
                for gt_point in ground_truth_dict[tar_bin.split("_")[0]][cdd_bin]:
                    if gt_point in tar_nodes:
                        tar_ground_truth_reuse_func.append(gt_point)
                        desc_funclist = nx.descendants(tar_fcg, gt_point)
                        for desc_func in desc_funclist:
                            if desc_func in tar_nodes and desc_func not in tar_ground_truth_reuse_func:
                                tar_ground_truth_reuse_func.append(desc_func)   
                
                true_positive = list(set(tar_reuse_funcs).intersection(set(tar_ground_truth_reuse_func)))
                false_positive = list(set(tar_reuse_funcs) - set(tar_ground_truth_reuse_func))
                false_negative = list(set(tar_ground_truth_reuse_func) - set(tar_reuse_funcs))
                try:
                    Precision = len(true_positive) / len(set(tar_reuse_funcs))
                    Recall = len(true_positive) / len(set(tar_ground_truth_reuse_func))
                except:
                    Precision = Recall = 0
                if Precision == 0 and Recall == 0:
                    F1_score = 0
                else:
                    F1_score = 2*Precision*Recall/(Precision+Recall)
                all_pre += Precision
                all_recall += Recall
                all_F1 += F1_score
                
                print(tar_bin + "___" + cdd_bin + " Precision: {}  Recall: {}  F1: {}".format(Precision, Recall, F1_score) )
                json.dump(true_positive, open(os.path.join(save_path, tar_bin+"___"+cdd_bin+"_true_positive.json"), "w"))
                json.dump(false_positive, open(os.path.join(save_path, tar_bin+"___"+cdd_bin+"_false_positive.json"), "w"))
                json.dump(false_negative, open(os.path.join(save_path, tar_bin+"___"+cdd_bin+"_false_negative.json"), "w"))
                
                all_score_dict[tar_bin]["Precision"] = Precision
                all_score_dict[tar_bin]["Recall"] = Recall
                all_score_dict[tar_bin]["F1_score"] = F1_score
        
    if i!=0:
        json.dump(all_score_dict, open(os.path.join(save_path, "all_result.json"), "w"))
        print("All Precision: {}  Recall: {}  F1: {}".format(all_pre/i, all_recall/i, all_F1/i) )



def main(adjust_area_result_path, save_path, ground_truth_path, tar_fcg_path, cdd_fcg_path):
    # base_path = "D:\\reuse_detection\\area_dataset\\"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    result_path = os.path.join(save_path, "libAE_align_results.json")
    ground_path = os.path.join(save_path, "ground_results.pkl")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            dra = json.load(f)
        pass
    else:
        dra = libae_area_eval(results_path=adjust_area_result_path, save_path=result_path, align_thre=0)
    if os.path.exists(ground_path):
        with open(ground_path, "rb") as f:
            ground_results = pickle.load(f)
    else:
        ground_results = get_ground_truth(list(dra.keys()), tar_fcg_path=tar_fcg_path, cdd_fcg_path = cdd_fcg_path, ground_truth=ground_truth_path, save_path=ground_path)
    print("---------------libAE---------------")
    area_eval(dra, ground_results=ground_results, save_path=save_path + "libAE_metrics.json")
    # print("---------------libDB---------------")
    # libdb_area_eval(result_path=libdb_result_path, ground_results=ground_results, save_path=save_path + "libDB_results.json")
    # print("---------------Gemini---------------")
    # gemini_area_aval(result_path=gemini_result_path, ground_results=ground_results, save_path=save_path + "Gemini_results.json")


if __name__ == '__main__':
    main(os.path.join(DATA_PATH, "9_area_adjustment_result/reuse_area_7_adjust"), 
         os.path.join(DATA_PATH, "10_reuse_area_result"), 
         os.path.join(DATA_PATH, "libae_ground_truth.json"),
         os.path.join(DATA_PATH, "2_target/fcg"), 
         os.path.join(DATA_PATH, "3_candidate/fcg") )
