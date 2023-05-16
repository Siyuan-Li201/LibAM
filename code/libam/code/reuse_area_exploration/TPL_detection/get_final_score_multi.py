import os, json, sys
from tqdm import tqdm
from multiprocessing import Process
sys.path.append(".")
from settings import DATA_PATH

def RARM_score_v2(alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, alignment_max, max_fcg):
    if alignment_num_score == 1:
        alignment_num_score_deal = 0.9
    elif alignment_num_score == 2:
        alignment_num_score_deal = 0.95
    elif alignment_num_score >= 3:
        alignment_num_score_deal = 1.0
        
    # if alignment_num_score > 3:
    #     alignment_num_score = 3
    # alignment_num_score_deal = alignment_num_score/alignment_max
    node_fcg_scale_score_deal = node_fcg_scale_score/max_fcg
    final_score = alignment_num_score_deal * node_gnn_score #* node_fcg_scale_diff_score# * node_fcg_scale_score_deal

    return final_score




def RARM_score(alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, align_rate):#, alignment_max, max_fcg):
    # if alignment_num_score > 3:
    #     alignment_num_score = 3
    # alignment_num_score_deal = alignment_num_score/alignment_max
    # node_fcg_scale_score_deal = node_fcg_scale_score/max_fcg
    align_rate_score = 0.3 * align_rate + 0.7
    final_score = node_gnn_score * align_rate_score# * node_fcg_scale_diff_score # *alignment_num_score_deal * # * node_fcg_scale_score_deal

    return final_score

def get_score_one_7(rein_file_dict_list, score_file_path_dict, opath):
    for rein_file_dict_item in tqdm(rein_file_dict_list):
        if rein_file_dict_item in score_file_path_dict:
            target_item = rein_file_dict_item
            # if target_item == "bzip2_arm_O1":
            score_file_list = score_file_path_dict[rein_file_dict_item]
            #获取最大alignment规模
            alignment_max = 0
            for score_file in score_file_list:
                alignment_num = int(score_file.split("_")[-1].split(".")[0])
                if alignment_num > alignment_max:
                    alignment_max = alignment_num
            if alignment_max > 2:
                alignment_max = 2
            if alignment_max >= 1:
                target_reuse_lib_dict = []
                target_reuse_area_dict = {}
                # 获取最大fcg规模
                max_fcg = 0
                score_file_dict_dict = {}
                for score_file in tqdm(score_file_list):
                    # target_name = os.path.basename(score_file).split("----")[0]
                    candidate_name = os.path.basename(score_file).split("----")[1].split("_feature_result")[0]
                    file_alignment_max = int(score_file.split("_")[-1].split(".")[0])
                    if file_alignment_max > 0:
                        try:
                            score_file_dict = json.load(open(score_file, "r"))
                        except:
                            print("error: " + score_file)
                        score_file_dict_dict[score_file] = score_file_dict
                        for node_pair_str in score_file_dict:
                            node_fcg_scale_pair = score_file_dict[node_pair_str]["fcg_scale"]
                            node_max_fcg = max(node_fcg_scale_pair[0], node_fcg_scale_pair[1])
                            if max_fcg < node_max_fcg:
                                max_fcg = node_max_fcg
                if max_fcg > 3:
                    max_fcg = 3
                for score_file in score_file_dict_dict:
                    # target_name = os.path.basename(score_file).split("----")[0]
                    candidate_name = os.path.basename(score_file).split("----")[1].split("_feature_result")[0]
                    file_alignment_max = int(score_file.split("_")[-1].split(".")[0])
                    if file_alignment_max > 0:
                        score_file_dict = score_file_dict_dict[score_file]
                        
                        for node_pair_str in score_file_dict:
                            node_alignment_num_score = score_file_dict[node_pair_str]["alignment_num"]
                            node_fcg_scale_pair = score_file_dict[node_pair_str]["fcg_scale"]
                            node_gnn_score = float(score_file_dict[node_pair_str]["gnn_score"])
                            raw_final_score = float(score_file_dict[node_pair_str]["final_score"])
                            node_fcg_scale_score = (node_fcg_scale_pair[0] + node_fcg_scale_pair[1])/2
                            node_fcg_scale_diff_score = 0.3 * min(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) / max(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) + 0.7
                            
                            if node_alignment_num_score > 0 and node_fcg_scale_pair[0] >=3 and node_fcg_scale_pair[1] >= 3:
                                final_score =  RARM_score(node_alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, alignment_max, max_fcg)
                                score_file_dict[node_pair_str]["final_score"] = final_score
                                if final_score >= 0.7:
                                    if candidate_name not in target_reuse_lib_dict:
                                        target_reuse_lib_dict.append(candidate_name)
                                    if candidate_name not in target_reuse_area_dict:
                                        target_reuse_area_dict[candidate_name] = {}
                                    if node_pair_str not in target_reuse_area_dict[candidate_name]:
                                        target_reuse_area_dict[candidate_name][node_pair_str] = []
                                    target_reuse_area_dict[candidate_name][node_pair_str].append(score_file_dict[node_pair_str])
                                    # print("final_score: {}".format(final_score))
                                    # print("raw_final_score: {}".format(raw_final_score))
                                # elif node_alignment_num_score >=3:
                                    # print("final_score: {}".format(final_score))# 成功通过gnn矫正对齐结果
                                    # print("raw_final_score: {}".format(raw_final_score))
                json.dump(target_reuse_lib_dict, open(os.path.join(opath,"reuse_result_7" , target_item+"_reuse_result.json"), "w"))
                json.dump(target_reuse_area_dict, open(os.path.join(opath, "reuse_area_7", target_item+"_reuse_area.json"), "w"))

def get_score_7(result_path, opath):
    if not os.path.exists(os.path.join(opath, "reuse_result_7")):
        os.makedirs(os.path.join(opath, "reuse_result_7"))
    if not os.path.exists(os.path.join(opath, "reuse_area_7")):
        os.makedirs(os.path.join(opath, "reuse_area_7"))
    rein_file_dict = {}
    rein_file = list(os.listdir(result_path))
    for f in rein_file:
        target_binary_name = f.split("----")[0]
        if target_binary_name not in rein_file_dict:
            rein_file_dict[target_binary_name] = []
        if str(os.path.join(result_path, f)) not in rein_file_dict[target_binary_name]:
            rein_file_dict[target_binary_name].append(str(os.path.join(result_path, f)))
    print("target binaries num: {}".format(len(rein_file_dict)))
    
    
    rein_file_dict_list = list(rein_file_dict.keys())
    
    p_list = []
    Process_num = 35
    for i in range(Process_num):
        p = Process(target=get_score_one_7, args=(rein_file_dict_list[int((i/Process_num)*len(rein_file_dict_list)):int(((i+1)/Process_num)*len(rein_file_dict_list))], rein_file_dict, opath))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
    for p in tqdm(p_list):
        p.join()
    
    # for target_item in tqdm(rein_file_dict):
    #     # if target_item == "precomp_x64_O2":
    #     get_score_one()
            


def get_score_one_8(rein_file_dict_list, score_file_path_dict, opath):
    for rein_file_dict_item in tqdm(rein_file_dict_list):
        if rein_file_dict_item in score_file_path_dict:
            target_item = rein_file_dict_item
            # if target_item == "bzip2_x64_O0":
            score_file_list = score_file_path_dict[rein_file_dict_item]
            # #获取最大alignment规模
            # alignment_max = 0
            # for score_file in score_file_list:
            #     alignment_num = int(score_file.split("_")[-1].split(".")[0])
            #     if alignment_num > alignment_max:
            #         alignment_max = alignment_num
            # if alignment_max > 3:
            #     alignment_max = 3
            # if alignment_max >= 1:
            target_reuse_lib_dict = []
            target_reuse_area_dict = {}
            #     # 获取最大fcg规模
            #     max_fcg = 0
            score_file_dict_dict = {}
            #     for score_file in score_file_list:
            #         # target_name = os.path.basename(score_file).split("----")[0]
            #         candidate_name = os.path.basename(score_file).split("----")[1].split("_feature_result")[0]
            #         file_alignment_max = int(score_file.split("_")[-1].split(".")[0])
            #         if file_alignment_max > 0:
            #             try:
            #                 score_file_dict = json.load(open(score_file, "r"))
            #             except:
            #                 print("error: " + score_file)
            #             score_file_dict_dict[score_file] = score_file_dict
            #             for node_pair_str in score_file_dict:
            #                 node_fcg_scale_pair = score_file_dict[node_pair_str]["fcg_scale"]
            #                 node_max_fcg = max(node_fcg_scale_pair[0], node_fcg_scale_pair[1])
            #                 if max_fcg < node_max_fcg:
            #                     max_fcg = node_max_fcg
            #     if max_fcg > 3:
            #         max_fcg = 3
            for score_file in score_file_list:
                # target_name = os.path.basename(score_file).split("----")[0]
                candidate_name = os.path.basename(score_file).split("----")[1].split("_feature_result")[0]
                
                # sim_funcs_list = json.load(open(os.path.join(sim_funcs_path, target_item+"----"+candidate_name+"_sim_funcs.json"), "r"))
                
                
                file_alignment_max = int(score_file.split("_")[-1].split(".")[0])
                try:
                    score_file_dict = json.load(open(score_file, "r"))
                except:
                    print("error: " + score_file)
                score_file_dict_dict[score_file] = score_file_dict
                if file_alignment_max >= 2:
                    score_file_dict = score_file_dict_dict[score_file]
                    
                    for node_pair_str in score_file_dict:
                        node_alignment_num_score = score_file_dict[node_pair_str]["alignment_num"]
                        node_fcg_scale_pair = score_file_dict[node_pair_str]["fcg_scale"]
                        node_gnn_score = float(score_file_dict[node_pair_str]["gnn_score"])
                        raw_final_score = float(score_file_dict[node_pair_str]["final_score"])
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
                                # print("final_score: {}".format(final_score))
                                # print("raw_final_score: {}".format(raw_final_score))
                            # elif node_alignment_num_score >=3:
                                # print("final_score: {}".format(final_score))# 成功通过gnn矫正对齐结果
                                # print("raw_final_score: {}".format(raw_final_score))
            json.dump(target_reuse_lib_dict, open(os.path.join(opath,"reuse_result_8" , target_item+"_reuse_result.json"), "w"))
            json.dump(target_reuse_area_dict, open(os.path.join(opath, "reuse_area_8", target_item+"_reuse_area.json"), "w"))

def get_score_8(result_path, opath):
    if not os.path.exists(os.path.join(opath, "reuse_result_8")):
        os.makedirs(os.path.join(opath, "reuse_result_8"))
    if not os.path.exists(os.path.join(opath, "reuse_area_8")):
        os.makedirs(os.path.join(opath, "reuse_area_8"))
    rein_file_dict = {}
    rein_file = list(os.listdir(result_path))
    for f in rein_file:
        # if "cknit----" in f:
        target_binary_name = f.split("----")[0]
        if target_binary_name not in rein_file_dict:
            rein_file_dict[target_binary_name] = []
        if str(os.path.join(result_path, f)) not in rein_file_dict[target_binary_name]:
            rein_file_dict[target_binary_name].append(str(os.path.join(result_path, f)))
    print("target binaries num: {}".format(len(rein_file_dict)))
    
    
    rein_file_dict_list = list(rein_file_dict.keys())
    
    p_list = []
    Process_num = 50
    for i in range(Process_num):
        p = Process(target=get_score_one_8, args=(rein_file_dict_list[int((i/Process_num)*len(rein_file_dict_list)):int(((i+1)/Process_num)*len(rein_file_dict_list))], rein_file_dict, opath))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
    for p in tqdm(p_list):
        p.join()




if __name__ == "__main__":
    # result_path = "/data/lisiyuan/libAE/TPL_detection_result/1109_5_libae_paper_top50_gnn_analog_0.001"
    # opath = "/data/lisiyuan/libAE/TPL_detection_result/1109_5_libae_paper_top50_gnn_analog_0.001_gnn_result"
    # #get_achor_align_graph(fcg_path, func_path, taint_fcg_path)
    # get_reuse_area_multi(fcg_path, func_path, feature_save_path, time_path)
    # get_libdb_reuse_multi(fcg_path, func_path, result_path)
    
    # fcg_path = "F:\\mypaper\\data\\openssl_result\\openssl_fcg"#"isrd_default_fcg"
    # func_path = "F:\\mypaper\\data\\openssl_result\\reuse_result_openssl"#"reuse_result_deal_pruning"
    # result_path = "F:\\mypaper\\data\\version_result.json"#"tainted_fcg"
    
    # get_libdb_version(fcg_path, func_path, result_path)
    
    
    get_score_7(os.path.join(DATA_PATH, "7_gnn_result/after_gnn_result"), os.path.join(DATA_PATH, "8_tpl_result"))
    get_score_8(os.path.join(DATA_PATH, "7_gnn_result/after_gnn_result"), os.path.join(DATA_PATH, "8_tpl_result"))
            
            
                
    print("all done")
            