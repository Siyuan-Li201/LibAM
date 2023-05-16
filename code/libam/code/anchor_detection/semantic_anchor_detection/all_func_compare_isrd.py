import json, os, sys, pickle, tqdm, time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Process
sys.path.append(".")
from settings import DATA_PATH
from annoy import AnnoyIndex

def get_json_data(filename):
    with open(filename,"r") as ff:
        data = json.load(ff)

        return data





def cosine_simi(src_embeddings, tgt_embeddings):
    """
    如果src只有一个也要reshape成二维计算
    """
    data = cosine_similarity(src_embeddings, tgt_embeddings)
    return (data + 1) / 2



def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def size_match(object_size, candidate_size):
    if object_size <  candidate_size*2 and candidate_size <  object_size*2:
        if object_size - candidate_size < 20 and candidate_size - object_size < 20:
            return True
    return False




def get_func_embeddings(object_path):
    
    detect_binary_func_vec = {}
    
    detect_vec = json.load(open(object_path, "r"))
    for detect_item in detect_vec:
        detect_name = "|||".join(detect_item.split("|||")[:-1])
        func_name = detect_item.split("|||")[-1]
        if detect_name not in detect_binary_func_vec:
            detect_binary_func_vec[detect_name] = {}
        if func_name not in detect_binary_func_vec[detect_name]:
            detect_binary_func_vec[detect_name][func_name] = np.array(detect_vec[detect_item]).reshape(-1,64)
    
    return detect_binary_func_vec



def get_matrix(func_vec_dict):
    num = 0
    for func_item in func_vec_dict:
        num  += 1
        if num == 1:
            matrix = np.array(func_vec_dict[func_item])
        else:
            matrix = np.concatenate([matrix, func_vec_dict[func_item]])
    
    return matrix


            

def comapre_func(detect_binary_func_vec_list, detect_binary_func_vec, candidate_binary_func_vec, score_opath, score_opath2, time_opath):
    
    time_dict = {}
    for detect_binary in tqdm.tqdm(detect_binary_func_vec):
        if detect_binary in detect_binary_func_vec_list and not os.path.exists(os.path.join(score_opath, detect_binary+"_reuse_func_dict.json")):
            # if "bzip2" in detect_binary:
            start = time.time()
            score_dict = {}
            deal_score_dict = {}
            raw_score_dict = {}
            detect_func_vec_dict =  detect_binary_func_vec[detect_binary]
            
            detect_matrix = get_matrix(detect_func_vec_dict)
            
            for candidate_binary in tqdm.tqdm(candidate_binary_func_vec):
                if candidate_binary not in detect_binary:
                    candidate_func_vec_dict =  candidate_binary_func_vec[candidate_binary]
                    
                    candidate_matrix = get_matrix(candidate_func_vec_dict)
                    
                    score_matrix = cosine_simi(detect_matrix, candidate_matrix)
                    
                    i = 0
                    for dict1_item in detect_func_vec_dict:
                        j = 0
                        for dict2_item in candidate_func_vec_dict:
                            if score_matrix[i][j] > 0.72:
                                # score_dict[detect_binary+"----"+dict1_item+"||||"+candidate_binary+"----"+dict2_item] = score_matrix[i][j]
                                if dict1_item not in score_dict:
                                    score_dict[dict1_item] = {}
                                score_dict[dict1_item][candidate_binary+"----"+dict2_item] = score_matrix[i][j]
                                raw_score_dict[detect_binary+"----"+dict1_item+"||||"+candidate_binary+"----"+dict2_item] = score_matrix[i][j]
                            j += 1
                        i += 1
            
            for detect_func in score_dict:
                object_cdd_func_list = sorted(score_dict[detect_func].items(),  key=lambda d: d[1], reverse=True)[:100]
                for object_cdd_func_item in object_cdd_func_list:
                    deal_score_dict[detect_binary+"----"+detect_func+"||||"+object_cdd_func_item[0]] = object_cdd_func_item[1]
            end = time.time()
            run_time = end - start
            time_dict[detect_binary] = run_time
            json.dump(raw_score_dict, open(os.path.join(score_opath, detect_binary+"_reuse_func_dict.json"), "w"))
            json.dump(deal_score_dict, open(os.path.join(score_opath2, detect_binary+"_reuse_func_dict.json"), "w"))
            json.dump(time_dict, open(os.path.join(time_opath, detect_binary+"isrd_triple_loss_time.json"), "w"))    
                

def func_compare_annoy(object_path, candidate_path, score_opath, score_opath2, time_opath, embed_path):
    
    black_func_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    
    
    detect_binary_func_vec = get_func_embeddings(object_path)
    candidate_binary_func_vec = get_func_embeddings(candidate_path)
    
    detect_binary_func_vec_list = list(detect_binary_func_vec.keys())

    if False == os.path.exists(score_opath):
        os.makedirs(score_opath)
    if False == os.path.exists(score_opath2):
        os.makedirs(score_opath2)
    if False == os.path.exists(time_opath):
        os.makedirs(time_opath)
    if False == os.path.exists(embed_path):
        os.makedirs(embed_path)
    time_dict = {}
    for detect_binary in tqdm.tqdm(detect_binary_func_vec):
        if detect_binary in detect_binary_func_vec_list and not os.path.exists(os.path.join(score_opath, detect_binary+"_reuse_func_dict.json")):
            # if "bzip2" in detect_binary:
            start = time.time()
            score_dict = {}
            deal_score_dict = {}
            raw_score_dict = {}
            detect_func_vec_dict =  detect_binary_func_vec[detect_binary]
            
            # detect_matrix = get_matrix(detect_func_vec_dict)
            
            for candidate_binary in tqdm.tqdm(candidate_binary_func_vec):
                if candidate_binary not in detect_binary:
                    
                    u = AnnoyIndex(64, 'angular')
                    if os.path.exists(os.path.join(embed_path, candidate_binary+".json")):
                        u.load(os.path.join(embed_path, candidate_binary+".ann")) # super fast, will just mmap the file
                        candidate_id_func_dict = json.load(open(os.path.join(embed_path, candidate_binary+".json"), "r"))
                    else:
                        
                        candidate_func_vec_dict =  candidate_binary_func_vec[candidate_binary]
                        
                        # tar_func_num = len(detect_func_vec_dict)
                        # cdd_func_num = len(candidate_func_vec_dict)
                        
                        candidate_id_func_dict = {}
                        # cdd_id_func_dict = []
                        i = 0
                        for func_name in candidate_func_vec_dict:
                            if func_name not in black_func_list:
                                candidate_id_func_dict[i] = func_name
                                u.add_item(i, candidate_func_vec_dict[func_name].tolist()[0])
                                i += 1
                        
                        u.build(10) # 10 trees
    
                        u.save(os.path.join(embed_path, candidate_binary+".ann"))
                        json.dump(candidate_id_func_dict, open(os.path.join(embed_path, candidate_binary+".json"), "w"))
                    
                    for target_funcname in detect_func_vec_dict:
                        query_result, distance_result = u.get_nns_by_vector(detect_func_vec_dict[target_funcname].tolist()[0], 100, include_distances=True)
                    
                        for i in range(len(query_result)):
                            if distance_result[i] < 0.7483:
                                if target_funcname not in score_dict:
                                    score_dict[target_funcname] = {}
                                score_dict[target_funcname][candidate_binary+"----"+candidate_id_func_dict[query_result[i]]] = distance_result[i]
                        
            for detect_func in score_dict:
                object_cdd_func_list = sorted(score_dict[detect_func].items(),  key=lambda d: d[1], reverse=True)[:100]
                for object_cdd_func_item in object_cdd_func_list:
                    deal_score_dict[detect_binary+"----"+detect_func+"||||"+object_cdd_func_item[0]] = object_cdd_func_item[1]
            end = time.time()
            run_time = end - start
            time_dict[detect_binary] = run_time
            # json.dump(raw_score_dict, open(os.path.join(score_opath, detect_binary+"_reuse_func_dict.json"), "w"))
            json.dump(deal_score_dict, open(os.path.join(score_opath2, detect_binary+"_reuse_func_dict.json"), "w"))
            json.dump(time_dict, open(os.path.join(time_opath, detect_binary+"isrd_triple_loss_time.json"), "w"))    



def func_compare_annoy_fast_one(detect_binary_func_vec_list, detect_binary_func_vec, candidate_binary_func_vec, score_opath, score_opath2, time_opath, embed_path):
    black_func_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    for detect_binary in tqdm.tqdm(detect_binary_func_vec_list, desc="all target bianry process..."):
        if detect_binary in detect_binary_func_vec and not os.path.exists(os.path.join(time_opath, detect_binary+"isrd_triple_loss_time.json")):
            # 
        # if "ASUS__AC68U_30043762048__cp" in detect_binary:
            time_dict = {}
            start = time.time()
            score_dict = {}
            deal_score_dict = {}
            raw_score_dict = {}
            detect_func_vec_dict =  detect_binary_func_vec[detect_binary]
    
            t = AnnoyIndex(64, 'angular')
            if os.path.exists(os.path.join(embed_path, "all_candidate_bin.json")):
                t.load(os.path.join(embed_path, "all_candidate.ann"))
                all_candidate_id_func_dict = json.load(open(os.path.join(embed_path, "all_candidate_func.json"), "r"))
                all_candidate_id_bin_dict = json.load(open(os.path.join(embed_path, "all_candidate_bin.json"), "r"))
            else:
                all_candidate_id_func_dict = {}
                all_candidate_id_bin_dict = {}
                i = 0
                for candidate_binary in tqdm.tqdm(candidate_binary_func_vec, desc="get all candidate vector lib process..."):
                    candidate_func_vec_dict =  candidate_binary_func_vec[candidate_binary]
                    for func_name in candidate_func_vec_dict:
                        if func_name not in black_func_list:
                            all_candidate_id_func_dict[str(i)] = func_name
                            all_candidate_id_bin_dict[str(i)] = candidate_binary
                            t.add_item(i, candidate_func_vec_dict[func_name].tolist()[0])
                            i += 1
                
                t.build(100) # 10 trees

                t.save(os.path.join(embed_path, "all_candidate.ann"))
                json.dump(all_candidate_id_func_dict, open(os.path.join(embed_path, "all_candidate_func.json"), "w"))
                json.dump(all_candidate_id_bin_dict, open(os.path.join(embed_path, "all_candidate_bin.json"), "w"))
                
            candidate_bin_dict = {}
            for target_funcname in tqdm.tqdm(detect_func_vec_dict, desc="detect top200 cdd lib process..."):
                if target_funcname not in black_func_list:
                    query_result, distance_result = t.get_nns_by_vector(detect_func_vec_dict[target_funcname].tolist()[0], 100, include_distances=True)
                    for i in range(len(query_result)):
                        if distance_result[i] < 1.058:#0.7483:
                            if target_funcname not in score_dict:
                                score_dict[target_funcname] = {}
                            score_dict[target_funcname][all_candidate_id_bin_dict[str(query_result[i])]+"----"+all_candidate_id_func_dict[str(query_result[i])]] = distance_result[i]
                            raw_score_dict[detect_binary+"----"+target_funcname+"||||"+all_candidate_id_bin_dict[str(query_result[i])]+"----"+all_candidate_id_func_dict[str(query_result[i])]] = distance_result[i]
                        else:
                            break
                        
            for detect_func in score_dict:
                object_cdd_func_list = sorted(score_dict[detect_func].items(),  key=lambda d: d[1], reverse=False)[:100]
                for object_cdd_func_item in object_cdd_func_list:
                    deal_score_dict[detect_binary+"----"+detect_func+"||||"+object_cdd_func_item[0]] = object_cdd_func_item[1]
            end = time.time()
            run_time = end - start
            time_dict[detect_binary] = run_time
            json.dump(raw_score_dict, open(os.path.join(score_opath, detect_binary+"_reuse_func_dict.json"), "w"))
            json.dump(deal_score_dict, open(os.path.join(score_opath2, detect_binary+"_reuse_func_dict.json"), "w"))
            json.dump(time_dict, open(os.path.join(time_opath, detect_binary+"isrd_triple_loss_time.json"), "w"))    


# def func_compare_annoy_fast_one(detect_binary_func_vec_list, detect_binary_func_vec, candidate_binary_func_vec, score_opath, score_opath2, time_opath, embed_path):
#     black_func_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
#     for detect_binary in tqdm.tqdm(detect_binary_func_vec_list, desc="all target bianry process..."):
#         if detect_binary in detect_binary_func_vec and not os.path.exists(os.path.join(time_opath, detect_binary+"isrd_triple_loss_time.json")):
#             # 
#         # if "ASUS__AC68U_30043762048__cp" in detect_binary:
#             time_dict = {}
#             start = time.time()
#             score_dict = {}
#             deal_score_dict = {}
#             raw_score_dict = {}
#             detect_func_vec_dict =  detect_binary_func_vec[detect_binary]
    
#             t = AnnoyIndex(64, 'angular')
#             if os.path.exists(os.path.join(embed_path, "all_candidate_bin.json")):
#                 t.load(os.path.join(embed_path, "all_candidate.ann"))
#                 all_candidate_id_func_dict = json.load(open(os.path.join(embed_path, "all_candidate_func.json"), "r"))
#                 all_candidate_id_bin_dict = json.load(open(os.path.join(embed_path, "all_candidate_bin.json"), "r"))
#             else:
#                 all_candidate_id_func_dict = {}
#                 all_candidate_id_bin_dict = {}
#                 i = 0
#                 for candidate_binary in tqdm.tqdm(candidate_binary_func_vec, desc="get all candidate vector lib process..."):
#                     candidate_func_vec_dict =  candidate_binary_func_vec[candidate_binary]
#                     for func_name in candidate_func_vec_dict:
#                         if func_name not in black_func_list:
#                             all_candidate_id_func_dict[str(i)] = func_name
#                             all_candidate_id_bin_dict[str(i)] = candidate_binary
#                             t.add_item(i, candidate_func_vec_dict[func_name].tolist()[0])
#                             i += 1
                
#                 t.build(100) # 10 trees

#                 t.save(os.path.join(embed_path, "all_candidate.ann"))
#                 json.dump(all_candidate_id_func_dict, open(os.path.join(embed_path, "all_candidate_func.json"), "w"))
#                 json.dump(all_candidate_id_bin_dict, open(os.path.join(embed_path, "all_candidate_bin.json"), "w"))
                
#             candidate_bin_dict = {}
#             for target_funcname in tqdm.tqdm(detect_func_vec_dict, desc="detect top200 cdd lib process..."):
#                 if target_funcname not in black_func_list:
#                     query_result, distance_result = t.get_nns_by_vector(detect_func_vec_dict[target_funcname].tolist()[0], 1000, include_distances=True)
#                     lib_set = set()
#                     for i in range(len(query_result)):
#                         if distance_result[i] < 0.7483:
#                             if all_candidate_id_bin_dict[str(query_result[i])] not in candidate_bin_dict:
#                                 candidate_bin_dict[all_candidate_id_bin_dict[str(query_result[i])]] = 0
#                             if all_candidate_id_bin_dict[str(query_result[i])] not in lib_set:
#                                 lib_set.add(all_candidate_id_bin_dict[str(query_result[i])])
#                                 candidate_bin_dict[all_candidate_id_bin_dict[str(query_result[i])]] += 1
#                         else:
#                             break
#                         # score_dict[target_funcname][candidate_binary+"----"+candidate_id_func_dict[query_result[i]]] = distance_result[i]

#             candidate_bin_list = sorted(candidate_bin_dict.items(),  key=lambda d: d[1], reverse=True)[:200]  #isrd这里为5w
            
#             for target_funcname in tqdm.tqdm(detect_func_vec_dict, desc="one target bianry process..."):
#                 # if target_funcname == "sub_F848":
#                 #     print("warning")
#                 if target_funcname not in black_func_list:
#                     for candidate_binary_tuple in candidate_bin_list:
#                         candidate_binary = candidate_binary_tuple[0]
#                         if candidate_binary not in detect_binary:
#                         # if candidate_binary == "cp":
#                             u = AnnoyIndex(64, 'angular')
#                             if os.path.exists(os.path.join(embed_path, candidate_binary+".json")):
#                                 u.load(os.path.join(embed_path, candidate_binary+".ann")) # super fast, will just mmap the file
#                                 candidate_id_func_dict = json.load(open(os.path.join(embed_path, candidate_binary+".json"), "r"))
#                             else:
                                
#                                 candidate_func_vec_dict =  candidate_binary_func_vec[candidate_binary]
                                
#                                 # tar_func_num = len(detect_func_vec_dict)
#                                 # cdd_func_num = len(candidate_func_vec_dict)
                                
#                                 candidate_id_func_dict = {}
#                                 # cdd_id_func_dict = []
#                                 i = 0
#                                 for func_name in candidate_func_vec_dict:
#                                     if func_name not in  black_func_list:
#                                         candidate_id_func_dict[str(i)] = func_name
#                                         u.add_item(i, candidate_func_vec_dict[func_name].tolist()[0])
#                                         i += 1
                                
#                                 u.build(100) # 10 trees

#                                 u.save(os.path.join(embed_path, candidate_binary+".ann"))
#                                 json.dump(candidate_id_func_dict, open(os.path.join(embed_path, candidate_binary+".json"), "w"))
                                    
#                             query_result, distance_result = u.get_nns_by_vector(detect_func_vec_dict[target_funcname].tolist()[0], 10, include_distances=True)
                        
#                             for i in range(len(query_result)):
#                                 if distance_result[i] < 0.7483:
#                                     if target_funcname not in score_dict:
#                                         score_dict[target_funcname] = {}
#                                     score_dict[target_funcname][candidate_binary+"----"+candidate_id_func_dict[str(query_result[i])]] = distance_result[i]
#                                     raw_score_dict[detect_binary+"----"+target_funcname+"||||"+candidate_binary+"----"+candidate_id_func_dict[str(query_result[i])]] = distance_result[i]
#                                 else:
#                                     break
                        
#             for detect_func in score_dict:
#                 object_cdd_func_list = sorted(score_dict[detect_func].items(),  key=lambda d: d[1], reverse=False)[:100]
#                 for object_cdd_func_item in object_cdd_func_list:
#                     deal_score_dict[detect_binary+"----"+detect_func+"||||"+object_cdd_func_item[0]] = object_cdd_func_item[1]
#             end = time.time()
#             run_time = end - start
#             time_dict[detect_binary] = run_time
#             json.dump(raw_score_dict, open(os.path.join(score_opath, detect_binary+"_reuse_func_dict.json"), "w"))
#             json.dump(deal_score_dict, open(os.path.join(score_opath2, detect_binary+"_reuse_func_dict.json"), "w"))
#             json.dump(time_dict, open(os.path.join(time_opath, detect_binary+"isrd_triple_loss_time.json"), "w"))    


def func_compare_annoy_fast_multi(object_path, candidate_path, score_opath, score_opath2, time_opath, embed_path):
    detect_binary_func_vec = get_func_embeddings(object_path)
    candidate_binary_func_vec = get_func_embeddings(candidate_path)
    
    
    
    detect_binary_func_vec_list = list(detect_binary_func_vec.keys())

    if False == os.path.exists(score_opath):
        os.makedirs(score_opath)
    if False == os.path.exists(score_opath2):
        os.makedirs(score_opath2)
    if False == os.path.exists(time_opath):
        os.makedirs(time_opath)
    if False == os.path.exists(embed_path):
        os.makedirs(embed_path)
    
    
    p_list = []
    Process_num = 35
    for i in range(Process_num):
        p = Process(target=func_compare_annoy_fast_one, args=(detect_binary_func_vec_list[int((i/Process_num)*len(detect_binary_func_vec_list)):int(((i+1)/Process_num)*len(detect_binary_func_vec_list))], detect_binary_func_vec, candidate_binary_func_vec, score_opath, score_opath2, time_opath, embed_path))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
        # time.sleep(15)
    for p in tqdm.tqdm(p_list):
        p.join()

    
    
    
def func_compare_annoy_fast(object_path, candidate_path, score_opath, score_opath2, time_opath, embed_path):
    detect_binary_func_vec = get_func_embeddings(object_path)
    candidate_binary_func_vec = get_func_embeddings(candidate_path)
    
    black_func_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    
    detect_binary_func_vec_list = list(detect_binary_func_vec.keys())

    if False == os.path.exists(score_opath):
        os.makedirs(score_opath)
    if False == os.path.exists(score_opath2):
        os.makedirs(score_opath2)
    if False == os.path.exists(time_opath):
        os.makedirs(time_opath)
    if False == os.path.exists(embed_path):
        os.makedirs(embed_path)
    time_dict = {}
    
    
    
    for detect_binary in tqdm.tqdm(detect_binary_func_vec, desc="all target bianry process..."):
        if detect_binary in detect_binary_func_vec_list and not os.path.exists(os.path.join(time_opath, detect_binary+"isrd_triple_loss_time.json")):
            # if "bzip2" in detect_binary:
            start = time.time()
            score_dict = {}
            deal_score_dict = {}
            # raw_score_dict = {}
            detect_func_vec_dict =  detect_binary_func_vec[detect_binary]
    
            t = AnnoyIndex(64, 'angular')
            if os.path.exists(os.path.join(embed_path, "all_candidate_bin.json")):
                t.load(os.path.join(embed_path, "all_candidate.ann"))
                all_candidate_id_func_dict = json.load(open(os.path.join(embed_path, "all_candidate_func.json"), "r"))
                all_candidate_id_bin_dict = json.load(open(os.path.join(embed_path, "all_candidate_bin.json"), "r"))
            else:
                all_candidate_id_func_dict = {}
                all_candidate_id_bin_dict = {}
                i = 0
                for candidate_binary in tqdm.tqdm(candidate_binary_func_vec, desc="get all candidate vector lib process..."):
                    candidate_func_vec_dict =  candidate_binary_func_vec[candidate_binary]
                    for func_name in candidate_func_vec_dict:
                        if func_name not in black_func_list:
                            all_candidate_id_func_dict[str(i)] = func_name
                            all_candidate_id_bin_dict[str(i)] = candidate_binary
                            t.add_item(i, candidate_func_vec_dict[func_name].tolist()[0])
                            i += 1
                
                t.build(10) # 10 trees

                t.save(os.path.join(embed_path, "all_candidate.ann"))
                json.dump(all_candidate_id_func_dict, open(os.path.join(embed_path, "all_candidate_func.json"), "w"))
                json.dump(all_candidate_id_bin_dict, open(os.path.join(embed_path, "all_candidate_bin.json"), "w"))
                
            candidate_bin_dict = {}
            for target_funcname in tqdm.tqdm(detect_func_vec_dict, desc="detect top200 cdd lib process..."):
                if target_funcname not in black_func_list:
                    query_result, distance_result = t.get_nns_by_vector(detect_func_vec_dict[target_funcname].tolist()[0], 1000, include_distances=True)
                
                    for i in range(len(query_result)):
                        if distance_result[i] < 0.7483:
                            if all_candidate_id_bin_dict[str(query_result[i])] not in candidate_bin_dict:
                                candidate_bin_dict[all_candidate_id_bin_dict[str(query_result[i])]] = 0
                            candidate_bin_dict[all_candidate_id_bin_dict[str(query_result[i])]] += 1
                        else:
                            break
                        # score_dict[target_funcname][candidate_binary+"----"+candidate_id_func_dict[query_result[i]]] = distance_result[i]

            candidate_bin_list = sorted(candidate_bin_dict.items(),  key=lambda d: d[1], reverse=True)[:200]  #isrd这里为5w
            
            for target_funcname in tqdm.tqdm(detect_func_vec_dict, desc="one target bianry process..."):
                if target_funcname not in black_func_list:
                    for candidate_binary_tuple in candidate_bin_list:
                        candidate_binary = candidate_binary_tuple[0]
                        if candidate_binary not in detect_binary:

                            u = AnnoyIndex(64, 'angular')
                            if os.path.exists(os.path.join(embed_path, candidate_binary+".json")):
                                u.load(os.path.join(embed_path, candidate_binary+".ann")) # super fast, will just mmap the file
                                candidate_id_func_dict = json.load(open(os.path.join(embed_path, candidate_binary+".json"), "r"))
                            else:
                                
                                candidate_func_vec_dict =  candidate_binary_func_vec[candidate_binary]
                                
                                # tar_func_num = len(detect_func_vec_dict)
                                # cdd_func_num = len(candidate_func_vec_dict)
                                
                                candidate_id_func_dict = {}
                                # cdd_id_func_dict = []
                                i = 0
                                for func_name in candidate_func_vec_dict:
                                    if func_name not in  black_func_list:
                                        candidate_id_func_dict[str(i)] = func_name
                                        u.add_item(i, candidate_func_vec_dict[func_name].tolist()[0])
                                        i += 1
                                
                                u.build(10) # 10 trees

                                u.save(os.path.join(embed_path, candidate_binary+".ann"))
                                json.dump(candidate_id_func_dict, open(os.path.join(embed_path, candidate_binary+".json"), "w"))
                                        
                            query_result, distance_result = u.get_nns_by_vector(detect_func_vec_dict[target_funcname].tolist()[0], 100, include_distances=True)
                        
                            for i in range(len(query_result)):
                                if distance_result[i] < 0.7483:
                                    if target_funcname not in score_dict:
                                        score_dict[target_funcname] = {}
                                    score_dict[target_funcname][candidate_binary+"----"+candidate_id_func_dict[str(query_result[i])]] = distance_result[i]
                                else:
                                    break
                            
            for detect_func in score_dict:
                object_cdd_func_list = sorted(score_dict[detect_func].items(),  key=lambda d: d[1], reverse=True)[:100]
                for object_cdd_func_item in object_cdd_func_list:
                    deal_score_dict[detect_binary+"----"+detect_func+"||||"+object_cdd_func_item[0]] = object_cdd_func_item[1]
            end = time.time()
            run_time = end - start
            time_dict[detect_binary] = run_time
            # json.dump(raw_score_dict, open(os.path.join(score_opath, detect_binary+"_reuse_func_dict.json"), "w"))
            json.dump(deal_score_dict, open(os.path.join(score_opath2, detect_binary+"_reuse_func_dict.json"), "w"))
            json.dump(time_dict, open(os.path.join(time_opath, detect_binary+"isrd_triple_loss_time.json"), "w"))    

# @click.command()
# @click.option('-o', '--object_path', 'object_path', default="/data/lisiyuan/libAE/paper_datasets/embeddings/isrd_target_in10_nn5_embeddings_torch_best.json", help='target function 1')
# @click.option('-c', '--candidate_path', 'candidate_path', default="/data/lisiyuan/libAE/cross_isrd_dataset/embeddings/isrd_candidates_in10_nn5_embeddings_torch_best.json", help='target function 2')
# @click.option('-s', '--sims', 'score_opath', default="/data/lisiyuan/code/new_code_20221015/DFcon_20221027/score/paper_dataset/isrd_triple_loss_score", help='model path')
# @click.option('-s', '--sims', 'score_opath2', default="/data/lisiyuan/code/new_code_20221015/DFcon_20221027/score/paper_dataset/isrd_triple_loss_score_top50", help='model path')
# @click.option('-s', '--sims', 'time_opath', default="/data/lisiyuan/code/new_code_20221015/DFcon_20221027/score/isrd/isrd_triple_loss_time", help='model path')
def cli(object_path, candidate_path, score_opath, score_opath2, time_opath):

# /data/lisiyuan/code/new_code_20221015/DFcon_20221027/score/isrd/isrd_triple_loss_score
    
    detect_binary_func_vec = get_func_embeddings(object_path)
    candidate_binary_func_vec = get_func_embeddings(candidate_path)
    
    detect_binary_func_vec_list = list(detect_binary_func_vec.keys())

    if False == os.path.exists(score_opath):
        os.makedirs(score_opath)
    if False == os.path.exists(score_opath2):
        os.makedirs(score_opath2)
    if False == os.path.exists(time_opath):
        os.makedirs(time_opath)

    p_list = []
    Process_num = 35
    for i in range(Process_num):
        p = Process(target=comapre_func, args=(detect_binary_func_vec_list[int((i/Process_num)*len(detect_binary_func_vec_list)):int(((i+1)/Process_num)*len(detect_binary_func_vec_list))], detect_binary_func_vec, candidate_binary_func_vec, score_opath, score_opath2, time_opath))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
    for p in tqdm.tqdm(p_list):
        p.join()



    # software_num = 0

    # #oo_list = ["bzip2_arm_O3", "bzip2_x86_O3", "bzip2"]
    # for object_software in object_funcs:
    #     software_num += 1
    #     if object_software in oo_list:
    #         print("")
    #         print("start compare "+object_software+":")
    #         func_num = 0
    #         for object_func in object_funcs[object_software]:
    #             func_num += 1
                
    #             object_candidate_score_dict = {}
    #             object_func_name = object_func
    #             object_func_embedding = object_funcs[object_software][object_func]
                
    #             cdd_software_num = 0
    #             for candidate_software in candidate_funcs:
    #                 if candidate_software != object_software:
    #                     cdd_software_num += 1
                        
    #                     cdd_func_num = 0
    #                     for candidate_func in candidate_funcs[candidate_software]:
    #                         cdd_func_num += 1
                            
    #                         candidate_func_name = candidate_func
    #                         candidate_func_embedding = candidate_funcs[candidate_software][candidate_func]
                            
    #                         cos_sim_score = cosine_simi(object_func_embedding, candidate_func_embedding).reshape(-1)[0]
                            
    #                         object_candidate_score_dict[object_software+"----"+object_func_name+"||||"+candidate_software+"----"+candidate_func_name] = cos_sim_score

                            
    #                     print("\r{}/{} candidate softwares, {}/{} object functions and {}/{} object software have been compared".format(cdd_software_num, len(candidate_funcs), func_num, len(object_funcs[object_software]), software_num, len(object_funcs)), end="")
    #             with open(os.path.join(sims_list_opath,object_software+"----"+object_func_name), "wb") as ff:
    #                 pickle.dump(object_candidate_score_dict, ff)


if __name__ == "__main__":
    cli(os.path.join(DATA_PATH, "4_embedding/target_embedding.json"), 
        os.path.join(DATA_PATH, "4_embedding/candidate_embedding.json"), 
        os.path.join(DATA_PATH, "5_func_compare_result/score"), 
        os.path.join(DATA_PATH, "5_func_compare_result/score_top50"), 
        os.path.join(DATA_PATH, "5_func_compare_result"))


