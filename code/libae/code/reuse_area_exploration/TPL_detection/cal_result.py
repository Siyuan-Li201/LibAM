import os, json, copy, sys
sys.path.append(".")
from settings import DATA_PATH

def get_json_data(filename):
    with open(filename,"r") as ff:
        data = json.load(ff)

        return data

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
    
    # full_P += Precision
    # full_R += Recall
    # full_F1 += F1

    #print(detected_item + "  Precision: {}  Recall: {}  F1: {}".format(Precision, Recall, F1) )
            
    # print("Full Precision: {}  Recall: {}  F1: {}".format(full_P/test_num, full_R/test_num, full_F1/test_num) )



def write_json_data(dict, path):
    json.dump(dict, open(path, "w"))

def cli(feature_save_path, libdb_path, ground_truth_path, result_json_path, gemini_path,b2sfinder, libae_path):
    
    # result_dict = {}
    error_list = ["cknit", "turbobench", "exjson"]
    
    # for result_item in os.listdir(feature_save_path):
    #     if result_item != "time.json":
    #         obj_item = result_item.split("----")[0]
    #         if obj_item not in error_list:
    #             cdd_item = result_item.split("----")[1].split("_datasets_isrd_to_gemini_feature_result")[0]
    #             alignment_num = int(result_item.split("----")[1].split("_datasets_isrd_to_gemini_feature_result_")[1].split(".json")[0])
                
    #             if alignment_num >= 3 or alignment_num < -10:
    #                 if obj_item not in result_dict:
    #                     result_dict[obj_item] = []
    #                 if cdd_item not in result_dict[obj_item]:
    #                     result_dict[obj_item].append(cdd_item)
        
    # write_json_data(result_dict, result_json_path)
    ground_truth_dict = get_json_data(ground_truth_path)
    libae_result = get_json_data(libae_path)
    libdb_result = get_json_data(libdb_path)
    libdx_result = get_json_data(libdx_path)
    Gemini_result = get_json_data(gemini_path)
    b2sfinder_str_result = get_json_data(b2sfinder)
    for error_item in error_list:
        if error_item in libdb_result:
            del libdb_result[error_item]
    for error_item in error_list:
        if error_item in libdx_result:
            del libdx_result[error_item]
    
    # new_libAE_result = copy.deepcopy(libae_result)
    # for libAE_item in libae_result:
    #     libAE_item_ = libAE_item.replace("___","_")
    #     for libDX_lib in libdx_result[libAE_item_]:
    #         if libDX_lib not in libae_result[libAE_item]:
    #             new_libAE_result[libAE_item].append(libDX_lib)
    # new_libAE_result =  dict( libdx_result, **result_dict )
    # new_libDB_result =  dict( libdx_result, **result_dict )
    
    COMPARE_RESULT = libae_result
    METHOD = "result_analog_gnn_0.8"
    ARCH = "x64"
    AVERAGE0 = ["arm_O2", "x86_O2", "x64_O0", "x64_O1", "x64_O2", "x64_O3"]
    AVERAGE1 = ["arm_O2", "x86_O2", "x64_O2"]
    AVERAGE2 = ["x64_O0", "x64_O1", "x64_O2", "x64_O3"]
    AVERAGE = AVERAGE2
    # AVERAGE = ["arm|||O2", "x86|||O2", "x64|||O0", "x64|||O1", "x64|||O2", "x64|||O3"]
    OPTI = "O0"
    ARGS = "triple"
    SCORE_PATH = "F:\\mypaper\\result\\"+METHOD+"_"+ARCH+"_"+OPTI+"_"+ARGS+"_"+"result.json"
    RESULT_PATH = "F:\\mypaper\\result\\" + METHOD + "_result.json"
    
    full_P = 0
    full_R = 0
    full_F1 = 0
    
    candidate_num = 0
    with open(SCORE_PATH, "w") as result_f:
        for obj_item in COMPARE_RESULT: 
            condition = "xz" not in obj_item  and "gipfeli" not in obj_item  and "libdeflate" not in obj_item and "zstd" not in obj_item  and "brotli" not in obj_item and "lzo" not in obj_item and "quicklz" not in obj_item  and "liblzg" not in obj_item
            if ARCH == "average" and OPTI == "average":
                average_flag = False
                for average_item in AVERAGE:
                    if average_item in obj_item or average_item.replace("_","___") in obj_item:
                        average_flag = True
            if (ARCH=="all" and OPTI=="all" and "xz" not in obj_item and condition == True) or (ARCH in obj_item and OPTI in obj_item and condition == True) or (ARCH=="isrd" and OPTI=="isrd" and "_" not in obj_item and condition == True) or (ARCH == "average" and OPTI == "average" and average_flag == True and condition == True):
                candidate_num += 1
                if "|||" in obj_item:
                    obj_item_name = obj_item.split("|||")[0]
                elif "_" in obj_item:
                    obj_item_name = obj_item.split("_")[0]
                else:
                    obj_item_name = obj_item
                
                (Precision, Recall, F1_s) = test_func(COMPARE_RESULT[obj_item], ground_truth_dict[obj_item_name])
                
                print(obj_item+" score: P:{} R:{} F1:{}".format(Precision, Recall, F1_s))
                result_f.write(obj_item+" score: P:{} R:{} F1:{}\n".format(Precision, Recall, F1_s))
                full_P += Precision
                full_R += Recall
                full_F1 += F1_s
            
        print("Full Precision: {}  Recall: {}  F1: {}".format(full_P/candidate_num, full_R/candidate_num, full_F1/candidate_num) )
        result_f.write("Full Precision: {}  Recall: {}  F1: {}\n".format(full_P/candidate_num, full_R/candidate_num, full_F1/candidate_num) )
    with open(RESULT_PATH, "w") as result_f:
        json.dump(COMPARE_RESULT, result_f)
            

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
    all = ["arm_O0", "arm_O1", "arm_O2", "arm_O3", "x86_O0", "x86_O1", "x86_O2", "x86_O3", "x64_O0", "x64_O1", "x64_O2", "x64_O3"]
    libae_result_new = copy.deepcopy(libae_result)
    
    
    for arch_opti in all:
        libsndfile_sndfile2k = []
        find_it = False
        for obj_item in libae_result:
            if "sndfile" in obj_item and arch_opti in obj_item:
                find_it = True
                for cdd_item in libae_result[obj_item]:
                    if cdd_item not in libsndfile_sndfile2k:
                        libsndfile_sndfile2k.append(cdd_item)
                libae_result_new.pop(obj_item)
        if find_it:
            libae_result_new["libsndfile-sndfile2k_"+arch_opti] = libsndfile_sndfile2k
    
    for obj_item in libae_result:
        if "sndfile" in obj_item:
            find_flag = False
            for arch_opti in all:
                if arch_opti in obj_item:
                    find_flag = True
            if False == find_flag:
                libsndfile_sndfile2k = []
                for cdd_item in libae_result[obj_item]:
                    if cdd_item not in libsndfile_sndfile2k:
                        libsndfile_sndfile2k.append(cdd_item)
                libae_result_new.pop(obj_item)
    libae_result_new["libsndfile-sndfile2k"] = libsndfile_sndfile2k
                        
    
    return libae_result_new


def cal_score(mode_func, arg2, result_save_path, mode_item, libae_result, ground_truth_dict):
    with open(os.path.join(result_save_path, "TPL_result_"+mode_item), "w") as result_f:
        full_P = 0
        full_R = 0
        full_F1 = 0
        candidate_num = 0
        
        libae_result_deal = deal_with_sndfile(libae_result)
        
        find_it = False
        
        for obj_item in libae_result_deal:
            if mode_func(obj_item, arg2) == 0:
                if obj_item.split("_")[0] in libae_result_deal[obj_item]:
                    libae_result_deal[obj_item].remove(obj_item.split("_")[0])
                candidate_num += 1
                (Precision, Recall, F1_s) = test_func(libae_result_deal[obj_item], ground_truth_dict[obj_item.split("_")[0]])
                find_it = True
                print(obj_item+" score: P:{} R:{} F1:{}".format(Precision, Recall, F1_s))
                result_f.write(obj_item+" score: P:{} R:{} F1:{}\n".format(Precision, Recall, F1_s))
                full_P += Precision
                full_R += Recall
                full_F1 += F1_s
        if find_it:
            if candidate_num > 0:
                print("Full Precision: {}  Recall: {}  F1: {}".format(full_P/candidate_num, full_R/candidate_num, full_F1/candidate_num) )
                result_f.write("Full Precision: {}  Recall: {}  F1: {}\n".format(full_P/candidate_num, full_R/candidate_num, full_F1/candidate_num) )
            else:
                print("Full Precision: {}  Recall: {}  F1: {}".format(full_P, full_R, full_F1) )
                result_f.write("Full Precision: {}  Recall: {}  F1: {}\n".format(full_P, full_R, full_F1) )


def cal_libae_result(libae_path, ground_truth_path, result_save_path):
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    libae_result = get_json_data(libae_path)
    ground_truth_dict = get_json_data(ground_truth_path)
    
    all = ["arm_O0", "arm_O1", "arm_O2", "arm_O3", "x86_O0", "x86_O1", "x86_O2", "x86_O3", "x64_O0", "x64_O1", "x64_O2", "x64_O3"]
    arch_average = ["arm_O2", "x86_O2", "x64_O2"]
    opti_average = ["x64_O0", "x64_O1", "x64_O2", "x64_O3"]
    
    mode_list = ["isrd", "x64_O0", "x64_O1", "x64_O2", "x64_O3", "opti_average", "arm_O2", "x86_O2", "x64_O2", "arch_average", "all"]
    
    for mode_item in mode_list:
        if mode_item == "isrd":
            cal_score(not_in_target, all, result_save_path, mode_item, libae_result, ground_truth_dict)
        elif mode_item == "opti_average":
            cal_score(is_in_target, opti_average, result_save_path, mode_item, libae_result, ground_truth_dict)
        elif mode_item == "arch_average":
            cal_score(is_in_target, arch_average, result_save_path, mode_item, libae_result, ground_truth_dict)
        elif mode_item == "all":
            cal_score(is_in_target, all, result_save_path, mode_item, libae_result, ground_truth_dict)
        else:
            cal_score(is_mode, mode_item, result_save_path, mode_item, libae_result, ground_truth_dict)


def main():
    # ground_truth_path = "F:\\mypaper\\libae_ground_truth.json"
    # libae_path = "F:\\mypaper\\data\\result\\result_analog_gnn_0.8.json"
    # result_save_path = "F:\\mypaper\\"
    # cli(feature_save_path, libdb_path, ground_truth_path, result_dict, gemini_path, b2sfinder, libae_path)
    cal_libae_result(os.path.join(DATA_PATH, "8_tpl_result/TPL_result_8.json"), os.path.join(DATA_PATH, "libae_ground_truth.json"), os.path.join(DATA_PATH, "8_tpl_result/TPL_score/"))



if __name__ == "__main__":
    main()
    