 
import json, os
from tqdm import tqdm
 
feature_save_path = "result/1109_6_libae_libdb_top100"
 
 
 
 
 
 
 
result_dict = {}

for result_item in tqdm(os.listdir(feature_save_path)):
    if result_item != "time.json":
        obj_item = result_item.split("----")[0]
        cdd_item = result_item.split("----")[1].split("_feature_result_")[0]
        alignment_num = int(result_item.split("----")[1].split("_feature_result_")[1].split(".json")[0])
        if obj_item not in result_dict:
                result_dict[obj_item] = []
        if alignment_num >= 3:# or alignment_num < -10:
            if cdd_item not in result_dict[obj_item]:
                result_dict[obj_item].append(cdd_item)
                
json.dump(result_dict, open("result/1109_6_libae_libdb_top100.json","w"))