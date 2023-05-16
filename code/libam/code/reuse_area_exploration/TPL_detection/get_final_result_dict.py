 
import json
import os
import sys

from tqdm import tqdm

sys.path.append(".")
from settings import DATA_PATH

# feature_save_path = "result/1109_4_libae_paper_top50"
 
# result_path = "/data/lisiyuan/libAE/TPL_detection_result/1109_5_libae_paper_top50_gnn_analog_0.001_gnn_result/reuse_result_0.8"
def get_result_json(result_path, savePath):
    
    
    result_dict = {}

    for result_item in tqdm(os.listdir(result_path)):
        item_name = result_item.split("_reuse_result")[0]
        reuse_item = json.load(open(os.path.join(result_path, result_item), "r"))
        if item_name in reuse_item:
            result_dict[item_name] = reuse_item[item_name]
                    
    json.dump(result_dict, open(savePath,"w"))
    
    
if __name__ == "__main__":
    get_result_json(os.path.join(DATA_PATH, "7_gnn_result/after_gnn_result/reuse_result_8"), os.path.join(DATA_PATH, "8_tpl_result/TPL_result_8.json"))