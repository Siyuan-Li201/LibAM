# -*- encoding: utf-8 -*-
'''
@File    :   binary_preprocess.py
@Time    :   2022/11/25 13:09:58
@Author  :   WangYongpan 
'''
import subprocess
import os
from multiprocessing import Process
import tqdm
import json
import time
import shutil

from settings import IDA_PATH, FEATURE_EXTRACT_PATH, FCG_EXTRACT_PATH

def getAllFiles(TIME_PATH, filePath, savePath, mode="multi-arch"):
    fileList = []
    tmp_extention = ['nam', 'til', 'id0', 'id1', 'id2', 'id3', 'id4', 'json', 'i64', 'a', "pkl"]
    files = os.walk(filePath)
    for path, dir_path, file_name in files:
        for file in file_name:
            # if file == "brotli" or file == "libblosc":
            if file.split(".")[-1] not in tmp_extention and "Os" not in path and "PPC" not in path:
            #if file.split(".")[-1] == "so":
                fP = str(os.path.join(path, file))
                fileList.append(os.path.join(os.path.abspath('.'), fP))
    # extract features
    feature_savePath = os.path.join(savePath, "feature/")
    if not os.path.exists(feature_savePath):
        os.makedirs(feature_savePath)
    if not os.path.exists(TIME_PATH):
        os.makedirs(TIME_PATH)
    multi_process_extract(TIME_PATH, fileList, feature_savePath, idapython=FEATURE_EXTRACT_PATH, mode=mode, type="feature")
    # extract fcg
    fcg_savePath = os.path.join(savePath, "fcg/")
    if not os.path.exists(fcg_savePath):
        os.makedirs(fcg_savePath)
    multi_process_extract(TIME_PATH, fileList, fcg_savePath, idapython=FCG_EXTRACT_PATH, mode=mode, type="fcg")
    pass

def multi_process_extract(TIME_PATH, fileList, savePath, idapython="", mode="multi-arch", type=""):
    process_num = min(30, len(fileList))
    p_list = []
    for i in range(process_num):
        files = fileList[int((i)/process_num*len(fileList)): int((i+1)/process_num*len(fileList))]
        p = Process(target=extract, args=(files, savePath, idapython, mode, type, TIME_PATH))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()

def extract(filePaths, savePath, idapython, mode, type, TIME_PATH):
    tf = tqdm.tqdm(filePaths)
    for filePath in tf:
        # if os.path.exists(os.path.join(TIME_PATH, "feature_timecost.json")):
        #     feature_timecost = json.load(open(os.path.join(TIME_PATH, "feature_timecost.json"), "r"))
        # else:
        #     feature_timecost = dict()
        # if type not in feature_timecost:
        #     feature_timecost[type] = dict()
        if not os.path.exists(os.path.join(TIME_PATH, os.path.basename(filePath)+"_"+type+"_timecost.json")):
            start_time = time.time()
            if mode == "multi-arch":
                arch = filePath.split('/')[-3]
                opt = filePath.split('/')[-2]
                binary_name = filePath.split("/")[-1]
                save_name = binary_name + "|||" + arch + "|||" + opt
            else:
                save_name = filePath.split("/")[-1]
            if type == "feature":
                save_name += ".json"
            else:
                save_name += "_fcg.pkl"
            save_name = os.path.join(savePath, save_name)
            if os.path.exists(save_name):
                continue
            (bpath, bbianry) = os.path.split(filePath)
            if not os.path.exists(os.path.join(bpath, bbianry+"_")):
                os.mkdir(os.path.join(bpath, bbianry+"_"))
                shutil.copy(filePath, os.path.join(bpath, bbianry+"_", bbianry))
            # print(os.path.join(bpath, bbianry+"_", bbianry))
            ida_cmd = 'TVHEADLESS=1 ' + IDA_PATH + ' -L/data/lisiyuan/work2023/code/libAE/libAE_github/libae/idalog.txt -c -A -B -S\'' + idapython + " " + save_name + '\' ' + os.path.join(bpath, bbianry+"_", bbianry)
            s, o = subprocess.getstatusoutput(ida_cmd)
            end_time = time.time()
            if s != 0:
                with open('error.txt', 'a') as file:
                    file.write(filePath + '\n')
                print("error: " + filePath)
            else:
                tf.set_description("[" + filePath.split("/")[-1] + "] Extract Success")
                timecost = end_time - start_time
                # if os.path.exists(os.path.join(TIME_PATH, "feature_timecost.json")):
                #     feature_timecost = json.load(open(os.path.join(TIME_PATH, "feature_timecost.json"), "r"))
                # else:
                #     feature_timecost = dict()
                # if type not in feature_timecost:
                feature_timecost = dict()
                feature_timecost[os.path.basename(filePath)] = timecost
                json.dump(feature_timecost, open(os.path.join(TIME_PATH, os.path.basename(filePath)+"_"+type+"_timecost.json"), "w"))
        
    pass


if __name__ == '__main__':
    filePath = '/data/lisiyuan/code/libAE/libae/data/isrd/1_binary/candidate'
    savePath = '/data/lisiyuan/code/libAE/libae/data/isrd/1_binary/'
    getAllFiles(filePath, savePath, "")
    print("提取完毕！")
    pass

