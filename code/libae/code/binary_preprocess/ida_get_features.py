# -*- encoding: utf-8 -*-
'''
@File    :   ida_get_features.py
@Time    :   2022/11/25 13:10:16
@Author  :   WangYongpan 
'''
import os.path
import os
from getAcfg import *
from idc import *
from idautils import *
from idaapi import *
import json
import idc
from time import gmtime, strftime

# 提取程序所有函数的基本块特征及函数特征
def extract_features():
    times = {}
    time_begin = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    analysis_flags = idc.GetShortPrm(idc.INF_START_AF)
    analysis_flags &= ~idc.AF_IMMOFF
    # 关闭启发式自动补正
    idc.SetShortPrm(idc.INF_START_AF, analysis_flags)

    # 修改提取结果的存储位置及名称
    # filePath = '/data/wangyongpan/paper/reuse_detection/datasets/paper_datasets/libsndfile-1.0.28-com-features/'
    # inputName = idc.GetInputFilePath()
    savePath = idc.ARGV[1]
    print("itss:", savePath)

    # arch = inputName.split('/')[-3]
    # opt = inputName.split('/')[-2]
    # softwareName = inputName.split('/')[-1]
    # # a = inputName.split('/')[-4]
    # # # # softwareName = inputName.split('/')[-2] + "|||" + inputName.split('/')[-1]
    # fileName = filePath + softwareName + "|||" + arch + "|||" + opt + '.json'
    # fileName = filePath + inputName.split('/')[-1] + ".json"

    # if os.path.exists(fileName):
    #     idc.Exit(0)
    #     return
    
    # 开始处理
    idaapi.autoWait()
    cfgs = get_func_cfgs_c(FirstSeg())
    # 对cfgs进行处理
    binaryName = idc.GetInputFilePath()
    for nodes in cfgs.func_acfg_list:
        dict = {}
        dict["src"] = binaryName
        # dict["succs"] = list(nodes.g.edges())
        succs = []
        for node in range(len(nodes.g)):
            succs.append([])
        for edge in nodes.g.edges():
            succs[edge[0]].append(edge[1])
        dict["succs"] = succs
        dict["n_num"] = len(nodes.g)
        features = []
        for node in nodes.g.nodes():
            features.append(nodes.g.node[node]['vec'])
            pass
        dict["features"] = list(features)
        dict["fname"] = nodes.funcName
        dict["calls"] = nodes.calls
        saveJsonDocument(dict, savePath)
    time_end = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    times[binaryName] = time_begin + "||" + time_end
    #saveJsonDocument(times, filePth + 'time.json')
    idc.Exit(0)
    return cfgs

# 对提取出来的acfg进行json存储
def saveJsonDocument(dicts, fileName):
    with open(fileName, 'a') as out:
        json.dump(dicts, out, ensure_ascii=False)
        out.write("\n")
    out.close()
    pass

if __name__ == '__main__':
    print("check point 1")
    extract_features()
    print("check point 2")
