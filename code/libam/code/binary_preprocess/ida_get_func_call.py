# -*- encoding: utf-8 -*-
'''
@File    :   ida_get_func_call.py
@Time    :   2022/11/25 13:10:27
@Author  :   WangYongpan 
'''
__author__ = 'installer'

import sys, os

from settings import *
sys.path.insert(0, PACKAGE_PATH)
# from settings import *
from idaapi import *
from idc import *
from idautils import *
import idc

# import matplotlib.pyplot as plt
print("hello fcg")
print(sys.path)
import networkx as nx


def get_func_len(func_ea):
    func_context_len = 0
    for head in FuncItems(func_ea):
        func_context_len += 1
    return func_context_len


def is_func_in_plt(function_ea):
    return False
    segm_name = idc.get_segm_name(function_ea).lower()
    if ".got" not in segm_name and ".plt" not in segm_name:
        return False
    else:
        return True


def auto_analysis():
    #
    # ea = ScreenEA()

    # func_num = 0
    # for func_item in  Functions():
    #     func_num += 1

    # callers = dict()
    inputName = idc.GetInputFilePath()
    callees = dict()
    func_addr_dict = dict()

    #
    for function_ea in Functions():  # SegStart(ea), SegEnd(ea)):

        if False == is_func_in_plt(function_ea):

            f_name = GetFunctionName(function_ea)
            func_addr_dict[f_name] = function_ea

            find_func = False

            for ref_ea in CodeRefsTo(function_ea, 0):
                if False == is_func_in_plt(ref_ea):
                    find_func = True

                    caller_name = GetFunctionName(ref_ea)

                    callees[caller_name] = callees.get(caller_name, set())

                    callees[caller_name].add(f_name)
            # for ref_from in CodeRefsFrom(function_ea, 0):
            #     if False == is_func_in_plt(ref_from):
            #         find_func = True
            #         # callee_from_name = GetFunctionName(ref_from)
                    
            #         # callees[f_name] = callees.get(f_name, set())
                    
            #         # callees[f_name].add(callee_from_name)
                    
            #         # buchong
                    
                    
                    
            # if find_func == False:
            #     callees[f_name] = set()

    # func_set = set()
    # for func in callees:
    #     func_set.add(func)
    #     for func_item in callees[func]:
    #         func_set.add(func_item)

    g = nx.DiGraph()

    # g.set_rankdir('LR')
    # g.set_size('11 11')
    # g.add_node(pydot.Node('node', shape='ellipse', color='lightblue', style='filled'))
    # g.add_node(pydot.Node('edge', color='lightgrey'))

    # functions = set.union(set(callees.keys()), set(callers.keys()))
    functions = set(callees.keys())

    for f in functions:
        if f in callees and f != "":
            g.add_node(f, func_addr=func_addr_dict[f])
            for f2 in callees[f]:
                g.add_node(f2, func_addr=func_addr_dict[f2])
                g.add_edge(f, f2)

    # g_str = g.to_string()

    # print(g_str)

    # nx.draw(g, with_labels=True)
    # plt.savefig('./fcg.png')
    # plt.show()
    savePath = idc.ARGV[1]
    # binary_name = os.path.basename(inputName)
    # project_name = inputName.split("/")[-2]
    # opti_name = inputName.split("/")[-3]
    # # arch_name = inputName.split("/")[-3]
    # binary_name = opti_name+"||||"+project_name+"||||" + binary_name
    # save_name = "/data/wangyongpan/libdb_dataset_fcg/" + binary_name + "_fcg.pkl"
    nx.write_gpickle(g, savePath)


Wait()
auto_analysis()
Exit(0)
