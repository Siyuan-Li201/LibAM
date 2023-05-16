# -*- encoding: utf-8 -*-
'''
@File    :   discovRe.py
@Time    :   2022/11/25 13:11:02
@Author  :   WangYongpan 
'''
from extract_statistical_features import *
from extract_betweeness_centrality import *


def get_discoverRe_feature(func, icfg):
    features = []
    FunctionCalls = getFuncCalls(func)
    #1
    features.append(FunctionCalls)
    LogicInstr = getLogicInsts(func)
    #2
    features.append(LogicInstr)
    Transfer = getTransferInsts(func)
    #3
    features.append(Transfer)
    Locals = getLocalVariables(func)
    #4
    features.append(Locals)
    BB = getBasicBlocks(func)
    #5
    features.append(BB)
    Edges = len(icfg.edges())
    #6
    features.append(Edges)
    Incoming = getIncommingCalls(func)
    #7
    features.append(Incoming)
    #8
    Instrs = getFuncInstrs(func)
    features.append(Instrs)
    between = retrieveGP(icfg)
    #9
    features.append(between)

    strings, consts = getfunc_consts(func)
    features.append(strings)
    features.append(consts)
    return features