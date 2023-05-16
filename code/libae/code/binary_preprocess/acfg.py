# -*- encoding: utf-8 -*-
'''
@File    :   acfg.py
@Time    :   2022/11/25 13:11:20
@Author  :   WangYongpan 
'''
import sys
from settings import *
sys.path.insert(0, PACKAGE_PATH)
import networkx as nx

# 函数的acfg图
class func_acfg:
    # 初始化
    def __init__(self, funcName, g, func_f, calls=[]):
        self.funcName = funcName
        self.old_g = g[0]
        self.g = nx.DiGraph()
        self.calls = calls
        self.entry = g[1]
        self.func_featuers = func_f  # 函数特征 用于对比
        self.getAttributes()
        pass

    def __len__(self):
        return len(self.g)

    def getAttributes(self):
        self.obtainIns(self.old_g)
        self.obtainOffsprings(self.old_g)
        # 构建节点
        for node in self.old_g:
            f_vector = self.retrieveVec(node, self.old_g)
            self.g.add_node(node)
            self.g.node[node]['vec'] = f_vector

        # 构建边
        for edge in self.old_g.edges():
            node1 = edge[0]
            node2 = edge[1]
            self.g.add_edge(node1, node2)
        pass

    # 获取CFG所有节点的入度数量
    def obtainIns(self, g):
        nodes = g.nodes()
        for node in nodes:
            ins = {}
            self.getIn(g, node, ins)
            g.node[node]['numIn'] = len(ins)
        return g
        pass

    #递归计算节点的入度数量
    def getIn(self, g, node, ins):
        pres = g.predecessors(node)
        for pre in pres:
            if pre not in ins:
                ins[pre] = 1
                self.getIn(g, pre, ins)
        pass

    # 获取CFG所有节点的后代节点数量
    def obtainOffsprings(self, g):
        nodes = g.nodes()
        for node in nodes:
            offsprings = {}
            self.getOffsprings(g, node, offsprings)
            g.node[node]['offsprings'] = len(offsprings)
        return g
        pass

    # 递归计算节点的后代节点数量
    def getOffsprings(self, g, node, offsprings):
        succs = g.successors(node)
        for succ in succs:
            if succ not in offsprings:
                offsprings[succ] = 1
                self.getOffsprings(g, succ, offsprings)
        pass

    # 返回基本块属性列表
    def retrieveVec(self, id_, g):
        feature_vec = []
        # 字符串常量
        strings = g.node[id_]['strings']
        # 求所有字符串常量的长度平均值
        # s_sum = 0
        # if len(strings) != 0:
        #     for s in strings:
        #         s_sum += len(s)
        #         pass
        #     s_sum = round(s_sum/len(strings), 5)
        s_sum = float(len(strings))
        feature_vec.append(s_sum)
        # 数值常数
        numConsts = g.node[id_]['consts']
        # 求所有数值常数的平均值
        # numC = 0
        # if len(numConsts) != 0:
        #     numC = round(sum(numConsts)/len(numConsts), 5)
        #     while numC > 10:
        #         numC = numC / 10
        #         pass
        numC = float(len(numConsts))
        feature_vec.append(numC)
        # 转移指令的数量
        numTransfer = float(g.node[id_]['numTransfer'])
        feature_vec.append(numTransfer)
        # 调用的数量
        numCalls = float(g.node[id_]['numCalls'])
        feature_vec.append(numCalls)
        # 指令的数量
        numInstr = float(g.node[id_]['numIns'])
        feature_vec.append(numInstr)
        # 算数指令的数量
        numAs = float(g.node[id_]['numAs'])
        feature_vec.append(numAs)
        # 后代节点数量
        offsprings = float(g.node[id_]['offsprings'])
        feature_vec.append(offsprings)

        # # 逻辑指令的数量
        # numLIs = g.node[id_]['numLIs']
        # feature_vec.append(numLIs)
        # # 入度数量
        # numIn = g.node[id_]['numIn']
        # feature_vec.append(numIn)
        return feature_vec
        pass

# 软件所有函数集的图
class funcs_acfg:
    def __init__(self, binary_name):
        self.binary_name = binary_name
        self.func_acfg_list = []

    # 添加函数acfg到列表中
    def append(self, func_g):
        self.func_acfg_list.append(func_g)

    def __len__(self):
        return len(self.func_acfg_list)

