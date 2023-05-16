# -*- encoding: utf-8 -*-
'''
@File    :   buildAcfg.py
@Time    :   2022/11/25 13:10:55
@Author  :   WangYongpan 
'''
import networkx as nx
from idautils import *
from idaapi import *
from idc import *
from extract_statistical_features import *

# 构建函数的ACFG
def buildCfg(func, externs_eas, ea_externs):
    func_start = func.startEA
    func_end = func.endEA
    cfg = nx.DiGraph()
    control_blocks, main_blocks = obtain_block_sequence(func)
    visited = {}
    for bl in control_blocks:
        start = control_blocks[bl][0]
        end = control_blocks[bl][1]
        src_node = (start, end)
        if src_node not in visited:
            src_id = len(cfg)
            visited[src_node] = src_id
            cfg.add_node(src_id)
            cfg.node[src_id]['label'] = src_node
        else:
            src_id = visited[src_node]
        if start == func_start:
            cfg.node[src_id]['c'] = "start"
        if end == func_end:
            cfg.node[src_id]['c'] = "end"
        # 只有跳转
        refs = CodeRefsTo(start, 0)
        for ref in refs:
            if ref in control_blocks:
                dst_node = control_blocks[ref]
                if dst_node not in visited:
                    visited[dst_node] = len(cfg)
                dst_id = visited[dst_node]
                cfg.add_edge(dst_id, src_id)
                cfg.node[dst_id]['label'] = dst_node
        # 包括代码顺序执行调用
        refs = CodeRefsTo(start, 1)
        for ref in refs:
            if ref in control_blocks:
                dst_node = control_blocks[ref]
                if dst_node not in visited:
                    visited[dst_node] = len(cfg)
                dst_id = visited[dst_node]
                cfg.add_edge(dst_id, src_id)
                cfg.node[dst_id]['label'] = dst_node
    attributingRe(cfg, externs_eas, ea_externs)
    # ACFG图可以优化 ----待做
    return cfg, 0
    pass

def attributingRe(cfg, externs_eas, ea_extern):
    for node_id in cfg:
        bl = cfg.node[node_id]['label']
        numIns = calInstrs(bl)
        # 指令数量
        cfg.node[node_id]['numIns'] = numIns
        numCalls = calCalls(bl)
        # 调用数量
        cfg.node[node_id]['numCalls'] = numCalls
        numAs = calArithmeticInstr(bl)
        # 算数指令数量
        cfg.node[node_id]['numAs'] = numAs
        strings, consts = getBBconsts(bl)
        # 数值常数
        cfg.node[node_id]['consts'] = consts
        # 字符串常量
        cfg.node[node_id]['strings'] = strings
        cfg.node[node_id]['numNc'] = len(strings) + len(consts)
        numTransfer = calTransferIns(bl)
        # 转移指令数量
        cfg.node[node_id]['numTransfer'] = numTransfer
        externs = retrieveExterns(bl, ea_extern)
        # 调用外部基本块地址
        cfg.node[node_id]['externs'] = externs
        numLIs = calLogicInstructions(bl)
        # 逻辑指令的数量
        cfg.node[node_id]['numLIs'] = numLIs
    pass

# 获取函数内所有基本块转移地址集合
def obtain_block_sequence(func):
    control_blocks = {}
    main_blocks = {}
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
    for bl in blocks:
        base = bl[0]
        end = PrevHead(bl[1])
        control_ea = checkCB(bl)
        control_blocks[control_ea] = bl
        control_blocks[end] = bl
        if func.startEA <= base <= func.endEA:
            main_blocks[base] = bl
    x = sorted(main_blocks)
    return control_blocks, x
    pass

# 获取该基本块的跳转指令地址
def checkCB(bl):
    start = bl[0]
    end = bl[1]
    ea = start
    while ea < end:
        if checkCondition(ea):
            return ea
        ea = NextHead(ea)
    return PrevHead(end)
    pass

# 判断是否为跳转指令
def checkCondition(ea):
    mips_branch = {"beqz": 1, "beq": 1, "bne": 1, "bgez": 1, "b": 1, "bnez": 1, "bgtz": 1, "bltz": 1, "blez": 1, "bgt": 1,  "bge": 1, "blt": 1, "ble": 1, "bgtu": 1, "bgeu": 1, "bltu": 1, "bleu": 1}
    x86_branch = {"jz": 1, "jnb": 1, "jne": 1, "je": 1, "jg": 1, "jle": 1, "jl": 1, "jge": 1, "ja": 1, "jae": 1, "jb": 1,  "jbe": 1, "jo": 1, "jno": 1, "js": 1, "jns": 1, "jr": 1}
    arm_branch = {"B": 1, "BL": 1, "BAL": 1, "BNE": 1, "BEQ": 1, "BPL": 1, "BMI": 1, "BCC": 1, "BLO": 1, "BCS": 1, "BHS": 1, "BVC": 1, "BVS": 1, "BGT": 1, "BGE": 1, "BLT": 1, "BLE": 1, "BHI": 1, "BLS": 1}
    conds = {}
    conds.update(mips_branch)
    conds.update(x86_branch)
    conds.update(arm_branch)
    opcode = GetMnem(ea)
    if opcode in conds:
        return True
    return False