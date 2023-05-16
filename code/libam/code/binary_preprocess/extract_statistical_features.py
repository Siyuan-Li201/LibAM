# -*- encoding: utf-8 -*-
'''
@File    :   extract_statistical_features.py
@Time    :   2022/11/25 13:10:47
@Author  :   WangYongpan 
'''
from idautils import *
from idaapi import *
from idc import *

# 获取基本块内每条指令的字符串常量和数值常量
def getConst(ea, offset):
    strings = []
    consts = []
    opType1 = GetOpType(ea, offset)
    if opType1 == idaapi.o_imm:
        imm_value = GetOperandValue(ea, offset)
        if 0 <= imm_value <= 10:
            consts.append(imm_value)
        else:
            if idaapi.isLoaded(imm_value) and idaapi.getseg(imm_value):
                str_value = GetString(imm_value)
                if str_value is None:
                    str_value = GetString(imm_value + 0x40000)
                    if str_value is None:
                        consts.append(imm_value)
                    else:
                        re = all(40 <= ord(c) < 128 for c in str_value)
                        if re:
                            strings.append(str_value)
                        else:
                            consts.append(imm_value)
                else:
                    re = all(40 <= ord(c) < 128 for c in str_value)
                    if re:
                        strings.append(str_value)
                    else:
                        consts.append(imm_value)
            else:
                consts.append(imm_value)
    return strings, consts
    pass

# 获取给定基本块的所有字符串常量和数值常量
def getBBconsts(bl):
    strings = []
    consts = []
    start = bl[0]
    end = bl[1]
    inst_addr = start
    while inst_addr < end:
        opcode = GetMnem(inst_addr)
        if opcode in ['la', 'jalr', 'call', 'jal']:
            inst_addr = NextHead(inst_addr)
            continue
        strings_src, consts_src = getConst(inst_addr, 0)
        strings_dst, consts_dst = getConst(inst_addr, 1)
        strings += strings_src
        strings += strings_dst
        consts += consts_src
        consts += consts_dst
        try:
            strings_dst, consts_dst = getConst(inst_addr, 2)
            consts += consts_dst
            strings += strings_dst
        except:
            pass
        inst_addr = NextHead(inst_addr)
    return strings, consts
    pass

# 获取每个汇编函数中的字符串常量和数值常量
def getfunc_consts(func):
    strings = []
    consts = []
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
    for bl in blocks:
        strs, conts = getBBconsts(bl)
        strings += strs
        consts += conts
    return strings, consts

# 计算基本块内转移指令的数量
def calTransferIns(bl):
    # x86_TI = {'jmp': 1, 'jz': 1, 'jnz': 1, 'js': 1, 'je': 1, 'jne': 1, 'jg': 1, 'jle': 1, 'jge': 1, 'ja': 1, 'jnc': 1, 'call': 1}
    # mips_TI = {'beq': 1, 'bne': 1, 'bgtz': 1, "bltz": 1, "bgez": 1, "blez": 1, 'j': 1, 'jal': 1, 'jr': 1, 'jalr': 1}
    # arm_TI = {'MVN': 1, "MOV": 1}
    mips_TI = {"beqz": 1, "beq": 1, "bne": 1, "bgez": 1, "b": 1, "bnez": 1, "bgtz": 1, "bltz": 1, "blez": 1, "bgt": 1,  "bge": 1, "blt": 1, "ble": 1, "bgtu": 1, "bgeu": 1, "bltu": 1, "bleu": 1}
    x86_TI = {"jz": 1, "jnb": 1, "jne": 1, "je": 1, "jg": 1, "jle": 1, "jl": 1, "jge": 1, "ja": 1, "jae": 1, "jb": 1,  "jbe": 1, "jo": 1, "jno": 1, "js": 1, "jns": 1, "jr": 1}
    arm_TI = {"B": 1, "BL": 1, "BAL": 1, "BNE": 1, "BEQ": 1, "BPL": 1, "BMI": 1, "BCC": 1, "BLO": 1, "BCS": 1, "BHS": 1, "BVC": 1, "BVS": 1, "BGT": 1, "BGE": 1, "BLT": 1, "BLE": 1, "BHI": 1, "BLS": 1}
    calls = {}
    calls.update(x86_TI)
    calls.update(mips_TI)
    calls.update(arm_TI)
    start = bl[0]
    end = bl[1]
    invoke_num = 0
    inst_addr = start
    while inst_addr < end:
        opcode = GetMnem(inst_addr)
        re = [v for v in calls if opcode.lower() in v.lower()]
        if len(re) > 0:
            invoke_num += 1
        inst_addr = NextHead(inst_addr)
    return invoke_num

# 计算函数内转移指令的数量
def getTransferInsts(func):
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
    sumcalls = 0
    for bl in blocks:
        callnum = calTransferIns(bl)
        sumcalls += callnum
    return sumcalls

# 计算基本块内调用的数量
def calCalls(bl):
    calls = {'call': 1, 'jal': 1, 'jalr': 1}
    start = bl[0]
    end = bl[1]
    invoke_num = 0
    inst_addr = start
    while inst_addr < end:
        opcode = GetMnem(inst_addr)
        if opcode.lower() in calls:
            invoke_num += 1
        inst_addr = NextHead(inst_addr)
    return invoke_num
    pass

# 计算函数内调用的数量
def getFuncCalls(func):
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
    sumcalls = 0
    for bl in blocks:
        callnum = calCalls(bl)
        sumcalls += callnum
    return sumcalls

# 计算基本块内指令的数量
def calInstrs(bl):
    start = bl[0]
    end = bl[1]
    inst_addr = start
    invoke_num = 0
    while inst_addr < end:
        invoke_num += 1
        inst_addr = NextHead(inst_addr)
    return invoke_num

# 计算函数的指令数量
def getFuncInstrs(func):
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
    sumInstr = 0
    for bl in blocks:
        instr = calInstrs(bl)
        sumInstr += instr
    return sumInstr

# 计算基本块内算数指令的数量
def calArithmeticInstr(bl):
    x86_AI = {'add': 1, 'sub': 1, 'div': 1, 'imul': 1, 'idiv': 1, 'mul': 1, 'shl': 1, 'dec': 1, 'inc': 1}
    mips_AI = {'add': 1, 'addu': 1, 'addi': 1, 'addiu': 1, 'mult': 1, 'multu': 1, 'div': 1, 'divu': 1}
    arm_AI = {'ADD': 1, 'ADC': 1, 'SUB': 1, 'RSB': 1, 'SBC': 1, 'RSC': 1, 'MUL': 1, 'MLA': 1, 'SMULL': 1, 'SMLAL': 1, 'UMULL': 1, 'UMLAL': 1}
    calls = {}
    calls.update(x86_AI)
    calls.update(mips_AI)
    calls.update(arm_AI)
    start = bl[0]
    end = bl[1]
    invoke_num = 0
    inst_addr = start
    while inst_addr < end:
        opcode = GetMnem(inst_addr)
        re = [v for v in calls if opcode.lower() in v.lower()]
        if len(re) > 0:
            invoke_num += 1
        inst_addr = NextHead(inst_addr)
    return invoke_num

# 计算基本块内逻辑指令的数量
def calLogicInstructions(bl):
    x86_LI = {'and': 1, 'andn': 1, 'andnpd': 1, 'andpd': 1, 'andps': 1, 'andnps': 1, 'test': 1, 'xor': 1, 'xorpd': 1, 'pslld': 1}
    mips_LI = {'and': 1, 'andi': 1, 'or': 1, 'ori': 1, 'xor': 1, 'nor': 1, 'slt': 1, 'slti': 1, 'sltu': 1}
    arm_LI = {'AND': 1, 'ORR': 1, 'EOR': 1, 'BIC': 1, 'TEQ': 1, 'TST': 1}
    calls = {}
    calls.update(x86_LI)
    calls.update(mips_LI)
    calls.update(arm_LI)
    start = bl[0]
    end = bl[1]
    invoke_num = 0
    inst_addr = start
    while inst_addr < end:
        opcode = GetMnem(inst_addr)
        re = [v for v in calls if opcode.lower() in v.lower()]
        if len(re) > 0:
            invoke_num += 1
        inst_addr = NextHead(inst_addr)
    return invoke_num

# 计算函数内的逻辑指令的数量
def getLogicInsts(func):
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
    sumcalls = 0
    for bl in blocks:
        callnum = calLogicInstructions(bl)
        sumcalls += callnum
    return sumcalls

# 获取该基本块调用的基本块地址
def retrieveExterns(bl, ea_externs):
    externs = []
    start = bl[0]
    end = bl[1]
    inst_addr = start
    while inst_addr < end:
        refs = CodeRefsFrom(inst_addr, 1)
        try:
            ea = [v for v in refs if v in ea_externs][0]
            externs.append(ea_externs[ea])
        except:
            pass
        inst_addr = NextHead(inst_addr)
    return externs


def getLocalVariables(func):
    args_num = get_stackVariables(func.startEA)
    return args_num

# 获取存储的本地变量
def get_stackVariables(func_addr):
    args = []
    stack = GetFrame(func_addr)
    if not stack:
        return 0
    firstM = GetFirstMember(stack)
    lastM = GetLastMember(stack)
    i = firstM
    while i <= lastM:
        mName = GetMemberName(stack, i)
        mSize = GetMemberSize(stack, i)
        if mSize:
            i = i + mSize
        else:
            i = i+4
        if mName not in args and mName and 'var_' in mName:
            args.append(mName)
    return len(args)

# 获取基本块的数量
def getBasicBlocks(func):
    blocks = [(v.startEA, v.endEA) for v in FlowChart(func)]
    return len(blocks)

# 获取调用该函数的地址数量
def getIncommingCalls(func):
    refs = CodeRefsTo(func.startEA, 0)
    re = len([v for v in refs])
    return re
