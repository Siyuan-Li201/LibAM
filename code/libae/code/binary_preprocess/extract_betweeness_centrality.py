# -*- encoding: utf-8 -*-
'''
@File    :   extract_betweeness_centrality.py
@Time    :   2022/11/25 13:11:13
@Author  :   WangYongpan 
'''
import networkx as nx

def betweeness(g):
    return nx.betweenness_centrality(g)

def eigenvector(g):
    return nx.eigenvector_centrality(g)

def closeness_centrality(g):
    return nx.closeness_centrality(g)\

def retrieveGP(g):
    bf = betweeness(g)
    x = sorted(bf.values())
    value = sum(x)/len(x)
    return round(value, 5)
