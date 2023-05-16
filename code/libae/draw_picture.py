import json
import os
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from settings import *




def cli(fcg_path, png_path):
    
    for object_item in os.listdir(fcg_path):
        g = nx.read_gpickle(os.path.join(fcg_path, object_item))
        
        pydot_graph = nx.nx_pydot.to_pydot(g)
        pydot_graph.set_rankdir('LR')
        pydot_graph.write_png(os.path.join(png_path, object_item[:-4]+'.png'))
            
            





if __name__ == "__main__":
    fcg_path = DATA_PATH+"1_binary/target/fcg"
    png_path = DATA_PATH+"11_fcg_png/target/fcg"
    
    if os.path.exists(png_path):
        os.makedirs(png_path)
    
    cli(fcg_path, png_path)
