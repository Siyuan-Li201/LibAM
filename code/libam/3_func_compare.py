import sys, os
# import click
from settings import *
sys.path.append("code/anchor_detection/semantic_anchor_detection")
sys.path.append("code/binary_preprocess")
sys.path.append("code/embeddings_generate")
sys.path.append("code/anchor_reinforcement/anchor_alignment")
sys.path.append("code/reuse_area_exploration/Embeded-GNN")
sys.path.append("code/reuse_area_exploration/TPL_detection")
sys.path.append("code/reuse_area_exploration/reuse_area_detection")


import all_func_compare_isrd as anchor_detection_module
import binary_preprocess as binary_preprocess_module
import Generate_func_embedding as embeddings_generate_module
import get_tainted_graph as anchor_reinforcement_module
import fcg_gnn_score as embeded_gnn_module
import get_final_score_multi as TPL_detection_module1
import get_final_result_dict as TPL_detection_module2
import cal_result as TPL_detection_module3
import adjust_area as area_adjustment_module
import compare_area as reuse_area_detection_module


def cli():
    print("hello libAE")
    
    # 1. get feature and fcg
    # print("start bianry preprocess......")
    # binary_preprocess_module.getAllFiles(DATA_PATH+"2_target/timecost", DATA_PATH+"1_binary/target", DATA_PATH+"2_target/", mode="1")
    # binary_preprocess_module.getAllFiles(DATA_PATH+"3_candidate/timecost", DATA_PATH + "1_binary/candidate", DATA_PATH + "3_candidate/", mode="1")
    
    # # # 2. get embedding
    # print("start embeding generation......")
    # embeddings_generate_module.subfcg_embedding(DATA_PATH+"4_embedding/timecost",
    #                                             DATA_PATH+"2_target/feature",
    #                                             DATA_PATH+"4_embedding/target_in9_bl5_embedding.json",
    #                                             model_path=WORK_PATH + "/code/embeddings_generate/gnn-best.pt")
    # embeddings_generate_module.subfcg_embedding(DATA_PATH+"4_embedding/timecost",
    #                                             DATA_PATH+"3_candidate/feature",
    #                                             DATA_PATH+"4_embedding/candidate_in9_bl5_embedding.json",
    #                                             model_path=WORK_PATH + "/code/embeddings_generate/gnn-best.pt")

    # embeddings_generate_module.generate_afcg(DATA_PATH+"4_embedding/tar_afcg",
    #                                         os.path.join(DATA_PATH, "2_target/fcg"), 
    #                                         DATA_PATH+"4_embedding/target_in9_embedding.json",
    #                                         model_path=os.path.join(WORK_PATH, "code/reuse_area_exploration/Embeded-GNN/fcg_gnn-best-0.01.pt"))
    
    # embeddings_generate_module.generate_subgraph(DATA_PATH+"4_embedding/tar_subgraph",
    #                                         os.path.join(DATA_PATH, "2_target/fcg"), 
    #                                         DATA_PATH+"4_embedding/target_in9_embedding.json",
    #                                         model_path=os.path.join(WORK_PATH, "code/reuse_area_exploration/Embeded-GNN/fcg_gnn-best-0.01.pt"))
   
    # embeddings_generate_module.generate_afcg(DATA_PATH+"4_embedding/cdd_afcg",
    #                                         os.path.join(DATA_PATH, "3_candidate/fcg"), 
    #                                         DATA_PATH+"4_embedding/candidate_in9_embedding.json",
    #                                         model_path=os.path.join(WORK_PATH, "code/reuse_area_exploration/Embeded-GNN/fcg_gnn-best-0.01.pt"))
    
    # embeddings_generate_module.generate_subgraph(DATA_PATH+"4_embedding/cdd_subgraph",
    #                                         os.path.join(DATA_PATH, "3_candidate/fcg"), 
    #                                         DATA_PATH+"4_embedding/candidate_in9_embedding.json",
    #                                         model_path=os.path.join(WORK_PATH, "code/reuse_area_exploration/Embeded-GNN/fcg_gnn-best-0.01.pt"))



    # # 3. function_compare
    print("start anchor detection......")
    anchor_detection_module.func_compare_annoy_fast_multi(os.path.join(DATA_PATH, "4_embedding/target_in9_embedding.json"), 
        os.path.join(DATA_PATH, "4_embedding/candidate_in9_embedding.json"), 
        os.path.join(DATA_PATH, "5_func_compare_result/score"), 
        os.path.join(DATA_PATH, "5_func_compare_result/score_top50"), 
        os.path.join(DATA_PATH, "5_func_compare_result"),
        os.path.join(DATA_PATH, "5_func_compare_result/embedding_annoy"))
   
    
    
    # # 4. TPL detection
    # print("start fast TPL detection......")
    # save_path = "6_tpl_fast_result/"
    # anchor_reinforcement_module.tpl_detection_fast_annoy(os.path.join(DATA_PATH, "2_target/fcg"),
    #                     os.path.join(DATA_PATH, "3_candidate/fcg"), 
    #                     os.path.join(DATA_PATH, "5_func_compare_result/score/"), 
    #                     os.path.join(DATA_PATH, save_path+"tpl_fast_result"), 
    #                     os.path.join(DATA_PATH, save_path+"tpl_fast_area"), 
    #                     os.path.join(DATA_PATH, save_path+"tpl_fast_time"),
    #                     os.path.join(DATA_PATH, "4_embedding"),
    #                     os.path.join(DATA_PATH, save_path+"sim_func_list"),
    #                     os.path.join(DATA_PATH, "4_embedding/target_in9_bl5_embedding.json"), 
    #                     os.path.join(DATA_PATH, "4_embedding/candidate_in9_bl5_embedding.json"),
    #                     os.path.join(WORK_PATH, "code/reuse_area_exploration/Embeded-GNN/fcg_gnn-best-0.01.pt"),
    #                     DATA_PATH+"4_embedding/tar_afcg",
    #                     DATA_PATH+"4_embedding/cdd_afcg",
    #                     DATA_PATH+"4_embedding/tar_subgraph",
    #                     DATA_PATH+"4_embedding/cdd_subgraph")
    # TPL_detection_module2.get_result_json(os.path.join(DATA_PATH, save_path+"tpl_fast_result"), os.path.join(DATA_PATH, save_path+"tpl_fast_result.json"))
    # TPL_detection_module3.cal_libae_result(os.path.join(DATA_PATH, save_path+"tpl_fast_result.json"), os.path.join(GT_PATH, "tpl_ground_truth.json"), os.path.join(DATA_PATH, save_path+"TPL_score/"))
    
    
    # # 5. TPL detection
    # print("start area detection......")
    # anchor_reinforcement_module.reuse_area_detection_annoy(os.path.join(DATA_PATH, "2_target/fcg"),
    #                     os.path.join(DATA_PATH, "3_candidate/fcg"), 
    #                     os.path.join(DATA_PATH, "5_func_compare_result/score_top50"), 
    #                     os.path.join(DATA_PATH, "7_reuse_detection_result/reuse_detection_result"), 
    #                     os.path.join(DATA_PATH, "7_reuse_detection_result/reuse_detection_area"), 
    #                     os.path.join(DATA_PATH, "7_reuse_detection_result/reuse_detection_time"),
    #                     os.path.join(DATA_PATH, "4_embedding"),
    #                     os.path.join(DATA_PATH, "6_alignment_result/sim_func_list"),
    #                     os.path.join(DATA_PATH, "4_embedding/target_in9_bl5_embedding.json"), 
    #                     os.path.join(DATA_PATH, "4_embedding/candidate_in9_bl5_embedding.json"),
    #                     os.path.join(WORK_PATH, "code/reuse_area_exploration/Embeded-GNN/fcg_gnn-best-0.01.pt"),
    #                     DATA_PATH+"4_embedding/tar_afcg",
    #                     DATA_PATH+"4_embedding/cdd_afcg",
    #                     DATA_PATH+"4_embedding/tar_subgraph",
    #                     DATA_PATH+"4_embedding/cdd_subgraph",
    #                     os.path.join(DATA_PATH, "6_tpl_fast_result/tpl_fast_result.json"))
       
    # print("start reuse area detection......")
    # reuse_area_detection_module.get_area_result_several(os.path.join(DATA_PATH, "7_reuse_detection_result/reuse_detection_area/"), 
    #      os.path.join(DATA_PATH, "8_reuse_area_result/reuse_detection_area"), 
    #      os.path.join(GT_PATH, "area_ground_truth.json"),
    #      os.path.join(DATA_PATH, "2_target/fcg"), 
    #      os.path.join(DATA_PATH, "3_candidate/fcg") )
    
    
   

if __name__ == "__main__":
    cli()
