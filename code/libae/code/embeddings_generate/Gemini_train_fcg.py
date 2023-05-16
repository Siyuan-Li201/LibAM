import json

import torch
import tqdm

from torch.utils.data import DataLoader
from torch.autograd import Variable
#import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import time
import pickle
from datetime import datetime
import random
from sklearn.metrics import auc, roc_curve
from multiprocessing import Pool

from Gemini_torch_gnn_model import graphnn
from Gemini_train_dataset import dataset
from Gemini_utils import get_f_name, get_f_dict, read_graph, partition_data, get_f_dict_fcg, read_graph_fcg

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
                    help='visible gpu device')
parser.add_argument('--workers', type=int, default=12,
                    help='workers')
parser.add_argument('--use_device', type=str, default='/gpu:0',
                    help='used gpu device')
parser.add_argument('--fea_dim', type=int, default=64,
                    help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64,
                    help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2,
                    help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64,
                    help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5,
                    help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--epoch', type=int, default=150,
                    help='epoch number')
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch size')
parser.add_argument('--neg_batch_size', type=int, default=512,
                    help='negative batch size')
parser.add_argument('--load_path', type=str, default=None,
                    help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--save_path', type=str,
                    default='saved_model', help='path for model saving')
parser.add_argument('--log_path', type=str, default=None,
                    help='path for training log')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--data_path', type=str,
                    default='/data/wangyongpan/paper/reuse_detection/code/reuse_minifcg/minifcg_no_analog/6.4_1_500/', help='path for all data')



def contra_loss_show(net, dataLoader, DEVICE):
    loss_val = []
    tot_cos = []
    tot_truth = []
    tq = tqdm.tqdm(dataLoader)
    for batch_id, (X1, X2, X3, m1, m2, m3) in enumerate(tq, 1):
        X1, X2, X3, m1, m2, m3 = X1[0], X2[0], X3[0], m1[0], m2[0], m3[0]
        if 'gpu' in DEVICE:
            X1, X2, X3, m1, m2, m3 = X1.cuda(non_blocking=True), X2.cuda(non_blocking=True), X3.cuda(
                non_blocking=True),  m1.cuda(non_blocking=True), m2.cuda(non_blocking=True), m3.cuda(non_blocking=True)
        # else:
        #     X1, X2, X3, m1, m2, m3 = Variable(X1), Variable(X2), Variable(
        #         X3),  Variable(m1), Variable(m2), Variable(m3)
        loss, cos_p, cos_n = net.forward(X1, X2, X3, m1, m2, m3)
        cos_p = list(cos_p.cpu().detach().numpy())
        cos_n = list(cos_n.cpu().detach().numpy())
        tot_cos += cos_p
        tot_truth += [1]*len(cos_p)
        tot_cos += cos_n
        tot_truth += [-1]*len(cos_n)
        loss_val.append(loss.item())
        tq.set_description("Eval:[" + str(loss.item()) + "]")
    cos = np.array(tot_cos)
    truth = np.array(tot_truth)
    fpr, tpr, thres = roc_curve(truth, (1+cos)/2)
    model_auc = auc(fpr, tpr)
    #return loss_val, model_auc, tpr
    return loss_val, model_auc, fpr, tpr, thres


if __name__ == '__main__':
    args = parser.parse_args()
    print("=================================")
    print(args)
    print("=================================")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    MAX_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    NEG_BATCH_SIZE = args.neg_batch_size
    LOAD_PATH = args.load_path
    SAVE_PATH = args.save_path
    LOG_PATH = args.log_path
    DEVICE = args.use_device
    WORKERS = args.workers
    DDATA_FILE_NAME_TRAIN_VALID = args.data_path
    # DATA_FILE_NAME_TEST = args.test_data

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 5


    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    print("start reading data")
    F_PATH_TRAIN_VALID = get_f_name(DDATA_FILE_NAME_TRAIN_VALID)
    FUNC_NAME_DICT_TRAIN_VALID = get_f_dict_fcg(F_PATH_TRAIN_VALID)

    # loading func embeddings
    print("start reading embeddings")
    embeddings_path = "/data/wangyongpan/paper/reuse_detection/code/reuse_minifcg/embeddings/hxl_embeddings_torch_best.json"
    with open(embeddings_path, "r") as f:
        func_embeddings = json.load(f)
    #F_PATH_TEST = get_f_name(DATA_FILE_NAME_TEST)
    #FUNC_NAME_DICT_TEST = get_f_dict(F_PATH_TEST)

    print("start reading graph")
    Gs_train_valid, classes_train_valid = read_graph_fcg(
        F_PATH_TRAIN_VALID, FUNC_NAME_DICT_TRAIN_VALID, func_embeddings, NODE_FEATURE_DIM)
    print("train : valid : test  ---- 8:1:1")
    print("{} graphs, {} functions".format(
        len(Gs_train_valid), len(classes_train_valid)))

    perm = np.random.permutation(len(classes_train_valid))
    Gs_train, classes_train, Gs_valid, classes_valid, Gs_test, classes_test =\
        partition_data(Gs_train_valid, classes_train_valid, [0.8, 0.1, 0.1], perm)
    print("Train: {} graphs, {} functions".format(
        len(Gs_train), len(classes_train)))
    print("Valid: {} graphs, {} functions".format(
        len(Gs_valid), len(classes_valid)))
    print("Test: {} graphs, {} functions".format(
        len(Gs_test), len(classes_test)))

    # print("Test")
    # Gs_test, classes_test = read_graph(
    #     F_PATH_TEST, FUNC_NAME_DICT_TEST, NODE_FEATURE_DIM)
    # print("{} graphs, {} functions".format(len(Gs_test), len(classes_test)))
    # Gs_test, classes_test = partition_data(
    #     Gs_test, classes_test, [1], list(range(len(classes_test))))

    trainSet = dataset(Gs_train, classes_train, BATCH_SIZE,
                       NEG_BATCH_SIZE, neg_batch_flag=False, train=True)
    validSet = dataset(Gs_valid, classes_valid, BATCH_SIZE,
                       NEG_BATCH_SIZE, neg_batch_flag=False, train=True)
    testSet = dataset(Gs_test, classes_test, BATCH_SIZE,
                      NEG_BATCH_SIZE, neg_batch_flag=False, train=True)

    trainLoader = DataLoader(
        trainSet, batch_size=1, shuffle=False, num_workers=WORKERS, pin_memory=True)
    validLoader = DataLoader(
        validSet, batch_size=1, shuffle=False, num_workers=WORKERS, pin_memory=True)

    testLoader = DataLoader(testSet, batch_size=1,
                            shuffle=False, num_workers=WORKERS, pin_memory=True)

    net = graphnn(NODE_FEATURE_DIM, EMBED_DIM, OUTPUT_DIM,
                  EMBED_DEPTH, ITERATION_LEVEL)
    # net = torch.load("saved_model/gnn-best.pt")

    net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad()

    time_start = time.time()
    loss_train, model_auc, _, _, _ = contra_loss_show(net, trainLoader, DEVICE)
    print('Initial train: loss:%.4f\tauc:%.4f\t@%s\ttime lapsed:\t%.2f s' %
          (np.mean(loss_train), model_auc, datetime.now(), time.time() - time_start))

    time_start = time.time()
    loss_train, model_auc, _, _, _ = contra_loss_show(net, validLoader, DEVICE)
    print('Initial valid: loss:%.4f\tauc:%.4f\t@%s\ttime lapsed:\t%.2f s' %
          (np.mean(loss_train), model_auc, datetime.now(), time.time() - time_start))

    time_start = time.time()
    loss_train, model_auc, _, _, _ = contra_loss_show(net, testLoader, DEVICE)
    print('Initial test: loss:%.4f\tauc:%.4f\t@%s\ttime lapsed:\t%.2f s' %
          (np.mean(loss_train), model_auc, datetime.now(), time.time() - time_start))

    train_loss = []
    time_start = time.time()

    best_loss = 99999999
    patience = 5
    best_auc = 0

    for i in range(1, MAX_EPOCH+1):
        trainSet.shuffle()
        trainLoader = DataLoader(
            trainSet, batch_size=1, shuffle=False, num_workers=WORKERS)
        loss_val = []
        tot_cos = []
        tot_truth = []
        time_start = time.time()
        net.train()
        p_n_gap = []
        tq = tqdm.tqdm(trainLoader)
        for batch_id, (X1, X2, X3, m1, m2, m3) in enumerate(tq, 1):
            X1, X2, X3, m1, m2, m3 = X1[0], X2[0], X3[0], m1[0], m2[0], m3[0]
            if 'gpu' in DEVICE:
                X1, X2, X3, m1, m2, m3 = X1.cuda(non_blocking=True), X2.cuda(non_blocking=True), X3.cuda(
                    non_blocking=True),  m1.cuda(non_blocking=True), m2.cuda(non_blocking=True), m3.cuda(non_blocking=True)
            # else:
                # X1, X2, X3, m1, m2, m3 = X1, Variable(X2), Variable(
                # X3),  Variable(m1), Variable(m2), Variable(m3)
            loss, cos_p, cos_n = net.forward(X1, X2, X3, m1, m2, m3)
            cos_p = cos_p.cpu().detach().numpy()
            cos_n = cos_n.cpu().detach().numpy()
            p_n_gap.append(np.mean(cos_p - cos_n))
            cos_p = list(cos_p)
            cos_n = list(cos_n)
            tot_cos += cos_p
            tot_truth += [1]*len(cos_p)
            tot_cos += cos_n
            tot_truth += [-1]*len(cos_n)
            loss_val.append(loss.item())
            tq.set_description("Train:EPOCH" + str(i) + "[" + str(loss.item()) + "]")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        cos = np.array(tot_cos)
        truth = np.array(tot_truth)

        fpr, tpr, thres = roc_curve(truth, (1+cos)/2)
        model_auc = auc(fpr, tpr)
        print('Epoch-Train: [%d]\tloss:%.4f\tp_n_gap:%.4f\tauc:%.4f\t@%s\ttime lapsed:\t%.2f s' %
              (i, np.mean(loss_val), np.mean(p_n_gap), model_auc, datetime.now(), time.time() - time_start))
        if i % SHOW_FREQ == 0:
            net.eval()
            with torch.no_grad():
                time_start = time.time()
                loss_val, model_auc, _, _, thre1 = contra_loss_show(net, validLoader, DEVICE)
                print('Valid: [%d]\tloss:%.4f\tauc:%.4f\t@%s\ttime lapsed:\t%.2f s' %
                      (i, np.mean(loss_val), model_auc, datetime.now(), time.time() - time_start))

                time_start = time.time()
                loss_test, test_model_auc, _,  _, thre2 = contra_loss_show(net, testLoader, DEVICE)
                print("#"*70)
                print('Test: [%d]\tloss:%.4f\tauc:%.4f\t@%s\ttime lapsed:\t%.2f s' %
                      (i, np.mean(loss_test), test_model_auc, datetime.now(), time.time() - time_start))
                print("#"*70)
                time_start = time.time()
                train_loss.append(np.mean(loss_test))

                # if model_auc > best_auc:
                #     torch.save(net, SAVE_PATH + "/model-inter-best.pt")
                patience -= 1
                if np.mean(loss_val) < best_loss:
                    torch.save(net, SAVE_PATH + "/fcg_gnn-best-0.001-1-500.pt")
                    patience = 5

        if i % SAVE_FREQ == 0:
            torch.save(net, SAVE_PATH +
                       '/fcg-gnn-0.001-1-500-' + str(i+1) + ".pt")
        if patience <= 0:
            break

    # learning_rate = learning_rate * 0.95

    _, auc1, fpr1, tpr1, thres1 = contra_loss_show(net, testLoader, DEVICE)
    print("testing auc = {0} @ {1}".format(auc1, datetime.now()))

    print(auc1)
    print(max((1 - fpr1 + tpr1) / 2))
    index = np.argmax(1 - fpr1 + tpr1)
    print("index:", index)
    print("fpr", fpr1[index])
    print("tpr", tpr1[index])
    print(thres1[index])

    _, auc2, fpr2, tpr2, thres2 = contra_loss_show(net, validLoader, DEVICE)
    print("validation auc = {0} @ {1}".format(auc2, datetime.now()))

    print(auc2)
    print(max((1 - fpr2 + tpr2) / 2))
    index = np.argmax((1 - fpr2 + tpr2) / 2)
    print("index:", index)
    print("fpr", fpr2[index])
    print("tpr", tpr2[index])
    print(thres2[index])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('ROC CURVE')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr1, tpr1, color='b')
    plt.plot(fpr1, 1 - fpr1 + tpr1, color='b')
    plt.plot(fpr2, tpr2, color='r')
    plt.plot(fpr2, 1 - fpr2 + tpr2, color='r')
    #     plt.plot([0, 1], [0, 1], color='m', linestyle='--')
    plt.savefig('analog_auc_depth5.png')

    # with open('train_loss.txt', 'w') as f:
    #     f.write(str(train_loss))

    # from core_fedora_embeddings import *
    # with Pool(10) as p:
    #     p.starmap(core_fedora_embedding, [(i, True) for i in range(10)])
    # valid_embedding_pairs(True)

