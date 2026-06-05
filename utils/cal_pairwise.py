import numpy as np
import torch

import sys
sys.path.append("..")
import Constants as C
import os

import time


def read_interaction(train_data=None, directory_path=None):

    # for temporal feature
    start_time = time.time()
    # print(start_time)
    if directory_path is None:
        directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
    if train_data is None:
        train_data = open(directory_path + '{dataset}_train.txt'.format(dataset=C.DATASET), 'r').readlines()
        train_data.extend(open(directory_path + '{dataset}_tune.txt'.format(dataset=C.DATASET), 'r').readlines())
    count = 0

    interaction_matrix = torch.zeros((C.USER_NUMBER, C.ITEM_NUMBER), device='cuda:0')
    item_matrix = torch.zeros((C.ITEM_NUMBER, C.ITEM_NUMBER), device='cuda:0')

    # print(interaction_matrix.size())
    for eachline in train_data:
        uid, lid, timestamp = eachline.strip().split()
        uid, lid, timestamp = int(uid), int(lid), int(timestamp)
        if C.DATASET == 'Yelp2018':
            lid = lid -1
        
        # print(uid, lid)
        interaction_matrix[uid][lid] = 1
        count += 1
        if count % 500000 == 0:
            print(count, time.time()-start_time)

    for i in range(C.USER_NUMBER):
        nwhere = torch.where(interaction_matrix[i]==1)[0]
        for j in nwhere:
            item_matrix[j][nwhere] = 1

    # print(nwhere)
    # print(ITEM_matrix)
    np.save(directory_path + 'item_matrix.npy', item_matrix.cpu().numpy())
