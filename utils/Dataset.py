import os
import torch
import numpy as np
import Constants as C
import scipy.sparse as sp
from utils.cal_pairwise import read_interaction

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T

import random
from collections import Counter

class Dataset(object):
    def __init__(self):
        self.user_num = C.USER_NUMBER
        self.item_num = C.ITEM_NUMBER
        self.directory_path = './data/{dataset}/'.format(dataset=C.DATASET)

        self.training_user, self.training_times = self.read_training_data()
        self.tuning_user, self.tuning_times = self.read_tuning_data()
        self.test_user, self.test_times = self.read_test_data()

        self.user_data, self.user_valid = self.read_data()
        self.ui_adj = self.load_adjacent_matrix()

    def parse(self, data):
        user_traj, user_times = [[] for i in range(self.user_num)], [[] for i in range(self.user_num)]
        for eachline in data:
            uid, lid, times = eachline.strip().split()
            uid, lid, times = int(uid), int(lid), int(times)
            try:
                user_traj[uid].append(lid + 1)
                user_times[uid].append(times + 1)
            except Exception as e:
                print(uid, len(user_traj))
        return user_traj, user_times

    def read_data(self):
        user_data, user_valid = [], []

        for i in range(self.user_num):
            user_data.append((i, self.training_user[i], self.tuning_times[i], self.tuning_user[i],), )

            valid_input = self.training_user[i].copy()
            valid_input.extend(self.tuning_user[i])
            valid_times = self.training_times[i].copy()
            valid_times.extend(self.tuning_times[i])
            user_valid.append((i, valid_input, valid_times, self.test_user[i],), )

        return user_data, user_valid

    def read_training_data(self):
        train_file = '{dataset}_train.txt'.format(dataset=C.DATASET)
        return self.parse(open(self.directory_path + train_file, 'r').readlines())

    def read_tuning_data(self):
        tune_file = '{dataset}_tune.txt'.format(dataset=C.DATASET)
        return self.parse(open(self.directory_path + tune_file, 'r').readlines())

    def read_test_data(self):
        test_file = '{dataset}_test.txt'.format(dataset=C.DATASET)
        return self.parse(open(self.directory_path + test_file, 'r').readlines())

    def load_adjacent_matrix(self):
        directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
        train_file = 'item_matrix.npy'
        if not os.path.exists(directory_path + train_file):
            print('item_matrix is not found, generating ...')
            read_interaction()
        print('Loading ', directory_path + train_file, '...')
        ui_adj = np.load(directory_path + train_file)
        ui_adj = sp.csr_matrix(ui_adj)
        print('Computing adj matrix ...')
        ui_adj = torch.tensor(self.normalize_graph_mat(ui_adj).toarray(), device='cuda:0')
        return ui_adj

    def normalize_graph_mat(self, adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        rowsum[rowsum == 0] = 1e-9
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def get_in_degree(self):
        in_degree = np.zeros(C.ITEM_NUMBER + 1)
        directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
        if os.path.exists(directory_path + 'in_degree.npy'):
            in_degree = np.load(directory_path + 'in_degree.npy')
        else:
            all_train_data = open(directory_path + '{dataset}_train.txt'.format(dataset=C.DATASET), 'r').readlines()
            all_tune_data = open(directory_path + '{dataset}_tune.txt'.format(dataset=C.DATASET), 'r').readlines()
            all_train_data.extend(all_tune_data)
            for eachline in all_train_data:
                uid, lid, times = eachline.strip().split()
                uid, lid, times = int(uid), int(lid), int(times)
                in_degree[lid + 1] += 1
            np.save(directory_path + 'in_degree.npy', in_degree)
        return torch.tensor(in_degree, device='cuda:0', dtype=torch.float16)

    def paddingLong2D(self, insts):
        """ Pad the instance to the max seq length in batch. """
        # max_len = 400
        max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([
            inst[:max_len] + [C.PAD] * (max_len - len(inst))
            for inst in insts])
        return torch.tensor(batch_seq, dtype=torch.long)

################################
# Additional Code for Recent + Random Sampling
    def recent_random_sample(self, seq, k_recent=10, max_len=50):
        if len(seq) <= max_len:
            return seq
        recent = seq[-k_recent:]
        pool = seq[:-k_recent]
        k_sample = max_len - k_recent
        if len(pool) <= k_sample:
            return pool + recent
        sampled = random.sample(pool, k_sample)
        return sampled + recent
################################

    def padding2D(self, insts):
        """ Pad the instance to the max seq length in batch. """
        # max_len = 400
        max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([
            inst[:max_len] + [C.PAD] * (max_len - len(inst))
            for inst in insts])
        return torch.tensor(batch_seq, dtype=torch.float32)

    def user_fn(self, insts):
        """ Collate function, as required by PyTorch. """
        (useridx, event_type, event_time, test_label) = list(zip(*insts))

################################
# Additional Code for Sequence Trimming
        ## Basic
        # max_len = 50
        # event_type = [et if len(et) <= max_len else et[-max_len:] for et in event_type]
        # event_time = [et if len(et) <= max_len else et[-max_len:] for et in event_time]
        # test_label = [tl if len(tl) <= max_len else tl[-max_len:] for tl in test_label]

        ## Recent + Random
        # max_len = 50
        # k_recent = 30

        # trimmed = [(
        #     self.recent_random_sample(et, k_recent, max_len),
        #     self.recent_random_sample(tm, k_recent, max_len),
        #     self.recent_random_sample(tl, k_recent, max_len)
        # ) for et, tm, tl in zip(event_type, event_time, test_label)]

        # event_type, event_time, test_label = zip(*trimmed)
################################

        useridx = torch.tensor(useridx, device='cuda:0')
################################
# Additional Code for time-weighted embeddings
        # max_len = max(
        #     max(len(seq) for seq in event_type),
        #     max(len(seq) for seq in event_time),
        #     max(len(seq) for seq in test_label)
        # )

        # def pad(seq_list, dtype=torch.long):
        #     return torch.tensor([
        #         s[:max_len] + [C.PAD] * (max_len - len(s)) for s in seq_list
        #     ], dtype=dtype)

        # event_type = pad(event_type, dtype=torch.long)
        # event_time = pad(event_time, dtype=torch.long)
        # test_label = pad(test_label, dtype=torch.long)
################################

        ## Original From CaDRec
        event_type = self.paddingLong2D(event_type)
        event_time = self.paddingLong2D(event_time)
        test_label = self.paddingLong2D(test_label)

################################
# Additional Code for time-weighted embeddings
        # decay_rate = 1e-3
       
        # valid_mask = (event_type != C.PAD)

        # masked_event_time = event_time.float().clone()
        # masked_event_time[~valid_mask] = 0
        # now = torch.max(masked_event_time).item()

        # time_weights = torch.exp(-decay_rate * (now - masked_event_time))
        # time_weights = time_weights * valid_mask

        # return useridx, event_type, event_time, test_label, time_weights
################################
        return useridx, event_type, event_time, test_label

    def get_user_dl(self, batch_size):
        user_dl = torch.utils.data.DataLoader(
            self.user_data,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=self.user_fn,
            shuffle=True
        )
        return user_dl

    def get_user_valid_dl(self, batch_size):
        user_valid_dl = torch.utils.data.DataLoader(
            self.user_valid,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=self.user_fn,
            shuffle=True
        )
        return user_valid_dl

#####################################
# Contrastive Learning Algorithm - Hard Negative Mining
#     def all_event_sequences(self):
#         return [seq for seq in self.training_user if len(seq) > 0]

# def get_item_popularity_bins(event_type_sequences, num_bins=20):
#     all_items = [item for seq in event_type_sequences for item in seq]
#     item_freq = Counter(all_items)

#     item_ids = list(item_freq.keys())
#     freqs = np.array([item_freq[i] for i in item_ids])
#     bins = np.quantile(freqs, np.linspace(0, 1, num_bins + 1))

#     item_to_bin = {}
#     for item in item_ids:
#         for i in range(num_bins):
#             if bins[i] <= item_freq[item] <= bins[i+1]:
#                 item_to_bin[item] = i
#                 break
#     return item_to_bin
#####################################