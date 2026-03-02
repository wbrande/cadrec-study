import argparse

import numpy as np
import time
import math

import torch
import torch.optim as optim

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T

import optuna

import Constants as C
from utils import Utils, metric

from utils.Dataset import Dataset as dataset
from Model.Models import Model
from tqdm import tqdm
import shutil
import os

################################
from torch.cuda.amp import autocast, GradScaler
from utils.contrastive import info_nce_loss, info_nce_with_hard_negatives, align_uniform_loss
from utils.Dataset import get_item_popularity_bins
import torch.nn.functional as F

# Additional code for testing.
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

################################

# def train_epoch(model, user_dl, adj_matrix, pop_encoding, optimizer, opt):
# def train_epoch(model, user_dl, adj_matrix, pop_encoding, optimizer, opt, scaler):
# def train_epoch(model, user_dl, adj_matrix, pop_encoding, optimizer, opt, item_to_bin): # Contrastive Learning Version
def train_epoch(model, teacher_model, user_dl, adj_matrix, pop_encoding, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    [pre, rec, map_, ndcg] = [[[] for i in range(4)] for j in range(4)]
    for batch in tqdm(user_dl, mininterval=2, desc='  - (Training)   ', leave=False):
        optimizer.zero_grad()

        """ prepare data """
        user_idx, event_type, event_time, test_label = map(lambda x: x.to(opt.device), batch)

        """ forward """
        prediction, user_embeddings, pop_vector = model(user_idx, event_type, event_time, adj_matrix, pop_encoding, evaluation=False)

################################
# Modified to include time_weights
        # """ prepare data """
        # user_idx, event_type, event_time, test_label, time_weights = map(lambda x: x.to(opt.device), batch)

        # """ forward """
        # prediction, user_embeddings, pop_vector = model(user_idx, event_type, event_time, adj_matrix, pop_encoding, time_weights, evaluation=False)

# Additional Code for Add Mixed Precision
        # with autocast():
        #     prediction, user_embeddings, pop_vector = model(user_idx, event_type, event_time, adj_matrix, pop_encoding, evaluation=False)
        #     loss = Utils.type_loss(prediction, event_type, event_time, test_label, opt)

        #     eta = 0.1 if C.DATASET not in C.eta_dict else C.eta_dict[C.DATASET]
        #     if C.ABLATION != 'w/oNorm':
        #         loss += Utils.l2_reg_loss(eta, model, event_type)
################################

        """ compute metric """
        metric.pre_rec_top(pre, rec, map_, ndcg, prediction, test_label, event_type)

        ## Original from CaDRec
        """ backward """
        loss = Utils.type_loss(prediction, event_type, event_time, test_label, opt)

################################
# Addtional Code for Knowledge Distillation Strategy
        loss_gt = Utils.type_loss(prediction, event_type, event_time, test_label, opt)
        if teacher_model is not None:
            with torch.no_grad():
                teacher_prediction, _, _ = teacher_model(user_idx, event_type, event_time, adj_matrix, pop_encoding, evaluation=False)
            loss_kd = Utils.distillation_loss(prediction, teacher_prediction, temperature=2.0)
            alpha = 0.7
            loss = alpha * loss_gt + (1 - alpha) * loss_kd
        else:
            loss = loss_gt
################################

        eta = 0.1 if C.DATASET not in C.eta_dict else C.eta_dict[C.DATASET]
        if C.ABLATION != 'w/oNorm':
            loss += Utils.l2_reg_loss(eta, model, event_type)

####################################
# Contrastive Learning -> shows small improvements in R and NDCG
        # Applying contrastive learning to loss

        # loss_cl = info_nce_loss(model.cl_view1, model.cl_view2)
        # loss_cl_user = info_nce_loss(model.user_cl_view1, model.user_cl_view2)
        
        # # loss_cl = info_nce_with_hard_negatives(
        # #     model.cl_view1, model.cl_view2, item_ids=event_type[:, 0], item_to_bin=item_to_bin
        # # )

        # # loss_cl = align_uniform_loss(model.cl_view1, model.cl_view2, lambda_uniform=1.0)
        # # loss_cl_user = align_uniform_loss(model.user_cl_view1, model.user_cl_view2, lambda_uniform=1.0)
        
        # lambda_cl = 0.1
        # lambda_user_cl = 0.1

        # loss += lambda_cl * loss_cl
        # loss += lambda_user_cl * loss_cl_user

        # # For Multi-View Contrastive Head
        # # loss_cl_layer = info_nce_loss(model.view_layer1, model.view_layerN)
        # # lambda_layer_cl = 0.1
        # # loss += lambda_layer_cl * loss_cl_layer

        # # # Check value similarity value
        # # with torch.no_grad():
        # #     sim = F.cosine_similarity(model.cl_view1, model.cl_view2, dim=-1)
        # #     sim_user = F.cosine_similarity(model.user_cl_view1, model.user_cl_view2, dim=-1)
        # #     print("Average Cosine Similarity (CL Views):", sim.mean().item(), "User CL avg sim:", sim_user.mean().item())

####################################
# Counter Factual Regularization
        # loss_cf = F.mse_loss(model.real_output, model.cf_output)
        # #alternate
        # # cosine_sim = F.cosine_similarity(model.real_output, model.cf_output, dim=1)
        # # loss_cf = 1 - cosine_sim.mean()

        # lambda_cf = 0.1
        # loss += lambda_cf * loss_cf

        # aux_accs = []

# With Contrastive Learning And Auxiliary Classification
        # B = model.aux_logits_real.size(0)
        # real_labels = torch.zeros(B, dtype=torch.long, device=model.aux_logits_real.device)
        # cf_labels = torch.ones(B, dtype=torch.long, device=model.aux_logits_cf.device)

        # loss_aux_real = F.cross_entropy(model.aux_logits_real, real_labels)
        # loss_aux_cf = F.cross_entropy(model.aux_logits_cf, cf_labels)
        # loss_aux = 0.5 * (loss_aux_real + loss_aux_cf)

        # for batch in user_dl:
        #     aux_real_logits = model.aux_logits_real
        #     aux_cf_logits = model.aux_logits_cf

        #     aux_real_probs = F.softmax(aux_real_logits, dim=1)
        #     aux_cf_probs = F.softmax(aux_cf_logits, dim=1)

        #     entropy_real = -torch.sum(aux_real_probs * torch.log(aux_real_probs + 1e-8), dim=1).mean()
        #     entropy_cf = -torch.sum(aux_cf_probs * torch.log(aux_cf_probs + 1e-8), dim=1).mean()

        #     entropy_reg_weight = 0.05
        #     loss_entropy = entropy_reg_weight * (entropy_real + entropy_cf)

        #     loss -= loss_entropy

        #     logits = torch.cat([aux_real_logits, aux_cf_logits], dim=0)
        #     labels = torch.cat([
        #         torch.zeros(B, dtype=torch.long, device=logits.device),
        #         torch.ones(B, dtype=torch.long, device=logits.device)
        #     ], dim=0)

        #     loss_aux = F.cross_entropy(logits, labels)

        #     with torch.no_grad():
        #         preds = logits.argmax(dim=1)
        #         aux_acc = (preds == labels).float().mean()

        # lambda_aux = 0.2
        # loss += lambda_aux * loss_aux

####################################

        loss.backward(retain_graph=True)
        """ update parameters """
        optimizer.step()

################################
# Additional Code for Auxiliary Classification
        # aux_accs.append(aux_acc.item())
        # avg_aux_acc = sum(aux_accs) / len(aux_accs)
        # print(f' aux acc: {avg_aux_acc:.4f}')

# Additional Code for Add Mixed Precision
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
################################

    results_np = map(lambda x: [np.around(np.mean(i), 5) for i in x], [pre, rec, map_, ndcg])
    return results_np


def eval_epoch(model, user_valid_dl, adj_matrix, in_degree, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()
    [pre, rec, map_, ndcg] = [[[] for i in range(4)] for j in range(4)]
    with torch.no_grad():
        for batch in tqdm(user_valid_dl, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare test data """
            user_idx, event_type, event_time, test_label = map(lambda x: x.to(opt.device), batch)

            """ forward """
            prediction, _, _ = model(user_idx, event_type, event_time, adj_matrix, in_degree)  # X = (UY+Z) ^ T
################################
# Modified to Include time_weights
            # """ prepare test data """
            # user_idx, event_type, event_time, test_label, time_weights = map(lambda x: x.to(opt.device), batch)

            # """ forward """
            # prediction, _, _ = model(user_idx, event_type, event_time, adj_matrix, in_degree, time_weights)  # X = (UY+Z) ^ T
################################

            # valid_user_embeddings[user_idx] = users_embeddings

            """ compute metric """
            metric.pre_rec_top(pre, rec, map_, ndcg, prediction, test_label, event_type)

    results_np = map(lambda x: [np.around(np.mean(i), 5) for i in x], [pre, rec, map_, ndcg])
    return results_np


# def train(model, data, optimizer, scheduler, opt):
# def train(model, data, optimizer, scheduler, opt, item_to_bin): # Contrastive Learning Version
def train(model, teacher_model, data, optimizer, scheduler, opt):
    """ Start training. """
################################
# Addtional Code for Add Mixed Precision
    # scaler = GradScaler()

################################
# Additional Code for Knowledge Distillation strategy
    use_teacher_epochs = opt.epoch // 2
################################

    best_ = [np.zeros(4) for i in range(4)]
    (user_valid_dl, user_dl, adj_matrix, pop_encoding) = data
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i + 1, ']')

        np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
        start = time.time()
        # [pre, rec, map_, ndcg] = train_epoch(model, user_dl, adj_matrix, pop_encoding, optimizer, opt)
        # [pre, rec, map_, ndcg] = train_epoch(model, user_dl, adj_matrix, pop_encoding, optimizer, opt, scaler)
        # [pre, rec, map_, ndcg] = train_epoch(model, user_dl, adj_matrix, pop_encoding, optimizer, opt, item_to_bin) # Contrastive Learning Version
################################
# Additional Code for Knowledge Distillation strategy
        use_teacher = (epoch_i < use_teacher_epochs)
        [pre, rec, map_, ndcg] = train_epoch(model, teacher_model if use_teacher else None, user_dl, adj_matrix, pop_encoding, optimizer, opt)
################################
        print('\r(Training)  P@k:{pre},    R@k:{rec}, \n'
              '(Training)map@k:{map_}, ndcg@k:{ndcg}, '
              'elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60, pre=pre, rec=rec, map_=map_, ndcg=ndcg))

        # Only evaluate model every 5th epoch or last epoch
        if (epoch_i+1) % 5 == 0 or epoch_i == opt.epoch-1:
            start = time.time()
            [pre, rec, map_, ndcg] = eval_epoch(model, user_valid_dl, adj_matrix, pop_encoding, opt)
            print('\r(Test)  P@k:{pre},    R@k:{rec}, \n'
                '(Test)map@k:{map_}, ndcg@k:{ndcg}, '
                'elapse:{elapse:3.3f} min'
                .format(elapse=(time.time() - start) / 60, pre=pre, rec=rec, map_=map_, ndcg=ndcg))

            scheduler.step()
            if best_[-1][1] < ndcg[1]:
                best_ = [pre, rec, map_, ndcg]
################################
# Additional Code for Knowledge Distillation strategy
                best_model = model.state_dict()
################################

    print('\n', '-' * 40, 'BEST', '-' * 40)
    print('k', C.Ks)
    print('\rP@k:{pre},    R@k:{rec}, \n'
          '(Best)map@k:{map_}, ndcg@k:{ndcg}'
          .format(pre=best_[0], rec=best_[1], map_=best_[2], ndcg=best_[3]))
    print('-' * 40, 'BEST', '-' * 40, '\n')

################################
# Additional Code for Knowledge Distillation strategy
    if opt.train_teacher:
        torch.save(best_model, f'cadrec_teacher_{C.DATASET}.pt')
################################

    return best_[-1][1]


def get_user_embeddings(model, user_dl, opt):
    """ Epoch operation in training phase. """

    valid_user_embeddings = torch.zeros((C.USER_NUMBER, opt.d_model), device='cuda:0')

    for batch in tqdm(user_dl, mininterval=2, desc='  - (Computing user embeddings)   ', leave=False):
        """ prepare data """
        user_idx, event_type, event_time, test_label = map(lambda x: x.to(opt.device), batch)

        """ forward """
        prediction, users_embeddings = model(event_type)  # X = (UY+Z) ^ Tc
        valid_user_embeddings[user_idx] = users_embeddings

    return valid_user_embeddings


def pop_enc(in_degree, d_model):
    """
    Input: batch*seq_len.
    Output: batch*seq_len*d_model.
    """
    pop_vec = torch.tensor(
        [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
        device=torch.device('cuda'), dtype=torch.float16)

    result = in_degree.unsqueeze(-1) / pop_vec
    result[:, 0::2] = torch.sin(result[:, 0::2])
    result[:, 1::2] = torch.cos(result[:, 1::2])
    return result


def main(trial):
    """ Main function. """
    parser = argparse.ArgumentParser()
################################
# Additional Code for Knowledge Distillation strategy
    parser.add_argument('--train_teacher', action='store_true', help='If set, trains the teacher model and saves it')
################################
    opt = parser.parse_args()
    opt.device = torch.device('cuda')

    # # # optuna setting for tuning hyperparameters
    # opt.n_layers = trial.suggest_int('n_layers', 2, 2)
    # opt.d_inner_hid = trial.suggest_int('n_hidden', 512, 1024, 128)
    # opt.d_k = trial.suggest_int('d_k', 512, 1024, 128)
    # opt.d_v = trial.suggest_int('d_v', 512, 1024, 128)
    # opt.n_head = trial.suggest_int('n_head', 1, 5, 1)
    # # opt.d_rnn = trial.suggest_int('d_rnn', 128, 512, 128)
    # opt.d_model = trial.suggest_int('d_model', 128, 1024, 128)
    # opt.dropout = trial.suggest_uniform('dropout_rate', 0.5, 0.7)
    # opt.smooth = trial.suggest_uniform('smooth', 1e-2, 1e-1)
    # opt.lr = trial.suggest_uniform('learning_rate', 0.00008, 0.0002)

    DATASET = C.DATASET
    if DATASET == 'Foursquare':
        beta, lambda_ = 0.3256, 0.4413  # 0.4, 0.4   # 0.4, 0.5  # 0.35, 0.5  # 0.5, 1
    elif DATASET == 'Gowalla':
        beta, lambda_ = 1.5, 4  # 0.38, 1  # 1.5, 4
    elif DATASET == 'Yelp2018':
        beta, lambda_ = 2.2977, 7.0342  # 1.8, 4  # 0.35, 1  # 1, 4
    elif DATASET == 'douban-book':
        beta, lambda_ = 0.9802, 0.7473
    elif DATASET == 'ml-1M':
        beta, lambda_ = 0.4645, 0.4098  # 0.9, 1
    else:
        beta, lambda_ = 0.5, 1
    opt.beta, opt.lambda_ = beta, lambda_

    opt.lr = 0.01
    opt.epoch = 30
    opt.n_layers = 1
    opt.batch_size = 32
    opt.dropout = 0.5
    opt.smooth = 0.03

    if DATASET == 'Foursquare': opt.d_model, opt.n_head = 768, 1
    elif DATASET == 'Gowalla': opt.d_model, opt.n_head = 512, 1
    elif DATASET == 'douban-book': opt.d_model, opt.n_head = 512, 1
    elif DATASET == 'Yelp2018': opt.d_model, opt.n_head = 512, 1
    elif DATASET == 'ml-1M': opt.d_model, opt.n_head = 512, 2
    else: opt.d_model, opt.n_head = 512, 1

################################
# Additional Code for Knowledge Distillation strategy
    if not opt.train_teacher:
        teacher_d_model = opt.d_model
        teacher_n_head = opt.n_head

        opt.d_model = 128
        opt.n_head = 1
################################

    print('[Info] parameters: {}'.format(opt))
    num_types = C.ITEM_NUMBER
    num_user = C.USER_NUMBER

    ## Original from CaDRec
    """ prepare model """
    model = Model(
        num_types=num_types,
        d_model=opt.d_model,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        device=opt.device
    )
    model = model.cuda()

################################
# Additional Code for Knowledge Distillation Strategy
    teacher_model = None
    if not opt.train_teacher:
        teacher_model = Model(
            num_types=num_types,
            d_model=teacher_d_model,
            n_layers=opt.n_layers,
            n_head=teacher_n_head,
            dropout=opt.dropout,
            device=opt.device
        )
        teacher_ckpt = f'cadrec_teacher_{C.DATASET}.pt'

        teacher_model.load_state_dict(torch.load(teacher_ckpt))
        teacher_model = teacher_model.cuda()
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
################################

    """ loading data"""
    ds = dataset()
    print('[Info] Loading data...')
    user_dl = ds.get_user_dl(opt.batch_size)
    user_valid_dl = ds.get_user_valid_dl(opt.batch_size)
    in_degree = ds.get_in_degree()
    pop_encoding = pop_enc(in_degree, opt.d_model)
    adj_matrix = ds.ui_adj

    data = (user_valid_dl, user_dl, adj_matrix, pop_encoding)

#####################################
# Contrastive Learning Algorithm - Hard Negative Mining
    # item_to_bin = get_item_popularity_bins(ds.all_event_sequences())

#####################################

    """ optimizer and scheduler """
    parameters = [
                  {'params': model.parameters(), 'lr': opt.lr},
                  ]
    optimizer = torch.optim.Adam(parameters)  # , weight_decay=0.01
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    # result = train(model, data, optimizer, scheduler, opt)
    # result = train(model, data, optimizer, scheduler, opt, item_to_bin) # Contrastive Learning Version
################################
# Additional Code for Knowledge Distillation strategy
    result = train(model, teacher_model, data, optimizer, scheduler, opt)
################################
    return result


if __name__ == '__main__':
    main(None)

    # if you want to tune hyperparameters, please comment out main(None) and use the following code
    # study = optuna.create_study(direction="maximize")
    # n_trials = 100
    # study.optimize(main, n_trials=n_trials)



