from utils import Utils
import torch.nn
import torch.nn as nn
from utils.Utils import *
from Model.hGCN import hGCNEncoder

from Model.lightGCN import LightGCNLayer
from utils.contrastive import random_event_mask

class Encoder(nn.Module):
    def __init__(
            self,
            num_types, d_model, n_layers, n_head, dropout):
        super().__init__()
        self.d_model = d_model

        ## Original from CaDRec
        self.layer_stack = nn.ModuleList([
            hGCNEncoder(d_model, n_head)
            for _ in range(n_layers)])

################################
# Additional Code for Swap to LightGCN
        # self.layer_stack = nn.ModuleList([
        #     LightGCNLayer(d_model)
        #     for _ in range(n_layers)
        # ])
################################

################################
# Additional Code for Session-Level Attention
        # self.session_attn = nn.Linear(d_model, 1)
################################

    def forward(self, user_id, event_type, enc_output, user_output, adjacent_matrix):
        """ Encode event sequences via masked self-attention. """

        # get individual adj
        adj = torch.zeros((event_type.size(0), event_type.size(1), event_type.size(1)), device='cuda:0')
        ## Original from CaDRec
        for i, e in enumerate(event_type):
            # the slicing operation
            adj[i] = adjacent_matrix[e - 1, :][:, e - 1]
            # performance can be enhanced by adding the element in the diagonal of the normalized adjacency matrix.
            adj[i] += adjacent_matrix[e - 1, e - 1]

################################
# Additional Code for Adjacency Matrix Sampling
        # for i, e in enumerate(event_type):
        #     sub_adj = adjacent_matrix[e - 1, :][:, e - 1]

        #     k = 10
        #     topk_values, topk_indices = torch.topk(sub_adj, k=min(k, sub_adj.size(-1)), dim=-1)

        #     sparse_sub_adj = torch.zeros_like(sub_adj)
        #     sparse_sub_adj.scatter_(-1, topk_indices, topk_values)

        #     sparse_sub_adj += torch.diag(sub_adj.diagonal())
        #     adj[i] = sparse_sub_adj
################################

        ## Original from CaDRec
        for i, enc_layer in enumerate(self.layer_stack):
            residual = enc_output
            enc_output = enc_layer(enc_output, user_output, adj, event_type)
            if C.DATASET in {'douban-book'}:
                enc_output += residual

################################
# Additional Code for Swap to LightGCN
        # for layer in self.layer_stack:
        #     enc_output = layer(enc_output, adj)
################################

################################
# Additional Code for Session-Level Attention
        # attn_logits = self.session_attn(enc_output).squeeze(-1)
        # attn_logits = attn_logits.masked_fill(event_type == C.PAD, float('-inf'))

        # attn_weights = torch.softmax(attn_logits, dim=-1)
        # attn_output = torch.sum(enc_output * attn_weights.unsqueeze(-1), dim=1)

        # output_mean = enc_output.mean(dim=1)
        # session_output = 0.5 * attn_output + 0.5 * output_mean

        # return session_output
################################

###########################################
# Contrastive Learning Algorithm with Multi-View Contrastive Head
        # all_layer_outputs = []
        # for i, enc_layer in enumerate(self.layer_stack):
        #     residual = enc_output
        #     enc_output = enc_layer(enc_output, user_output, adj, event_type)
        #     if C.DATASET in {'douban-book'}:
        #         enc_output += residual
        #     all_layer_outputs.append(enc_output.mean(1))

        # return all_layer_outputs

###########################################

        ## Original from CaDRec
        return enc_output.mean(1)


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.dropout = nn.Dropout(0.5)
        self.temperature = 512 ** 0.5
        self.dim = dim

    def forward(self, user_embeddings, embeddings, pop_encoding, evaluation):
        outputs = []
        if C.ABLATION != 'w/oMatcher':
            if not evaluation:
                # C.BETA_1：
                #   Foursquare 0.5 Yelp2018 0.3 Gowalla 0.1 Brightkite 0.3 ml-1M 0.8 lastfm-2k 0.3 douban-book 0.05
                item_encoding = torch.concat([embeddings[1:], pop_encoding[1:] * C.BETA_1], dim=-1)
                out = user_embeddings.matmul(item_encoding.T)
            else:
                item_encoding = embeddings[1:]
                out = user_embeddings[:, :self.dim].matmul(item_encoding.T)

            # out = user_embeddings.matmul(embeddings.T[:,1:])
            out = F.normalize(out, p=2, dim=-1, eps=1e-05)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=0).sum(0)
        out = torch.tanh(outputs)
        return out


class Model(nn.Module):
    def __init__(
            self, num_types, d_model=256, n_layers=4, n_head=4, dropout=0.1, device=0):
        super(Model, self).__init__()

        self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=C.PAD)
        self.user_emb = nn.Embedding(C.USER_NUMBER, d_model, padding_idx=C.PAD)

        self.encoder = Encoder(
            num_types=num_types, d_model=d_model,
            n_layers=n_layers, n_head=n_head, dropout=dropout)
        self.num_types = num_types

        self.predictor = Predictor(d_model, num_types)

        self.time_gate = nn.Linear(d_model, d_model)
        self.time_dropout = nn.Dropout(dropout)

#####################################
# Contrastive learning augmentation -> shows small improvements in R and NDCG
        ## Member variables / functions
        # self.cl_proj = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, d_model)
        # )

        # self.cl_proj_user = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, d_model)
        # )

        # self.cl_proj_layer = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, d_model)
        # )
#####################################
# Counterfactual Regularization with Contrastive Learning
        # num_classes = 2 # defines what type of classification 2 = binary
        # self.aux_head = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.ReLU(),
        #     nn.Linear(d_model // 2, num_classes)
        # )

#####################################

    ## Original from CaDRec
    def forward(self, user_id, event_type, event_time, adjacent_matrix, pop_encoding, evaluation=True):
    # Modified for time-weighted embeddings
    # def forward(self, user_id, event_type, event_time, adjacent_matrix, pop_encoding, time_weights, evaluation=True):

        non_pad_mask = Utils.get_non_pad_mask(event_type)

        # (K M)  event_emb: Embedding
        enc_output = self.event_emb(event_type)
################################
# Additional Code for time-aware positional encoding
        # pe = Utils.time_positional_encoding(event_time, enc_output.size(-1))
        # # enc_output = enc_output + pe

        # gate = torch.sigmoid(self.time_gate(pe))
        # gate = self.time_dropout(gate)
        # enc_output = enc_output * (1 - gate) + pe * gate

# Additional Code for time-weighted embeddings
        # enc_output = enc_output * time_weights.unsqueeze(-1)

################################
        user_output = self.user_emb(user_id)

#####################################
# Contrastive learning augmentation -> shows small improvements in R and NDCG
        # implementation and assignment of member variables
        ## Part
        # view1 = self.cl_proj(F.dropout(enc_output, p=0.3, training=self.training))
        # view2 = self.cl_proj(F.dropout(enc_output, p=0.3, training=self.training))

        # self.cl_view1 = view1.mean(dim=1)
        # self.cl_view2 = view2.mean(dim=1)

        ## Part 2
        # event_type_view1 = random_event_mask(event_type, mask_prob=0.4, pad_token=C.PAD)
        # event_type_view2 = random_event_mask(event_type, mask_prob=0.4, pad_token=C.PAD)

        # event_input_view1 = self.event_emb(event_type_view1)
        # event_input_view2 = self.event_emb(event_type_view2)

        # user_input = self.user_emb(user_id)

        # z1 = self.encoder(user_id, event_type_view1, event_input_view1, user_input, adjacent_matrix)
        # z2 = self.encoder(user_id, event_type_view2, event_input_view2, user_input, adjacent_matrix)

        # self.user_cl_view1 = self.cl_proj_user(z1)
        # self.user_cl_view2 = self.cl_proj_user(z2)

        # self.cl_view1 = self.cl_proj(z1)
        # self.cl_view2 = self.cl_proj(z2)

        # # Using Multi-view contrastive head
        # self.user_cl_view1 = self.cl_proj_user(z1[0])
        # self.user_cl_view2 = self.cl_proj_user(z2[-1])

        # self.cl_view1 = self.cl_proj(z1[0])
        # self.cl_view2 = self.cl_proj(z2[-1])

#####################################

        pop_output = pop_encoding[event_type] * non_pad_mask

        if C.ABLATION != 'w/oUSpec' and C.ABLATION != 'w/oDisen':
            enc_output += torch.sign(enc_output)\
                          * F.normalize(user_output.unsqueeze(1), dim=-1) # * torch.sign(user_output.unsqueeze(1)) \

        output = self.encoder(user_id, event_type, enc_output, user_output, adjacent_matrix)

#####################################
# Counter Factual Regularization
        # zero_adj = torch.zeros_like(adjacent_matrix)
        # cf_output = self.encoder(user_id, event_type, enc_output, user_output, zero_adj)
        # #cf_output = cf_output.mean(1)
        # real_output = output#.mean(1)

        # # real_output = F.normalize(real_output, dim=1)
        # # cf_output = F.normalize(cf_output, dim=1)

        # self.cf_output = cf_output.mean(1)
        # self.real_output = real_output.mean(1)

        # real_repr = self.encoder(user_id, event_type, enc_output, user_output, adjacent_matrix)
        # cf_repr = self.encoder(user_id, event_type, enc_output, user_output, zero_adj)

        # self.aux_logits_real = self.aux_head(real_repr)
        # self.aux_logits_cf = self.aux_head(cf_repr)

#####################################
# Contrastive Learning Algorithm with Multi-View Contrastive Head

        # layer_outputs = self.encoder(user_id, event_type, enc_output, user_output, adjacent_matrix)
        # output = layer_outputs[-1]

        # z1 = layer_outputs[0]
        # z2 = layer_outputs[-1]

        # self.view_layer1 = self.cl_proj_layer(z1)
        # self.view_layerN = self.cl_proj_layer(z2)

#####################################

        user_embeddings = torch.concat([output, torch.mean(pop_output, dim=1) * C.BETA_1], dim=-1)

        prediction = self.predictor(user_embeddings, self.event_emb.weight, pop_encoding, evaluation)

#####################################
# counterfactual regularization with contrastive learning / auxiliary classification
        # self.aux_logits = self.aux_head(prediction)

#####################################

        return prediction, user_embeddings, pop_output
