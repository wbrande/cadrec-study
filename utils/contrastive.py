# New file for use with contrastive learning augmentation -> shows small improvements in R and NDCG
# contains functions for performing contrastive learning algorithm
import torch
import torch.nn.functional as F

def info_nce_loss(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.matmul(representations, representations.T)

    # Remove self-similarity
    mask = torch.eye(sim_matrix.size(0), dtype=torch.bool).to(z1.device)
    sim_matrix.masked_fill_(mask, float('-inf'))

    sim_exp = torch.exp(sim_matrix / temperature)
    pos_sim = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    denom = sim_exp.sum(dim=1)
    loss = -torch.log(pos_sim / denom)

    return loss.mean()

def info_nce_with_hard_negatives(z1, z2, item_ids, item_to_bin, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    device = z1.device

    representations = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.matmul(representations, representations.T)

    all_ids = item_ids.tolist() + item_ids.tolist()
    bins = [item_to_bin.get(i, -1) for i in all_ids]

    mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device)
    for i in range(2 * batch_size):
        for j in range(2 * batch_size):
            if i != j and bins[i] == bins[j] and bins[i] != -1:
                mask[i, j] = 1

    sim_matrix = torch.exp(sim_matrix / temperature)
    sim_matrix = sim_matrix.masked_fill(~mask, 0.0)

    positives = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
    positives = torch.cat([positives, positives], dim=0)

    denom = sim_matrix.sum(dim=1) + 1e-8
    loss = -torch.log(positives / denom)

    return loss.mean()

def align_uniform_loss(z1, z2, lambda_uniform=1.0):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # alignment
    align = (z1 - z2).norm(dim=2).pow(2).mean() if z1.dim() == 3 else (z1 - z2).norm(dim=1).pow(2).mean()

    # uniformity
    z = torch.cat([z1, z2], dim=0)
    sq_pdist = torch.pdist(z, p=2).pow(2)
    uniform = torch.log(torch.exp(-2 * sq_pdist).mean() + 1e-8)

    return align + lambda_uniform * uniform

def random_event_mask(event_type, mask_prob=0.2, pad_token=0):
    mask = (torch.rand(event_type.shape) > mask_prob).to(event_type.device)
    return torch.where(mask, event_type, pad_token)