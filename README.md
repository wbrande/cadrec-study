# CaDRec: Study, Framework Port, and Enhancement Experiments

**Course:** CPSC 8470 — Clemson University  
**Authors:** Carl Brandenburg, Keller Brandenburg

This repository contains our work from a three-phase course project built around the paper [*CaDRec: Contextualized and Debiased Recommender Model*](https://arxiv.org/pdf/2404.06895) (SIGIR 2024). The original source code is available at [WangXFng/CaDRec](https://github.com/WangXFng/CaDRec).

---

## What CaDRec Is

CaDRec is a graph-based recommender system that addresses two common problems in recommendation:

- **Over-smoothing of node embeddings** — as embeddings are passed between nodes in a graph, they tend to become similar, leading to generalized rather than personalized recommendations. CaDRec uses a hypergraph convolution operator that selects effective neighbors using both structural and sequential context.
- **Popularity and user bias** — popular items can dominate recommendations. CaDRec uses positional encoding to track item popularity and models individual user interaction quality to produce less biased embeddings.

---

## Phase 1 — Paper Analysis

We analyzed the CaDRec paper to understand its goals, methods, and limitations. Key findings:

- CaDRec's self-attention perturbations apply learned noise to adjust which neighbors are used, improving personalization.
- The model is sensitive to hyperparameter tuning; improper tuning can cause over- or under-regularization.
- The model assumes sequential context is always useful and that user preferences are static over time — both assumptions break down in practice over long time horizons.

We identified potential improvements for later phases: time-weighted embeddings to decay the influence of older interactions, and sparse attention convolutions to improve scalability.

---

## Phase 2 — Reproduction and Framework Port

### Reproduction

We ran the original CaDRec implementation on Clemson's Palmetto GPU cluster and compared our results against the paper's reported figures across four datasets.

| Dataset | Metric | Ours | Paper |
|---|---|---|---|
| Yelp2018 | R@20 / N@20 | .0792 / .0667 | .0792 / .0667 |
| Foursquare | R@20 / N@20 | .1300 / .1064 | .1314 / .1068 |
| Douban-book | R@20 / N@20 | .2075 / .1959 | .2085 / .1960 |
| ML-1M | R@20 / N@20 | .2905 / .3278 | .2907 / .3261 |

Results are close to published figures, with small variance attributable to hardware and library version differences.

We also ran the paper's four ablation tests (removing self-attention, disengagement, normalization, and weight sharing). Results were broadly consistent with the paper, with some discrepancies likely due to environment differences.

### TensorFlow Port

We attempted to port the model from PyTorch to TensorFlow/Keras. Our workflow was to replace PyTorch-specific functions with TensorFlow equivalents throughout the codebase (hGCNLayer, hGCNEncoder, Encoder, Predictor, and training loop).

The port ran — training and validation epochs completed and metrics were recorded — but the metrics were significantly worse than the original. Additionally, GPU access on Palmetto required elevated privileges that we did not have, so the TensorFlow version ran on CPU only, taking approximately 9 minutes per training epoch compared to ~0.8 minutes for the PyTorch baseline.

The TensorFlow port code is preserved in this repository with the original PyTorch code commented out alongside it for comparison.

---

## Phase 3 — Enhancement Experiments

We designed and tested 8 modifications to the model, measuring Recall (R) and Normalized Discounted Cumulative Gain (NDCG) across all four datasets. None consistently improved on the base model.

| Enhancement | Summary of Result |
|---|---|
| Adaptive fusion gating | No improvement on any dataset; likely too much added complexity without sufficient regularization |
| Contrastive learning + hard negative mining | No significant effect on metrics; nearly doubled runtime on Yelp2018 |
| Contrastive learning + multi-view head | Similar to above — no metric improvement, sharp runtime increase |
| Counterfactual regularization | No significant change in metrics; increased runtime due to second forward pass |
| Time-weighted embeddings | Slight metric drop across most datasets; modest runtime reduction |
| Sequence trimming | Significant metric drop on ML-1M and Douban-book (denser datasets); modest runtime reduction |
| Session-level attention | Near-zero R and NDCG on Douban-book and ML-1M; likely softmax collapse or padding mismatch |
| Adjacency matrix sampling | Performance dropped across all datasets, especially ML-1M; faster runtime |
| Knowledge distillation (teacher–student) | Performance drop on all datasets; modest runtime reduction |

The enhancement code is preserved in `Main.py` wrapped in comment blocks. To enable an enhancement, uncomment the relevant block and comment out the original code it replaces. The knowledge distillation enhancement requires running with `--train_teacher` first to generate a teacher model.

The base model proved difficult to improve, which reflects its design — it is a well-tuned system that already accounts for the problems these enhancements were targeting.

---

## Setup (Palmetto Cluster)

```bash
module load anaconda3/2023.09-0
module load cuda/12.3.0
module load ngc/pytorch/23.06

pip install --no-cache-dir --user optuna==3.3.0
pip install --no-cache-dir --user sqlalchemy
```

Add to `~/.bashrc`:
```bash
export PYTHONPATH=$PYTHONPATH:/home/{username}/.local/lib/python3.11/site-packages
```

For the TensorFlow port only:
```bash
pip install --no-cache-dir --user tensorflow
```

### Run

```bash
python Main.py
```

Configuration is set in `Constants.py` and `Main.py`.

---

## Datasets

Three files required per dataset: `train.txt`, `tune.txt`, `test.txt`.

Each line: `[USER_ID]\t[ITEM_ID]\t[INTERACTION_COUNT]`

---

## Citation

```bibtex
@inproceedings{wang2023cadrec,
  title={CaDRec: Contextualized and Debiased Recommender Model},
  author={Wang, Xinfeng and Fukumoto, Fumiyo and Cui, Jin and Suzuki, Yoshimi and Li, Jiyi and Yu, Dongjin},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={405--415},
  year={2024}
}
```
