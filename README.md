# HW2: Kaggle top-20 recommendation (SKKU ESW4024)

Competition: <https://www.kaggle.com/competitions/skku-esw-4024-recommender-system/>
Metric: NDCG@20 over a ranked list of 20 `movieId`s per `userId` in `test.csv`.

Every number below was produced by the code in this folder. All models are implemented from
scratch (numpy / PyTorch); RecBole is not used.

```
RS_HW2/
├── src/
│   ├── common.py        # data loading, temporal validation split, NDCG@20, submission writer
│   ├── q1_ease.py       # Q1: EASE + grid search + submission
│   └── q2_neumf.py      # Q2: GMF / MLP / NeuMF + grid search + full-data refit + submission
├── run_q2_when_gpu_free.sh  # waits for an idle GPU on the shared server, then runs Q2
├── results/             # validation tables, learning curves, logs
└── submissions/         # q1_ease.csv, q2_neumf.csv (Kaggle format)
```

## Setup

Python 3.11, CUDA 12.1 driver.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python -r requirements.txt
```

Put the Kaggle files (`train.csv`, `test.csv`, `sample_submission.csv`) in `data/`.

## Data and design choices (check list)

| | |
|---|---|
| Data | 80,621 interactions, 610 users, 8,237 movies (density 1.6%); 588 users in `test.csv` |
| Train / validation split | Per user, the last 20% of interactions by `time_idx` are held out (16,120 interactions). `time_idx` is 0..n-1 per user and the dataset looks like MovieLens-latest-small with ~20% removed, so the hidden test is most likely each user's most recent interactions; a temporal hold-out mimics that. |
| User / item sets | `userId` is already 0..609. Item set = the 8,237 movies that appear in `train.csv` (movies never seen in training cannot be scored by CF). Items a user has already rated are excluded from their list. |
| Ratings | Treated as implicit feedback. For EASE, three inputs were compared on validation: binary, rating-weighted, and rating >= 4 only (table below). NeuMF uses binary positives, as in the paper. |
| `time_idx` | Used only for the validation split; no sequential model (left for the project). |
| NDCG@20 on validation | Binary relevance, full ranking over all items, averaged over the 588 test users (`common.ndcg_at_k`). |
| Final model | Refit on all of `train.csv` with the best validated hyper-parameters (for NeuMF, the validated number of epochs). |

## Q1: EASE

```bash
.venv/bin/python src/q1_ease.py     # ~7 min on CPU; writes submissions/q1_ease.csv
```

EASE (Steck, WWW'19): `B = I - P diagMat(1 / diag(P))`, `P = (XᵀX + λI)⁻¹`, scores = `X B`.
λ ∈ {10, ..., 5000} for each input type (`results/q1_ease_val.csv`).

| Model | Input | best λ | Validation NDCG@20 |
|---|---|---|---|
| MostPop | - | - | 0.0991 |
| EASE | binary | 200 | 0.1557 |
| EASE | rating >= 4 | 100 | 0.1584 |
| **EASE** | **rating-weighted** | **2000** | **0.1599** |

Submission: rating-weighted EASE, λ = 2000, refit on the full train set.

## Q2: NeuMF

```bash
.venv/bin/python src/q2_neumf.py search --factors 16 32 64 --device cuda:0   # ~6 min on an RTX 3090
.venv/bin/python src/q2_neumf.py refit --device cuda:0                        # writes submissions/q2_neumf.csv
```

Follows He et al., WWW'17 and the official code:
- every observed (user, movie) is a positive; 4 negatives per positive, sampled uniformly from unobserved items and re-sampled every epoch; binary cross-entropy (log loss).
- GMF with `f` factors; MLP tower `[8f, 4f, 2f, f]` (user/item embeddings of size `4f`); NeuMF concatenates the last GMF and MLP layers into one output layer.
- NeuMF with pre-training: initialised from the trained GMF and MLP (output layer `[0.5·h_GMF, 0.5·h_MLP]`), then fine-tuned with SGD (lr 0.01, as in the paper) or Adam (lr 1e-4).
- Adam lr 1e-3, batch 1024, no weight decay; early stopping on validation NDCG@20 (patience 10).

Validation NDCG@20 (`results/q2_neumf_val.csv`, learning curves in `results/q2_neumf_curves.json`):

| f | GMF | MLP | NeuMF (no pre-train) | NeuMF pre-train + Adam | NeuMF pre-train + SGD |
|---|---|---|---|---|---|
| 16 | 0.1145 | 0.1175 | 0.1240 | 0.1284 | 0.1281 |
| 32 | 0.1208 | 0.1233 | 0.1298 | 0.1339 | **0.1342** |
| 64 | 0.1194 | 0.1195 | 0.1259 | 0.1328 | 0.1315 |

Submission: pre-trained NeuMF, f = 32, SGD fine-tuning; refit on the full train set with the validated
epochs (GMF 20, MLP 6, NeuMF 5).

As in the paper, NeuMF beats GMF and MLP and pre-training helps. It stays below EASE on this data
(0.134 vs 0.160), which matches reports that a well-tuned linear / MF model is hard to beat on
MovieLens (Rendle et al., RecSys'20; Dacrema et al., RecSys'19).

## Leaderboard

| Submission | Validation NDCG@20 | Public leaderboard |
|---|---|---|
| `q1_ease.csv` | 0.1599 | 0.13178 |
| `q2_neumf.csv` | 0.1342 | 0.10370 |

The leaderboard keeps the validation ordering (EASE > NeuMF), with both scores about 20% below
validation.
