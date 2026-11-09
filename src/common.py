"""Shared data loading, temporal validation split, NDCG@20 evaluation and submission writing."""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submissions"
K = 20


def load():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    return train, test


class Index:
    """userId is already 0..n_users-1; movieId is remapped to a contiguous item index."""

    def __init__(self, train):
        self.n_users = int(train.userId.max()) + 1
        self.items = np.sort(train.movieId.unique())  # item index -> movieId
        self.item2idx = pd.Series(np.arange(len(self.items)), index=self.items)
        self.n_items = len(self.items)

    def item_idx(self, movie_ids):
        return self.item2idx.loc[np.asarray(movie_ids)].to_numpy()


def temporal_split(df, ratio=0.2):
    """Per user, hold out the last `ratio` of interactions (by time_idx) as validation.

    This mirrors the assumed construction of the hidden test set: the competition train.csv
    keeps an ordered prefix of each user's history (time_idx = 0..n-1).
    """
    df = df.sort_values(["userId", "time_idx"])
    n = df.groupby("userId")["movieId"].transform("size").to_numpy()
    rank = df.groupby("userId").cumcount().to_numpy()
    n_val = np.maximum(1, np.round(n * ratio)).astype(int)
    is_val = rank >= n - n_val
    return df[~is_val].copy(), df[is_val].copy()


def interaction_matrix(df, idx, values="binary"):
    """User x item CSR matrix. values: 'binary' | 'rating' | float threshold (keep rating >= t)."""
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        df = df[df.rating >= values]
        data = np.ones(len(df), dtype=np.float32)
    elif values == "rating":
        data = df.rating.to_numpy(dtype=np.float32)
    else:
        data = np.ones(len(df), dtype=np.float32)
    rows = df.userId.to_numpy()
    cols = idx.item_idx(df.movieId)
    return sp.csr_matrix((data, (rows, cols)), shape=(idx.n_users, idx.n_items), dtype=np.float32)


def topk(scores, seen, users, k=K):
    """scores: (len(users), n_items) array. seen: CSR of items to exclude. Returns ranked item indices."""
    scores = np.array(scores, dtype=np.float32, copy=True)
    sub = seen[users]
    r, c = sub.nonzero()
    scores[r, c] = -np.inf
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


_DISCOUNT = 1.0 / np.log2(np.arange(2, K + 2))


def ndcg_at_k(ranked, truth, k=K):
    """ranked: (n, k) item indices. truth: list of arrays of relevant item indices. Binary relevance."""
    vals = np.empty(len(truth))
    for u, (rec, rel) in enumerate(zip(ranked, truth)):
        hits = np.isin(rec[:k], rel)
        idcg = _DISCOUNT[: min(len(rel), k)].sum()
        vals[u] = (hits * _DISCOUNT[:k]).sum() / idcg
    return vals.mean()


class Validation:
    """Holds the temporal split and evaluates a scoring function on the test users."""

    def __init__(self, train_df, test_df, idx, ratio=0.2):
        self.idx = idx
        self.tr, self.va = temporal_split(train_df, ratio)
        self.seen = interaction_matrix(self.tr, idx)
        # evaluate on the users the leaderboard scores (all of them have >= 1 validation item)
        self.users = np.sort(test_df.userId.unique())
        va_items = self.va.assign(item=idx.item_idx(self.va.movieId)).groupby("userId")["item"].apply(np.array)
        self.truth = [va_items.get(u, np.array([], dtype=int)) for u in self.users]

    def evaluate(self, scores):
        """scores: (len(self.users), n_items)."""
        return ndcg_at_k(topk(scores, self.seen, self.users), self.truth)


def write_submission(test_df, ranked_by_user, idx, path):
    """ranked_by_user: dict-like userId -> ranked item indices (length K)."""
    recs = [" ".join(map(str, idx.items[ranked_by_user[u]])) for u in test_df.userId]
    sub = pd.DataFrame({"id": test_df.id, "recommendations": recs})
    sample = pd.read_csv(DATA / "sample_submission.csv")
    assert list(sub.columns) == list(sample.columns) and (sub.id.values == sample.id.values).all()
    assert all(len(set(r.split())) == K for r in sub.recommendations)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return sub
