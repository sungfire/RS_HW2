"""Q1: EASE (Steck, WWW'19) on the HW2 setup.

1. Temporal hold-out (last 20% of each user's history) for validation, NDCG@20 on the test users.
2. Grid search over the L2 weight lambda and over how ratings enter the input matrix.
3. Refit on the full train.csv with the best setting and write the top-20 submission.
"""
import json
import time

import numpy as np

from common import (RESULTS, SUBMISSIONS, Index, Validation, interaction_matrix, load, topk,
                    write_submission)

LAMBDAS = [10, 50, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000]
INPUTS = ["binary", "rating", 4.0]  # all interactions / rating-weighted / only rating >= 4


def gram(X):
    return (X.T @ X).toarray().astype(np.float64)


def ease_weights(G, lam):
    G = G.copy()
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = P / (-np.diag(P))
    B[np.diag_indices_from(B)] = 0.0
    return B


def popularity_scores(X, n_rows):
    pop = np.asarray(X.sum(axis=0)).ravel()
    return np.tile(pop, (n_rows, 1))


def main():
    train, test = load()
    idx = Index(train)
    val = Validation(train, test, idx)
    print(f"val split: {len(val.tr)} train / {len(val.va)} validation interactions, "
          f"{len(val.users)} eval users, {idx.n_items} items")

    X_bin = interaction_matrix(val.tr, idx)
    rows = [{"model": "MostPop", "input": "binary", "lambda": None,
             "ndcg@20": val.evaluate(popularity_scores(X_bin, len(val.users)))}]
    print(f"MostPop            NDCG@20 = {rows[0]['ndcg@20']:.4f}")

    for inp in INPUTS:
        X = interaction_matrix(val.tr, idx, inp)
        G = gram(X)
        for lam in LAMBDAS:
            t = time.time()
            B = ease_weights(G, lam)
            ndcg = val.evaluate(X[val.users] @ B)
            rows.append({"model": "EASE", "input": str(inp), "lambda": lam, "ndcg@20": ndcg})
            print(f"EASE input={str(inp):6s} lambda={lam:5d}  NDCG@20 = {ndcg:.4f}  ({time.time() - t:.1f}s)")

    RESULTS.mkdir(exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(RESULTS / "q1_ease_val.csv", index=False)

    best = max((r for r in rows if r["model"] == "EASE"), key=lambda r: r["ndcg@20"])
    inp = best["input"] if best["input"] in ("binary", "rating") else float(best["input"])
    print(f"best: input={best['input']} lambda={best['lambda']} NDCG@20={best['ndcg@20']:.4f}")

    # refit on the full train.csv and predict for the test users
    X_full = interaction_matrix(train, idx, inp)
    B = ease_weights(gram(X_full), best["lambda"])
    users = test.userId.to_numpy()
    ranked = topk(X_full[users] @ B, interaction_matrix(train, idx), users)
    write_submission(test, dict(zip(users, ranked)), idx, SUBMISSIONS / "q1_ease.csv")
    json.dump(best, open(RESULTS / "q1_ease_best.json", "w"), indent=2)
    print("wrote submissions/q1_ease.csv")


if __name__ == "__main__":
    main()
