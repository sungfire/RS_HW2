"""Q2: NeuMF (He et al., WWW'17) on the HW2 setup.

Follows the paper / official code (github.com/hexiangnan/neural_collaborative_filtering):
  * implicit feedback: every (user, movie) in train.csv is a positive,
  * pointwise log loss with `neg` uniformly sampled unobserved items per positive (re-sampled each epoch),
  * GMF (embedding f), MLP (tower [8f, 4f, 2f, f], embedding 4f each), NeuMF = concat(GMF, MLP) -> linear,
  * NeuMF initialised from pre-trained GMF and MLP with alpha = 0.5.

Usage:
  python src/q2_neumf.py search --factors 16 32 64 --device cuda:0   # temporal validation, early stopping
  python src/q2_neumf.py refit --device cuda:0                         # best config on full train.csv -> submission
"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import (RESULTS, SUBMISSIONS, Index, Validation, interaction_matrix, load, topk,
                    write_submission)

CKPT = RESULTS / "ckpt"
SEED = 2026


# ----------------------------------------------------------------------------- models
def _init_linear(m):
    nn.init.xavier_uniform_(m.weight)
    nn.init.zeros_(m.bias)


class GMF(nn.Module):
    def __init__(self, n_users, n_items, f):
        super().__init__()
        self.P = nn.Embedding(n_users, f)
        self.Q = nn.Embedding(n_items, f)
        self.h = nn.Linear(f, 1)
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        _init_linear(self.h)

    def features(self, u, i):
        return self.P(u) * self.Q(i)

    def forward(self, u, i):
        return self.h(self.features(u, i)).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, n_users, n_items, f):
        super().__init__()
        sizes = [8 * f, 4 * f, 2 * f, f]  # official code default layers=[64,32,16,8] for f=8
        self.P = nn.Embedding(n_users, sizes[0] // 2)
        self.Q = nn.Embedding(n_items, sizes[0] // 2)
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.tower = nn.Sequential(*layers)
        self.h = nn.Linear(f, 1)
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        for m in self.tower:
            if isinstance(m, nn.Linear):
                _init_linear(m)
        _init_linear(self.h)

    def features(self, u, i):
        return self.tower(torch.cat([self.P(u), self.Q(i)], dim=-1))

    def forward(self, u, i):
        return self.h(self.features(u, i)).squeeze(-1)


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, f):
        super().__init__()
        self.gmf = GMF(n_users, n_items, f)
        self.mlp = MLP(n_users, n_items, f)
        self.h = nn.Linear(2 * f, 1)
        _init_linear(self.h)

    def load_pretrained(self, gmf, mlp, alpha=0.5):
        self.gmf.load_state_dict(gmf.state_dict())
        self.mlp.load_state_dict(mlp.state_dict())
        with torch.no_grad():
            self.h.weight.copy_(torch.cat([alpha * gmf.h.weight, (1 - alpha) * mlp.h.weight], dim=1))
            self.h.bias.copy_(alpha * gmf.h.bias + (1 - alpha) * mlp.h.bias)

    def forward(self, u, i):
        return self.h(torch.cat([self.gmf.features(u, i), self.mlp.features(u, i)], dim=-1)).squeeze(-1)


MODELS = {"gmf": GMF, "mlp": MLP, "neumf": NeuMF}


# ----------------------------------------------------------------------------- training
class NegSampler:
    def __init__(self, X, n_items, rng):
        self.users, self.items = X.nonzero()
        self.n_items = n_items
        self.pos_codes = np.sort(self.users.astype(np.int64) * n_items + self.items)
        self.rng = rng

    def _is_pos(self, u, j):
        codes = u.astype(np.int64) * self.n_items + j
        pos = np.searchsorted(self.pos_codes, codes)
        pos[pos == len(self.pos_codes)] = 0
        return self.pos_codes[pos] == codes

    def epoch(self, neg):
        u = np.repeat(self.users, neg)
        j = self.rng.integers(0, self.n_items, size=len(u))
        bad = self._is_pos(u, j)
        while bad.any():
            j[bad] = self.rng.integers(0, self.n_items, size=bad.sum())
            bad[bad] = self._is_pos(u[bad], j[bad])
        users = np.concatenate([self.users, u])
        items = np.concatenate([self.items, j])
        labels = np.concatenate([np.ones(len(self.users)), np.zeros(len(u))]).astype(np.float32)
        perm = self.rng.permutation(len(users))
        return users[perm], items[perm], labels[perm]


@torch.no_grad()
def score_all(model, users, n_items, device, chunk=32):
    model.eval()
    items = torch.arange(n_items, device=device)
    out = []
    for s in range(0, len(users), chunk):
        u = torch.as_tensor(users[s:s + chunk], device=device)
        uu = u.repeat_interleave(n_items)
        ii = items.repeat(len(u))
        out.append(model(uu, ii).view(len(u), n_items).float().cpu())
    return torch.cat(out).numpy()


def make_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["opt"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["wd"])


def train(cfg, X, idx, device, val=None, epochs=None, pretrained=None, log=print):
    """Train one model. With `val`, early-stop on validation NDCG@20 and return the best state.
    Without `val`, train for exactly `epochs` epochs (refit on the full data)."""
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = MODELS[cfg["model"]](idx.n_users, idx.n_items, cfg["f"]).to(device)
    if pretrained is not None:
        model.load_pretrained(*pretrained)
    opt = make_optimizer(model, cfg)
    sampler = NegSampler(X, idx.n_items, rng)
    loss_fn = nn.BCEWithLogitsLoss()
    best = {"ndcg": -1.0, "epoch": 0, "state": None}
    curve = []
    if val is not None and pretrained is not None:  # starting point of the pre-trained NeuMF
        ndcg0 = val.evaluate(score_all(model, val.users, idx.n_items, device))
        best.update(ndcg=ndcg0, epoch=0, state=copy.deepcopy(model.state_dict()))
        curve.append({"epoch": 0, "loss": None, "ndcg@20": ndcg0})
    n_epochs = cfg["max_epochs"] if val is not None else epochs
    for ep in range(1, n_epochs + 1):
        t = time.time()
        model.train()
        u, i, y = (torch.as_tensor(a, device=device) for a in sampler.epoch(cfg["neg"]))
        total = 0.0
        for s in range(0, len(y), cfg["batch"]):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(u[s:s + cfg["batch"]], i[s:s + cfg["batch"]]), y[s:s + cfg["batch"]])
            loss.backward()
            opt.step()
            total += loss.item() * len(y[s:s + cfg["batch"]])
        row = {"epoch": ep, "loss": total / len(y)}
        if val is not None:
            row["ndcg@20"] = val.evaluate(score_all(model, val.users, idx.n_items, device))
            if row["ndcg@20"] > best["ndcg"]:
                best.update(ndcg=row["ndcg@20"], epoch=ep, state=copy.deepcopy(model.state_dict()))
        curve.append(row)
        log(f"  [{cfg['name']}] ep {ep:3d} loss {row['loss']:.4f}"
            + (f" ndcg@20 {row['ndcg@20']:.4f} (best {best['ndcg']:.4f} @ {best['epoch']})" if val is not None else "")
            + f" {time.time() - t:.1f}s")
        if val is not None and ep - best["epoch"] >= cfg["patience"]:
            break
    if val is not None:
        model.load_state_dict(best["state"])
    return model, best, curve


# ----------------------------------------------------------------------------- experiments
def base_cfg(model, f, **kw):
    cfg = dict(model=model, f=f, neg=4, lr=1e-3, wd=0.0, opt="adam", batch=1024,
               max_epochs=100, patience=10, seed=SEED)
    cfg.update(kw)
    cfg["name"] = kw.pop("name", None) or f"{model}-f{f}"
    return cfg


def search(args):
    device = torch.device(args.device)
    train_df, test_df = load()
    idx = Index(train_df)
    val = Validation(train_df, test_df, idx)
    X = interaction_matrix(val.tr, idx)
    CKPT.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / "q2_neumf_val.csv"
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    curves = {}
    curves_path = RESULTS / "q2_neumf_curves.json"
    if curves_path.exists():
        curves = json.load(open(curves_path))

    def record(cfg, best, curve, extra=None):
        r = {k: v for k, v in cfg.items() if k != "name"}
        r.update(name=cfg["name"], best_epoch=best["epoch"], **{"ndcg@20": best["ndcg"]})
        if extra:
            r.update(extra)
        rows[:] = [x for x in rows if x["name"] != cfg["name"]] + [r]
        curves[cfg["name"]] = curve
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        json.dump(curves, open(curves_path, "w"))
        print(f"==> {cfg['name']}: NDCG@20 {best['ndcg']:.4f} at epoch {best['epoch']}", flush=True)

    for f in args.factors:
        pre = {}
        for m in ("gmf", "mlp"):
            cfg = base_cfg(m, f, neg=args.neg, wd=args.wd)
            model, best, curve = train(cfg, X, idx, device, val=val)
            torch.save(model.state_dict(), CKPT / f"{cfg['name']}.pt")
            pre[m] = model
            record(cfg, best, curve)
        cfg = base_cfg("neumf", f, neg=args.neg, wd=args.wd, name=f"neumf-f{f}-scratch")
        _, best, curve = train(cfg, X, idx, device, val=val)
        record(cfg, best, curve, {"pretrain": False})
        for opt, lr in args.pre_opts:
            cfg = base_cfg("neumf", f, neg=args.neg, wd=args.wd, opt=opt, lr=lr,
                           name=f"neumf-f{f}-pre-{opt}{lr:g}")
            _, best, curve = train(cfg, X, idx, device, val=val, pretrained=(pre["gmf"], pre["mlp"]))
            record(cfg, best, curve, {"pretrain": True})


def refit(args):
    """Retrain the best validated NeuMF (and its pre-training) on all of train.csv for the validated
    number of epochs, then write the submission."""
    device = torch.device(args.device)
    train_df, test_df = load()
    idx = Index(train_df)
    X = interaction_matrix(train_df, idx)
    res = pd.read_csv(RESULTS / "q2_neumf_val.csv")
    neu = res[res.model == "neumf"].sort_values("ndcg@20", ascending=False).iloc[0]
    print(f"refit best NeuMF config: {neu['name']} (val NDCG@20 {neu['ndcg@20']:.4f})")
    cols = ["model", "f", "neg", "lr", "wd", "opt", "batch", "max_epochs", "patience", "seed"]
    cfg_of = lambda r: {**{c: (r[c].item() if hasattr(r[c], "item") else r[c]) for c in cols}, "name": r["name"] + "-full"}
    pretrained = None
    if str(neu.get("pretrain")) == "True":
        pre = []
        for m in ("gmf", "mlp"):
            r = res[res.name == f"{m}-f{int(neu.f)}"].iloc[0]
            model, _, _ = train(cfg_of(r), X, idx, device, epochs=max(1, int(r.best_epoch)))
            pre.append(model)
        pretrained = tuple(pre)
    model, _, _ = train(cfg_of(neu), X, idx, device, epochs=int(neu.best_epoch), pretrained=pretrained)
    users = test_df.userId.to_numpy()
    ranked = topk(score_all(model, users, idx.n_items, device), X, users)
    write_submission(test_df, dict(zip(users, ranked)), idx, SUBMISSIONS / "q2_neumf.csv")
    json.dump({k: (v.item() if hasattr(v, "item") else v) for k, v in neu.items()},
              open(RESULTS / "q2_neumf_best.json", "w"), indent=2)
    print("wrote submissions/q2_neumf.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["search", "refit"])
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--factors", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--neg", type=int, default=4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--pre-opts", type=lambda s: (s.split(":")[0], float(s.split(":")[1])), nargs="+",
                   default=[("adam", 1e-4), ("sgd", 1e-2)], help="optimizer:lr for pre-trained NeuMF")
    args = p.parse_args()
    torch.set_num_threads(8)
    search(args) if args.mode == "search" else refit(args)


if __name__ == "__main__":
    main()
