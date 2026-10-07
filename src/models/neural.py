"""
RQ3 - neural architectural framings for how BAG enters the downstream model.

One PyTorch network and one training routine serve every framing, so framings
differ only in how BAG is used (proposal 3.4.2), never in architecture or
training details:

  single      one outcome head, no BAG supervision. With covariates+biomarkers
              as input this is the equivalent-capacity baseline; with estimated
              BAG added as an input column it is the two-stage framing.
  multitask   shared encoder, a BAG head and an outcome head trained jointly:
              loss = w * L_BAG + (1 - w) * L_outcome.
  bottleneck  multitask, but the BAG head's single scalar is also fed into the
              outcome head. predict(..., ablate=True) zeroes that scalar to
              test whether the outcome head actually relies on it.
  transfer    continual learning: train encoder + BAG head on BAG first, then
              replace the head and fine-tune on the outcome with a reduced
              encoder learning rate (to limit forgetting).

Targets are passed as one array so the estimator works inside sklearn
pipelines and GridSearchCV:
  regression  y = column_stack([outcome, bag])
  survival    structured array with fields (event, time, bag)
The BAG labels are only used for training; prediction needs inputs only.

Outcome losses: mean squared error (cognitive score, standardised) or the Cox
negative partial log-likelihood (incident dementia, as in DeepSurv).
"""

import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.metrics import concordance_index_censored

FRAMING_KINDS = ("single", "multitask", "bottleneck", "transfer")


class _Net(nn.Module):
    def __init__(self, n_in, hidden, dropout, bottleneck):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.encoder = nn.Sequential(*layers)
        self.bag_head = nn.Linear(d, 1)
        self.bottleneck = bottleneck
        self.out_head = nn.Linear(d + int(bottleneck), 1)

    def forward(self, x, ablate=False):
        h = self.encoder(x)
        b = self.bag_head(h)
        if self.bottleneck:
            # BAG is standardised, so 0 is "an average BAG": the ablation keeps
            # the input in range while removing the participant-specific value
            h = torch.cat([h, torch.zeros_like(b) if ablate else b], dim=1)
        return self.out_head(h).squeeze(1), b.squeeze(1)


def cox_loss(risk, time, event):
    """Negative Cox partial log-likelihood (Breslow handling of ties)."""
    order = torch.argsort(time, descending=True)
    r, e = risk[order], event[order]
    log_risk_set = torch.logcumsumexp(r, dim=0)  # log sum of exp(risk) over everyone still at risk
    return -((r - log_risk_set) * e).sum() / e.sum().clamp(min=1.0)


def _split_targets(y, task):
    if task == "regression":
        y = np.asarray(y, float)
        return {"outcome": y[:, 0], "bag": y[:, 1]}
    # fields of a structured array are strided views; torch needs contiguous copies
    return {"event": np.ascontiguousarray(y["event"], bool), "time": np.ascontiguousarray(y["time"], float),
            "bag": np.ascontiguousarray(y["bag"], float)}


class NeuralFraming(BaseEstimator):
    """sklearn-compatible estimator for one RQ3 framing (see module docstring)."""

    def __init__(self, task="regression", framing="single", bag_weight=0.5, hidden=(128, 64),
                 dropout=0.2, weight_decay=1e-4, lr=3e-3, max_epochs=1000, patience=30,
                 val_fraction=0.15, finetune_lr_factor=0.1, random_state=42):
        self.task = task
        self.framing = framing
        self.bag_weight = bag_weight
        self.hidden = hidden
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.finetune_lr_factor = finetune_lr_factor
        self.random_state = random_state

    # --- training -------------------------------------------------------------
    def _outcome_loss(self, out, t, idx):
        if self.task == "regression":
            return nn.functional.mse_loss(out[idx], t["outcome"][idx])
        return cox_loss(out[idx], t["time"][idx], t["event"][idx])

    def _loss(self, phase, idx, t, X):
        out, b = self.net_(X)
        bag_loss = nn.functional.mse_loss(b[idx], t["bag"][idx])
        if phase == "bag":
            return bag_loss
        out_loss = self._outcome_loss(out, t, idx)
        if phase == "joint":
            return self.bag_weight * bag_loss + (1 - self.bag_weight) * out_loss
        return out_loss

    def _train(self, phase, param_groups, X, t, tr, va):
        opt = torch.optim.Adam(param_groups, weight_decay=self.weight_decay)
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.max_epochs):  # full-batch: the data are small
            self.net_.train()
            opt.zero_grad()
            self._loss(phase, tr, t, X).backward()
            opt.step()
            self.net_.eval()
            with torch.no_grad():
                v = float(self._loss(phase, va, t, X))
            if v < best - 1e-5:
                best, wait = v, 0
                best_state = {k: p.detach().clone() for k, p in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)

    def fit(self, X, y):
        if self.framing not in FRAMING_KINDS:
            raise ValueError(f"framing must be one of {FRAMING_KINDS}")
        torch.set_num_threads(1)  # parallelism comes from the grid search
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, np.float32)
        raw = _split_targets(y, self.task)

        # standardise targets on this training data only
        self.bag_mu_, self.bag_sd_ = raw["bag"].mean(), raw["bag"].std() or 1.0
        t = {"bag": torch.tensor((raw["bag"] - self.bag_mu_) / self.bag_sd_, dtype=torch.float32)}
        if self.task == "regression":
            self.y_mu_, self.y_sd_ = raw["outcome"].mean(), raw["outcome"].std() or 1.0
            t["outcome"] = torch.tensor((raw["outcome"] - self.y_mu_) / self.y_sd_, dtype=torch.float32)
        else:
            t["time"] = torch.tensor(raw["time"], dtype=torch.float32)
            t["event"] = torch.tensor(raw["event"], dtype=torch.float32)

        n = len(X)
        perm = rng.permutation(n)
        n_va = max(int(self.val_fraction * n), 1)
        va, tr = torch.tensor(perm[:n_va]), torch.tensor(perm[n_va:])
        if self.task == "survival" and t["event"][va].sum() < 2:
            va = tr  # too few events to early-stop on: fall back to the training loss
        Xt = torch.tensor(X)

        self.net_ = _Net(X.shape[1], tuple(self.hidden), self.dropout,
                         bottleneck=self.framing == "bottleneck")
        enc, bag_h, out_h = (list(self.net_.encoder.parameters()), list(self.net_.bag_head.parameters()),
                             list(self.net_.out_head.parameters()))
        if self.framing == "single":
            self._train("outcome", [{"params": enc + out_h, "lr": self.lr}], Xt, t, tr, va)
        elif self.framing in ("multitask", "bottleneck"):
            self._train("joint", [{"params": enc + bag_h + out_h, "lr": self.lr}], Xt, t, tr, va)
        else:  # transfer: BAG pretraining, fresh outcome head, gentle fine-tuning
            self._train("bag", [{"params": enc + bag_h, "lr": self.lr}], Xt, t, tr, va)
            self.net_.out_head.reset_parameters()
            self._train("outcome", [{"params": enc, "lr": self.lr * self.finetune_lr_factor},
                                    {"params": out_h, "lr": self.lr}], Xt, t, tr, va)

        if self.task == "survival":
            risk = self._forward(X)[0]
            self.breslow_ = BreslowEstimator().fit(risk, raw["event"], raw["time"])
        return self

    # --- prediction ---------------------------------------------------------
    def _forward(self, X, ablate=False):
        self.net_.eval()
        with torch.no_grad():
            out, b = self.net_(torch.tensor(np.asarray(X, np.float32)), ablate=ablate)
        return out.numpy().astype(float), b.numpy().astype(float)

    def predict(self, X, ablate=False):
        """Outcome prediction (cognitive score) or log-risk (dementia)."""
        out = self._forward(X, ablate)[0]
        return out * self.y_sd_ + self.y_mu_ if self.task == "regression" else out

    def predict_bag(self, X):
        """The BAG head's estimate, in years (meaningful for framings trained on BAG)."""
        return self._forward(X)[1] * self.bag_sd_ + self.bag_mu_

    def predict_survival_function(self, X):
        return self.breslow_.get_survival_function(self.predict(X))

    def score(self, X, y):
        """R2 for the cognitive score, Harrell's C-index for dementia (tuning criterion)."""
        raw = _split_targets(y, self.task)
        if self.task == "regression":
            return r2_score(raw["outcome"], self.predict(X))
        return concordance_index_censored(raw["event"], raw["time"], self.predict(X))[0]


def make_targets(task, bag, outcome=None, event=None, time=None):
    """Pack the outcome and BAG labels into the single y the estimator expects."""
    if task == "regression":
        return np.column_stack([np.asarray(outcome, float), np.asarray(bag, float)])
    y = np.empty(len(bag), dtype=[("event", bool), ("time", float), ("bag", float)])
    y["event"], y["time"], y["bag"] = np.asarray(event, bool), np.asarray(time, float), np.asarray(bag, float)
    return y
