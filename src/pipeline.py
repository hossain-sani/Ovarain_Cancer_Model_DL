import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from .config import Config
from .data import build_datasets, make_dataloaders
from .eval import evaluate, compute_metrics, save_predictions
from .models import AttResNet50, AttCNN


def train_feature_extractor(cfg, train_loader, val_loader, device="cpu"):
    model = AttResNet50(num_classes=2, dropout=cfg.fe_dropout, dropout2=cfg.fe_dropout2).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.fe_lr, weight_decay=cfg.fe_weight_decay, betas=(cfg.fe_momentum, 0.999)
    )
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(cfg.fe_epochs):
        model.train()
        running_loss, total = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            total += x.size(0)
        tr_loss = running_loss / total
        val_loss, val_acc, val_auc = _validate(model, val_loader, device)
        print(f"[FE] epoch {epoch+1}/{cfg.fe_epochs} loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _validate(model, loader, device="cpu"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, total = 0.0, 0
    all_y, all_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            total += x.size(0)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_y.extend(y.cpu().numpy())
            all_prob.extend(prob)
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(np.asarray(all_y), np.asarray(all_prob))
    pred = (np.asarray(all_prob) > 0.5).astype(int)
    acc = (pred == np.asarray(all_y)).mean()
    return running_loss / total, acc, auc


@torch.no_grad()
def extract_features(model, loader, device="cpu"):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        _, f = model(x)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats, 0), np.concatenate(labels, 0)


def select_features_shap(X_tr, y_tr, n_selected=500, num_round=100, seed=42):
    model = xgb.XGBClassifier(
        n_estimators=num_round,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        use_label_encoder=False,
    )
    model.fit(X_tr, y_tr)
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_tr)
    if isinstance(shap_values, list):
        shap_values = np.asarray(shap_values[1]) if len(shap_values) == 2 else np.asarray(shap_values[-1])
    importance = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(-importance)[:n_selected]
    return top_idx, model, importance


def select_features_pca(X_tr, y_tr, n_selected=500, seed=42):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(n_selected, X_tr.shape[1]))
    pca.fit(X_tr)
    loadings = np.abs(pca.components_).sum(axis=0)
    top_idx = np.argsort(-loadings)[:n_selected]
    return top_idx, pca, loadings


def select_features_lasso(X_tr, y_tr, n_selected=500, seed=42):
    from sklearn.linear_model import Lasso

    lasso = Lasso(alpha=0.001, max_iter=10000, random_state=seed)
    lasso.fit(X_tr, y_tr)
    coef = np.abs(lasso.coef_)
    top_idx = np.argsort(-coef)[:n_selected]
    return top_idx, lasso, coef


def select_features_mrmr(X_tr, y_tr, n_selected=500, seed=42):
    n, d = X_tr.shape
    selected = []
    remaining = list(range(d))
    mi = np.zeros(d)
    for j in remaining:
        mi[j] = _mutual_information(X_tr[:, j], y_tr)
    while len(selected) < min(n_selected, d):
        best_j, best_score = None, -1e18
        for j in remaining:
            red = 0.0
            for s in selected:
                red += _mutual_information(X_tr[:, j], X_tr[:, s])
            score = mi[j] - red / max(1, len(selected))
            if score > best_score:
                best_score, best_j = score, j
        selected.append(best_j)
        remaining.remove(best_j)
    return np.asarray(selected), None, np.asarray(mi)


def _mutual_information(x, y):
    from sklearn.feature_selection import mutual_info_classif

    return mutual_info_classif(x.reshape(-1, 1), y, random_state=42)[0]


def fit_baseline_classifiers(X_tr, y_tr, X_te, y_te, seed=42):
    clfs = {
        "DT": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, eval_metric="logloss", random_state=seed, use_label_encoder=False
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=seed),
    }
    results = {}
    for name, clf in clfs.items():
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = clf.predict(X_te)
        m = compute_metrics(y_te, y_pred, np.asarray(y_prob))
        m["time(s)"] = round(time.time() - t0, 4)
        results[name] = m
    return results


def train_attcnn(cfg, X_tr, y_tr, X_va, y_va, X_te, y_te, device="cpu"):
    in_dim = X_tr.shape[1]
    model = AttCNN(in_dim, hidden=cfg.cls_hidden, dropout=cfg.cls_dropout, dropout2=cfg.cls_dropout2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.cls_lr)
    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32)
    train_ds = torch.utils.data.TensorDataset(to_tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32))
    val_ds = torch.utils.data.TensorDataset(to_tensor(X_va), torch.tensor(y_va, dtype=torch.float32))
    test_ds = torch.utils.data.TensorDataset(to_tensor(X_te), torch.tensor(y_te, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.cls_batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.cls_batch, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.cls_batch, shuffle=False)
    best_auc, best_state, best_metrics = 0.0, None, None
    for epoch in range(cfg.cls_epochs):
        model.train()
        running_loss, total = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            total += x.size(0)
        y_true, y_pred, y_prob = _eval_attcnn(model, val_loader, device)
        m = compute_metrics(y_true, y_pred, np.asarray(y_prob))
        if m["AUC"] > best_auc:
            best_auc = m["AUC"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = m
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"[AttCNN] epoch {epoch+1}/{cfg.cls_epochs} loss={running_loss/total:.4f} val_auc={m['AUC']:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, y_pred, y_prob = _eval_attcnn(model, test_loader, device)
    m = compute_metrics(y_true, y_pred, np.asarray(y_prob))
    m["time(s)"] = round(cfg.cls_epochs * 0.0, 4)
    return model, m


def _eval_attcnn(model, loader, device="cpu"):
    model.eval()
    all_y, all_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).squeeze(1)
            prob = torch.sigmoid(logits).cpu().numpy()
            all_y.extend(y.numpy())
            all_prob.extend(prob)
    y_prob = np.asarray(all_prob)
    y_pred = (y_prob > 0.5).astype(int)
    return np.asarray(all_y), y_pred, y_prob


def run_pipeline(cfg: Config, device="cpu", root="Dataset", outdir="output"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("== Step 1: Loading + preprocessing ==")
    train_ds, val_ds, test_ds = build_datasets(cfg, root=root)
    train_loader, val_loader, test_loader = make_dataloaders(cfg, train_ds, val_ds, test_ds)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    print("== Step 2: Feature extraction via AttResNet50 ==")
    fe_model = train_feature_extractor(cfg, train_loader, val_loader, device)
    torch.save(fe_model.state_dict(), outdir / "attresnet50.pt")
    X_tr, y_tr = extract_features(fe_model, train_loader, device)
    X_va, y_va = extract_features(fe_model, val_loader, device)
    X_te, y_te = extract_features(fe_model, test_loader, device)
    print(f"features train={X_tr.shape} val={X_va.shape} test={X_te.shape}")
    np.savez(outdir / "features.npz", X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, X_te=X_te, y_te=y_te)

    print(f"== Step 3: Feature selection ({cfg.fs_method}, top {cfg.n_selected}) ==")
    if cfg.fs_method == "shap":
        top_idx, fs_model, _ = select_features_shap(X_tr, y_tr, cfg.n_selected, cfg.num_round, cfg.seed)
    elif cfg.fs_method == "pca":
        top_idx, fs_model, _ = select_features_pca(X_tr, y_tr, cfg.n_selected, cfg.seed)
    elif cfg.fs_method == "lasso":
        top_idx, fs_model, _ = select_features_lasso(X_tr, y_tr, cfg.n_selected, cfg.seed)
    elif cfg.fs_method == "mrmr":
        top_idx, fs_model, _ = select_features_mrmr(X_tr, y_tr, cfg.n_selected, cfg.seed)
    else:
        raise ValueError(cfg.fs_method)
    np.save(outdir / "selected_features_idx.npy", top_idx)
    X_tr_s, X_va_s, X_te_s = X_tr[:, top_idx], X_va[:, top_idx], X_te[:, top_idx]
    print(f"selected features: train={X_tr_s.shape}")

    print("== Step 4: Classifiers ==")
    sc = StandardScaler().fit(X_tr_s)
    X_tr_s, X_va_s, X_te_s = sc.transform(X_tr_s), sc.transform(X_va_s), sc.transform(X_te_s)
    baseline = fit_baseline_classifiers(X_tr_s, y_tr, X_te_s, y_te, cfg.seed)
    attcnn_model, attcnn_metrics = train_attcnn(cfg, X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te, device)
    torch.save(attcnn_model.state_dict(), outdir / "attcnn.pt")
    results = dict(baseline)
    results["AttCNN"] = attcnn_metrics
    return results