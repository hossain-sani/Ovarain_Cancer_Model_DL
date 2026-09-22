import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "ACC": accuracy_score(y_true, y_pred),
        "SN": recall_score(y_true, y_pred),
        "SP": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "PRE": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if y_score is not None:
        scores = y_score[:, 1] if y_score.ndim == 2 and y_score.shape[1] == 2 else y_score
        metrics["AUC"] = roc_auc_score(y_true, scores)
    else:
        metrics["AUC"] = 0.0
    return metrics


def evaluate(model, loader, device="cpu", binary=True):
    model.eval()
    all_y, all_y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if binary:
                logits = model(x)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(1)
                prob = torch.sigmoid(logits)
                prob = prob.squeeze(-1)
            else:
                logits = model(x)
                prob = torch.softmax(logits, dim=1)[:, 1]
            all_y.append(y.numpy())
            all_y_prob.append(prob.cpu().numpy())
    y_true = pd.Series([i for batch in all_y for i in batch]).values
    y_prob = [p for batch in all_y_prob for p in batch]
    y_pred = (torch.tensor(y_prob) > 0.5).int().numpy()
    return y_true, y_pred, y_prob


def metrics_table(results):
    return pd.DataFrame(results).T


def save_predictions(y_true, y_pred, y_prob, path):
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}).to_csv(path, index=False)