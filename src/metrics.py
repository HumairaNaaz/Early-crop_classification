"""OWNER: Eval Lead (Divyanshi). Accuracy, macro/weighted F1, per-class F1 (+support), confusion
matrix, rotated-field accuracy, rare-class (<0.2% frequency) F1."""


def compute_metrics(y_true, y_pred, y_prev=None, n_classes: int = 18):
    raise NotImplementedError("TODO(Divyanshi)")
