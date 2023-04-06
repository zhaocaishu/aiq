import numpy as np


def reg2multilabel(logits, min=-0.1, step=0.01, num_classes=20):
    labels = np.zeros((logits.shape[0], num_classes), dtype=np.int32)
    for i, logit in enumerate(logits):
        labels[i, 0:int((logit - min) / step)+1] = 1
    return labels


def multilabel2reg(probs, min=-0.1, step=0.01):
    logits = ((probs > 0.5).cumprod(axis=1).sum(axis=1) - 1) * step + min
    return logits
