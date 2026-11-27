import torch
from torch.nn import functional as F

from aiq.losses.mr_loss import MarginRankingLoss


def test_head_loss_weights_pairs_by_return_rank_distance():
    loss = MarginRankingLoss(top_k=1, head_weight=1.0, num_pairs=None)
    preds = torch.tensor([2.0, 1.0, 0.0, -1.0, 3.0], requires_grad=True)
    targets = torch.tensor([1.0, 3.0, 0.0, 2.0, float("nan")])

    # The last stock has no label. Stock 0 heads the remaining four.
    gaps = (preds[0] - preds[1:4]) / preds[:4].std(unbiased=False)
    pairs = F.softplus(gaps * torch.tensor([1.0, -1.0, 1.0]))
    expected = (2 * pairs[0] + pairs[1] + pairs[2]) / 4

    actual = loss(preds, targets)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, preds)[0], torch.autograd.grad(expected, preds)[0]
    )
