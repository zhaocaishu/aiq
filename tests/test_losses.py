import pytest
import torch

from aiq.losses import HybridLoss, ICLoss, TopKLoss


@pytest.mark.parametrize("direction, expected", [(1.0, 0.0), (-1.0, 2.0)])
def test_ic_loss_perfect_correlation(direction, expected):
    targets = torch.tensor([[-2.0], [-0.5], [1.0], [4.0]])
    preds = (direction * targets + 3.0).requires_grad_()

    loss = ICLoss()(preds, targets)

    torch.testing.assert_close(loss, torch.tensor(expected))
    loss.backward()
    assert torch.isfinite(preds.grad).all()


def test_ic_loss_matches_pearson_and_is_affine_invariant():
    preds = torch.tensor([[1.0], [-2.0], [4.0], [0.5]], requires_grad=True)
    targets = torch.tensor([[0.5], [3.0], [-1.0], [2.0]])
    criterion = ICLoss()
    expected = 1.0 - torch.corrcoef(torch.stack([preds.flatten(), targets.flatten()]))[0, 1]

    torch.testing.assert_close(criterion(preds, targets), expected)
    torch.testing.assert_close(
        criterion(3.0 * preds + 7.0, 2.0 * targets - 5.0), expected
    )


@pytest.mark.parametrize(
    "size, value", [(0, 1.0), (1, 1.0), (4, 1.0), (31, 0.1), (31, 1e10)]
)
def test_ic_loss_uninformative_targets_return_differentiable_zero(size, value):
    preds = torch.arange(size, dtype=torch.float32).reshape(-1, 1).requires_grad_()
    targets = torch.full((size, 1), value)

    loss = ICLoss()(preds, targets)

    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(preds.grad, torch.zeros_like(preds))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_ic_loss_constant_predictions_keep_finite_ranking_gradients(dtype):
    preds = torch.zeros(4, 1, dtype=dtype, requires_grad=True)
    targets = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]], dtype=dtype)

    loss = ICLoss()(preds, targets)

    assert loss.item() == 1.0
    loss.backward()
    assert torch.isfinite(preds.grad).all()
    assert preds.grad[0].item() > 0
    assert preds.grad[-1].item() < 0


def test_ic_loss_ignores_nonfinite_pairs():
    preds = torch.tensor(
        [[1.0], [float("nan")], [2.0], [float("inf")], [3.0]], requires_grad=True
    )
    targets = torch.tensor([[1.0], [2.0], [float("nan")], [4.0], [2.0]])

    loss = ICLoss()(preds, targets)

    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    assert torch.isfinite(preds.grad).all()
    torch.testing.assert_close(preds.grad[1:4], torch.zeros(3, 1))


def test_ic_loss_all_invalid_pairs_return_differentiable_zero():
    preds = torch.tensor([[float("nan")], [float("inf")]], requires_grad=True)
    targets = torch.tensor([[1.0], [2.0]])

    loss = ICLoss()(preds, targets)

    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(preds.grad, torch.zeros_like(preds))


def test_ic_loss_gradcheck():
    preds = torch.tensor(
        [[0.5], [-1.0], [2.0], [0.1]], dtype=torch.float64, requires_grad=True
    )
    targets = torch.tensor([[1.0], [0.2], [-1.0], [3.0]], dtype=torch.float64)

    assert torch.autograd.gradcheck(ICLoss(), (preds, targets))


def test_ic_loss_near_constant_targets_keep_finite_gradients():
    preds = torch.tensor([[0.0], [1.0], [-1.0], [2.0]], requires_grad=True)
    targets = torch.tensor([[0.0], [1e-5], [2e-5], [3e-5]])

    loss = ICLoss()(preds, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(preds.grad).all()
    assert preds.grad.abs().sum().item() > 0


@pytest.mark.parametrize("epsilon", [0.0, -1e-8, float("nan"), float("inf")])
def test_ic_loss_rejects_invalid_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        ICLoss(epsilon=epsilon)


def test_ic_loss_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same number"):
        ICLoss()(torch.ones(3, 1), torch.ones(2, 1))


def test_hybrid_loss_uses_ic_auxiliary_and_preserves_weighting():
    preds = torch.linspace(-1.0, 1.0, 40).reshape(-1, 1).requires_grad_()
    targets = torch.linspace(1.0, -1.0, 40).reshape(-1, 1)
    criterion = HybridLoss(aux_loss_weight=0.3)

    assert isinstance(criterion.aux_loss_fn, ICLoss)
    expected = TopKLoss()(preds, targets) + 0.3 * ICLoss()(preds, targets)
    loss = criterion(preds, targets)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(preds.grad).all()
