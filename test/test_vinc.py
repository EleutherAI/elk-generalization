import torch
from torch import Tensor

from elk_generalization.elk.vincs import VincsReporter


def batch_cov(x: Tensor) -> Tensor:
    """Compute a batch of covariance matrices.

    Args:
        x: A tensor of shape [..., n, d].

    Returns:
        A tensor of shape [..., d, d].
    """
    x_ = x - x.mean(dim=-2, keepdim=True)
    return x_.mT @ x_ / (x_.shape[-2] - 1)


def cov_mean_fused(x: Tensor) -> Tensor:
    """Compute the mean of the covariance matrices of a batch of data matrices.

    The computation is done in a memory-efficient way, without materializing all
    the covariance matrices in VRAM.

    Args:
        x: A tensor of shape [batch, n, d].

    Returns:
        A tensor of shape [d, d].
    """
    b, n, d = x.shape

    x_ = x - x.mean(dim=1, keepdim=True)
    x_ = x_.reshape(-1, d)
    return x_.mT @ x_ / (b * (n - 1))


def test_eigen_reporter():
    num_clusters = 5
    hidden_size = 10
    N = 100

    x = torch.randn(N, num_clusters, 2, hidden_size, dtype=torch.float64)
    y = torch.randint(0, 2, (N,), dtype=torch.float64)
    x_neg, x_pos = x.unbind(2)

    reporter = VincsReporter(
        in_features=hidden_size,
        device=torch.device("cpu"),
        dtype=torch.float64,
        w_var=1.0,
        w_cov=1.0,
        w_supervised=0.0,
    )
    reporter.fit(x, y)

    # testing against https://github.com/EleutherAI/ccs/blob/main/tests/test_eigen_reporter.py
    neg_mu, pos_mu = x_neg.mean(dim=(0, 1)), x_pos.mean(dim=(0, 1))

    # Check that the covariance is correct
    neg_centroids, pos_centroids = x_neg.mean(dim=1), x_pos.mean(dim=1)
    true_cov = 0.5 * (batch_cov(neg_centroids) + batch_cov(pos_centroids))
    torch.testing.assert_close(reporter.var, true_cov)

    # Check that the negative covariance is correct
    true_xcov = (neg_centroids - neg_mu).mT @ (pos_centroids - pos_mu) / (N - 1)
    true_xcov = 0.5 * (true_xcov + true_xcov.mT)
    torch.testing.assert_close(reporter.cov, true_xcov)

    # Check that the invariance (intra-cluster variance) is correct.
    # This is actually the same whether or not we track class means.
    expected_invariance = 0.5 * (cov_mean_fused(x_neg) + cov_mean_fused(x_pos))
    torch.testing.assert_close(reporter.inv, expected_invariance)


def test_supervised_eigen_reporter():
    num_clusters = 5
    hidden_size = 10
    N = 100

    x = torch.randn(N, num_clusters, 2, hidden_size, dtype=torch.float64)
    y = torch.randint(0, 2, (N,), dtype=torch.float64)

    reporter = VincsReporter(
        in_features=hidden_size,
        device=torch.device("cpu"),
        dtype=torch.float64,
        w_var=0.0,
        w_inv=0.0,
        w_cov=0.0,
        w_supervised=1.0,
    )
    reporter.fit(x, y)

    # check that the probe weight is proportional to the difference in means
    x_cat = x.mean(dim=1).reshape(-1, hidden_size)
    y_cat = torch.cat([1 - y, y], dim=0).bool()

    true_mu, false_mu = x_cat[y_cat].mean(dim=0), x_cat[~y_cat].mean(dim=0)

    mean_diff = true_mu - false_mu
    mean_diff = mean_diff / mean_diff.norm()

    reporter_weight = (
        reporter.linear.weight.squeeze() / reporter.linear.weight.squeeze().norm()
    )
    if reporter_weight[0] * mean_diff[0] < 0:
        mean_diff = -mean_diff
    torch.testing.assert_close(mean_diff, reporter_weight)


test_eigen_reporter()
test_supervised_eigen_reporter()
