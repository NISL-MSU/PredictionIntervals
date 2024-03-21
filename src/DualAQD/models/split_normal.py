# Code downloaded from https://github.com/tarik/pi-snm-qde/blob/master/neural_pi/estimator/split_normal.py
# Used only for QD+. Only works in Linux because of jax

import numpy as np
import jax.numpy as jnp
from scipy.special import erf, erfinv
from jax.scipy.special import erf as jax_erf
from jax import grad, jit
from pprint import pprint


class Randomness(np.random.RandomState):

    _SEED_MIN = 1
    _SEED_MAX = int(1e9)

    def random_seed(self, low=_SEED_MIN, high=_SEED_MAX, size=None, dtype='l'):
        return self.randint(low=low, high=high, size=size, dtype=dtype)


# __all__ = [
#     'fit_split_normal',
#     'pdf',
#     'pdf',
#     'cdf',
#     'ppf',
#     'jax_pdf',
#     'jax_cdf',
# ]

_SQRT_2 = np.sqrt(2)
_SQRT_2_PI = np.sqrt(2 / np.pi)


def pdf(x, loc, scale_1, scale_2):
    x, loc, scale_1, scale_2 = map_as_array(x, loc, scale_1, scale_2)
    scale = np.where(x <= loc, scale_1, scale_2)
    a = _SQRT_2_PI / (scale_1 + scale_2)
    return a * np.exp(- (x - loc) ** 2 / (2.0 * scale ** 2))


def cdf(x, loc, scale_1, scale_2):
    x, loc, scale_1, scale_2 = map_as_array(x, loc, scale_1, scale_2)
    scale = np.where(x <= loc, scale_1, scale_2)
    return (scale_1 + erf((x - loc) / (_SQRT_2 * scale)) * scale) / (scale_1 + scale_2)


def jax_pdf(x, loc, scale_1, scale_2):
    a = jnp.sqrt(2 / jnp.pi) / (scale_1 + scale_2)
    scale = scale_1 if x <= loc else scale_2
    return a * jnp.exp(- (x - loc) ** 2 / (2.0 * scale ** 2))


def jax_cdf(x, loc, scale_1, scale_2):
    scale = scale_1 if x <= loc else scale_2
    return (scale_1 + jax_erf((x - loc) / (jnp.sqrt(2) * scale)) * scale) / (scale_1 + scale_2)


def ppf(p, loc, scale_1, scale_2):
    """
    Percent point function or inverse of CDF or quantile function.
    """
    p, loc, scale_1, scale_2 = map_as_array(p, loc, scale_1, scale_2)
    scale = np.where(p <= cdf(loc, loc, scale_1, scale_2), scale_1, scale_2)
    a = (p * (scale_1 + scale_2) - scale_1)
    return loc + _SQRT_2 * scale * erfinv(a / scale)


def map_as_array(*args):
    return map(np.asarray, args)


def numerical_ppf_from_cdf(x_cdf, y_cdf, p):
    index = min(range(len(y_cdf)), key=lambda i: abs(y_cdf[i] - p))
    return x_cdf[index]


def loss_func(alpha, q_l, q_u, mode, std_1, std_2):
    return jnp.power(alpha / 2 - jax_cdf(q_l, mode, std_1, std_2), 2) + \
           jnp.power(1 - alpha / 2 - jax_cdf(q_u, mode, std_1, std_2), 2)


grad_loss_func = jit(grad(loss_func, argnums=(4, 5)), static_argnums=(0, 1, 2, 3))


def fit_split_normal(alpha, q_l, q_u, mode, epochs=10, lr=None, lr_min=0.0001, lr_max=0.1,
                     momentum=0., loss_limit=1e-6, num_retries=1, seed=None, verbose=False):
    """
    Fits standard deviation parameters of a split normal distribution given a prediction interval.
    :param alpha: float
    ]

        Size of the lower and upper quantile
    :param q_l: float
        Boundary of lower quantile defined by `alpha/2`.
    :param q_u: float
        Boundary of upper quantile defined by `1 - alpha/2`.
    :param mode: float
        Point estimate.
    :param epochs: int
        Maximum number of epochs to train.
    :param lr: float
        Fixed learning rate.
    :param lr_min: float
        If fixed learning rate is `None`, minimum learning rate for "one cycle" learning is required.
    :param lr_max: float
        If fixed learning rate is `None`, maximum learning rate for "one cycle" learning is required.
    :param momentum: float
        Momentum.
    :param num_retries: int
        Maximum retries if the desired `loss_limit` is not reached.
    :param loss_limit: float
        The desired loss to reach. Utilizied for early stopping and retry logic.
    :param seed: int
        Random seed.
    :return: tuple
        The standard deviations of a split normal distribution.
    """
    if verbose and (mode < q_l or q_u < mode):
        print('Mode outside the given quantiles!')

    randomness = Randomness(seed)

    std_1, std_2 = None, None
    best_std_1, best_std_2 = None, None
    best_loss = float('inf')

    for i in range(num_retries):
        if i == 0:
            std_1, std_2 = _init_scales(q_l=q_l, q_u=q_u,
                                        p_l=alpha / 2, p_u=1 - alpha / 2,
                                        mode=mode)
        else:
            # In case the fitting with the above initialization fails.
            std_1 = randomness.random()
            std_2 = randomness.random()

        loss = loss_func(alpha, q_l, q_u, mode, std_1, std_2)
        _print_progress(0, loss, verbose)
        if verbose:
            print('std_1=%f\nstd_2=%f' % (std_1, std_2))

        v_t_1 = 0
        v_t_2 = 0
        for epoch in range(epochs):
            grad_std_1, grad_std_2 = grad_loss_func(alpha, q_l, q_u, mode, std_1, std_2)
            lr_ = lr if lr else lr_min + (lr_max - lr_min) * np.sin(np.pi * epoch / (epochs - 1))

            # Momentum accelerated gradient
            v_t_1 = momentum * v_t_1 + lr_ * grad_std_1
            v_t_2 = momentum * v_t_2 + lr_ * grad_std_2
            std_1 = std_1 - v_t_1
            std_2 = std_2 - v_t_2

            loss = loss_func(alpha, q_l, q_u, mode, std_1, std_2)
            _print_progress(epoch, loss, verbose)

            if loss < best_loss:
                best_loss = loss
                best_std_1 = std_1
                best_std_2 = std_2

            if loss <= loss_limit:
                if verbose:
                    print('Early stopping...')
                break

        if loss <= loss_limit:
            break
        else:
            if verbose and i == num_retries - 1:
                pprint(dict(
                    std_1=float(best_std_1),
                    std_2=float(best_std_2),
                    q_l=q_l,
                    q_u=q_u,
                    mode=mode,
                    loss=float(best_loss)
                ))
            if verbose:
                print('Retry...')

    return best_std_1, best_std_2


def _init_scales(q_l, q_u, p_l, p_u, mode):
    """
    Initializes the parameters of a split normal distribution, i.e. `std_1` and `std_2`.
    """
    return _calc_normal_scale(q_l, mode, p_l), _calc_normal_scale(q_u, mode, p_u)


def _calc_normal_scale(q, mean, p):
    """
    Calculates `sigma` of a normal distribution given a quantile and mean.
    """
    scale = (q - mean) / (np.sqrt(2) * erfinv(2 * p - 1))
    return scale


def _print_progress(epoch, loss, verbose=False):
    if verbose:
        print('ep: %4d  loss %.12f' % (epoch, loss))