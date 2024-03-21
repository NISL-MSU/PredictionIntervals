# Code downloaded from https://github.com/tarik/pi-snm-qde/blob/master/neural_pi/estimator/functions.py
# Used only for QD+

import numpy as np
import multiprocessing as mp

from .split_normal import Randomness
from . import split_normal as sn

__all__ = [
    'mv_aggreg',
    'sem_aggreg',
    'std_aggreg',
    'snm_aggreg',
    'mean_aggreg',
    'no_aggreg'
]


# --------------------------------------------------------------------------------------------------
# AGGREGATION METHODS
# --------------------------------------------------------------------------------------------------

def snm_aggreg(y_pred_all, alpha, seed=None, **kwargs):
    """
    Parallelization of the split normal aggregation.
    """
    print('\nAggregating. Have a tea... This may take a while depending on the dataset size.')
    return _parallelized_aggreg(alpha, y_pred_all, _split_normal_aggregator, seed)


def _split_normal_aggregator(alpha, y_pred, seed, *args):
    randomness = Randomness(seed)
    mixture_params = []
    for y_l, y_u, y_p in y_pred:  # Ensemble
        std_1, std_2 = sn.fit_split_normal(alpha=alpha,
                                           q_l=y_l, q_u=y_u, mode=y_p,
                                           seed=randomness.random_seed())
        mixture_params.append(dict(
            loc=y_p,
            scale_1=std_1,
            scale_2=std_2
        ))
    y_l_agg, y_p_agg, y_u_agg = _calc_y_from_sn_mixture(alpha, mixture_params)
    return y_p_agg, y_l_agg, y_u_agg


def mv_aggreg(y_pred_all, alpha, **kwargs):
    """
    Parallelizes the aggregation of results from normal estimators (MVE) to
    a normal mixture and calculates the desired PIs.
    """
    print('\nAggregating...')
    return _parallelized_aggreg(alpha, y_pred_all, _normal_aggregator)


def _normal_aggregator(alpha, y_pred, *args):
    """
    Aggregates ensembled normals as a normal mixture and calculates the desired PIs.
    Note: We use the split normal implementation since it is a generalization
    of the normal distribution.
    """
    mixture_params = []
    for y_sigma, y_mu in y_pred:  # Ensemble
        mixture_params.append(dict(
            loc=y_mu,
            scale_1=y_sigma,
            scale_2=y_sigma
        ))
    y_l_agg, y_p_agg, y_u_agg = _calc_y_from_sn_mixture(alpha, mixture_params)
    return y_p_agg, y_l_agg, y_u_agg


def _calc_y_from_sn_mixture(alpha, mixture_params):
    """
    Returns the final PI boundaries and point estimate given the `alpha` and
    mixture distribution parameters.
    """
    y_space = _estimate_snm_space(mixture_params)

    y_p_agg = 0
    mixture_pdf = None
    for pdf_params in mixture_params:  # Ensemble of distr. parameters
        p = sn.pdf(y_space, **pdf_params)
        y_p_agg += pdf_params['loc']
        if mixture_pdf is None:
            mixture_pdf = p
        else:
            mixture_pdf += p

    y_start = y_space[0]
    y_end = y_space[-1]
    num_samples = y_space.shape[0]
    ensemble_size = len(mixture_params)

    mixture_pdf = mixture_pdf / ensemble_size
    mixture_cdf = np.cumsum(mixture_pdf * (y_end - y_start) / num_samples)
    y_l_agg = sn.numerical_ppf_from_cdf(y_space, mixture_cdf, alpha / 2)
    y_u_agg = sn.numerical_ppf_from_cdf(y_space, mixture_cdf, 1 - alpha / 2)
    y_p_agg = y_p_agg / ensemble_size  # Point estimate

    return y_l_agg, y_p_agg, y_u_agg


def _estimate_snm_space(mixture_params, p_boundary=1e-6, step=1e-4):
    mixture_params = np.array(
        [[params[k] for k in ['loc', 'scale_1', 'scale_2']] for params in mixture_params])
    x_start = np.min(sn.ppf(p=p_boundary,
                            loc=mixture_params[:, 0],
                            scale_1=mixture_params[:, 1],
                            scale_2=mixture_params[:, 2]))
    x_end = np.max(sn.ppf(p=1 - p_boundary,
                          loc=mixture_params[:, 0],
                          scale_1=mixture_params[:, 1],
                          scale_2=mixture_params[:, 2]))
    num_samples = int((x_end - x_start) / step)
    return np.linspace(x_start, x_end, num_samples)


def _parallelized_aggreg(alpha, y_pred_all, aggreg_func, seed=None, **kwargs):
    """
    Parallelization of the prediction interval aggregation.
    """
    randomness = Randomness(seed)

    # y_pred_all.shape -> (sample, ensemble, {0, 1, 2})
    y_pred_all = np.swapaxes(y_pred_all, 0, 1)
    y_pred_l, y_pred_p, y_pred_u = np.zeros((3, y_pred_all.shape[0]), dtype=np.float32)

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(aggreg_func, [(alpha, y_pred, randomness.random_seed()) for y_pred in y_pred_all])
    pool.close()

    for i in range(y_pred_all.shape[0]):  # sample
        y_pred_l[i] = results[i][1]
        y_pred_u[i] = results[i][2]
        y_pred_p[i] = results[i][0]

    return y_pred_l, y_pred_p, y_pred_u


def sem_aggreg(y_pred_all, z_factor=1.96, **kwargs):
    """
    The original QDE aggregation function according to the implementation by
    Pearce et al.
    https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals
    """
    ddof = 1 if y_pred_all.shape[0] > 1 else 0
    y_pred_l = np.mean(y_pred_all[:, :, 0], axis=0) \
               - z_factor * np.std(y_pred_all[:, :, 0], axis=0, ddof=ddof) \
               / np.sqrt(y_pred_all.shape[0])  # !
    y_pred_u = np.mean(y_pred_all[:, :, 1], axis=0) \
               + z_factor * np.std(y_pred_all[:, :, 1], axis=0, ddof=ddof) \
               / np.sqrt(y_pred_all.shape[0])  # !
    y_pred_p = np.mean(y_pred_all[:, :, 2], axis=0)
    return y_pred_l, y_pred_p, y_pred_u


def std_aggreg(y_pred_all, z_factor=1.96, **kwargs):
    """
    The original QDE aggregation function according to the paper by
    Pearce et al.
    https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals
    """
    ddof = 1 if y_pred_all.shape[0] > 1 else 0
    y_pred_l = np.mean(y_pred_all[:, :, 0], axis=0) \
               - z_factor * np.std(y_pred_all[:, :, 0], axis=0, ddof=ddof)
    y_pred_u = np.mean(y_pred_all[:, :, 1], axis=0) \
               + z_factor * np.std(y_pred_all[:, :, 1], axis=0, ddof=ddof)
    y_pred_p = np.mean(y_pred_all[:, :, 2], axis=0)
    return y_pred_l, y_pred_p, y_pred_u


def mean_aggreg(y_pred_all, **kwargs):
    y_pred_l = np.mean(y_pred_all[:, :, 0], axis=0)
    y_pred_u = np.mean(y_pred_all[:, :, 1], axis=0)
    y_pred_p = np.mean(y_pred_all[:, :, 2], axis=0)
    return y_pred_l, y_pred_p, y_pred_u


def no_aggreg(y_pred_all, **kwargs):
    y_pred_l = y_pred_all[:, :, 0]
    y_pred_u = y_pred_all[:, :, 1]
    y_pred_p = y_pred_all[:, :, 2]
    return y_pred_l, y_pred_p, y_pred_u
