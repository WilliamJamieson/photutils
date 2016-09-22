# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functions for centroiding sources and measuring their morphological
properties.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import warnings

import numpy as np
from astropy.modeling.models import Gaussian1D, Gaussian2D, Const1D, Const2D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.utils.exceptions import AstropyUserWarning

from ..morphology import data_properties


__all__ = ['GaussianConst2D', 'centroid_com', 'gaussian1d_moments',
           'marginalize_data2d', 'centroid_1dg', 'centroid_2dg',
           'fit_2dgaussian']


class _GaussianConst1D(Const1D + Gaussian1D):
    """A model for a 1D Gaussian plus a constant."""


class GaussianConst2D(Const2D + Gaussian2D):
    """
    A model for a 2D Gaussian plus a constant.

    Parameters
    ----------
    amplitude_0 : float
        Value of the constant.
    amplitude_1 : float
        Amplitude of the Gaussian.
    x_mean_1 : float
        Mean of the Gaussian in x.
    y_mean_1 : float
        Mean of the Gaussian in y.
    x_stddev_1 : float
        Standard deviation of the Gaussian in x.
        ``x_stddev`` and ``y_stddev`` must be specified unless a covariance
        matrix (``cov_matrix``) is input.
    y_stddev_1 : float
        Standard deviation of the Gaussian in y.
        ``x_stddev`` and ``y_stddev`` must be specified unless a covariance
        matrix (``cov_matrix``) is input.
    theta_1 : float, optional
        Rotation angle in radians. The rotation angle increases
        counterclockwise.
    cov_matrix_1 : ndarray, optional
        A 2x2 covariance matrix. If specified, overrides the ``x_stddev``,
        ``y_stddev``, and ``theta`` specification.
    """


def _convert_image(data, mask=None):
    """
    Convert the input data to a float64 (double) `numpy.ndarray`,
    required for input to `skimage.measure.moments` and
    `skimage.measure.moments_central`.

    The input ``data`` is copied unless it already has that
    `numpy.dtype`.

    If ``mask`` is input, then masked pixels are set to zero in the
    output ``data``.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are set to zero in the output ``data``.

    Returns
    -------
    image : `~numpy.ndarray` (float64)
        The converted 2D array of the image, where masked pixels have
        been set to zero.
    """

    if mask is None:
        copy = False
    else:
        copy = True

    data = np.asarray(data).astype(np.float, copy=copy)

    if mask is not None:
        mask = np.asarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape')
        data[mask] = 0.0

    return data


def centroid_com(data, mask=None):
    """
    Calculate the centroid of a 2D array as its "center of mass"
    determined from image moments.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    xcen, ycen : float
        The (x, y) coordinates of the centroid.
    """

    from skimage.measure import moments

    data = _convert_image(data, mask=mask)
    m = moments(data, 1)
    xcen = m[1, 0] / m[0, 0]
    ycen = m[0, 1] / m[0, 0]

    return xcen, ycen


def gaussian1d_moments(data, mask=None):
    """
    Estimate 1D Gaussian parameters from the moments of 1D data.

    This function can be useful for providing initial parameter values
    when fitting a 1D Gaussian to the ``data``.

    Parameters
    ----------
    data : array_like (1D)
        The 1D array.

    mask : array_like (1D bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    amplitude, mean, stddev : float
        The estimated parameters of a 1D Gaussian.
    """

    if mask is not None:
        mask = np.asanyarray(mask)
        data = data.copy()
        data[mask] = 0.
    x = np.arange(data.size)
    x_mean = np.sum(x * data) / np.sum(data)
    x_stddev = np.sqrt(abs(np.sum(data * (x - x_mean)**2) / np.sum(data)))
    amplitude = np.nanmax(data) - np.nanmin(data)
    return amplitude, x_mean, x_stddev


def marginalize_data2d(data, error=None, mask=None):
    """
    Generate the marginal x and y distributions from a 2D data array.

    Parameters
    ----------
    data : array_like
        The 2D data array.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    marginal_data : list of `~numpy.ndarray`
        The marginal x and y distributions of the input ``data``.

    marginal_error : list of `~numpy.ndarray`
        The marginal x and y distributions of the input ``error``.

    marginal_mask : list of `~numpy.ndarray` (bool)
        The marginal x and y distributions of the input ``mask``.
    """

    if error is not None:
        marginal_error = np.array(
            [np.sqrt(np.sum(error**2, axis=i)) for i in [0, 1]])
    else:
        marginal_error = [None, None]

    if mask is not None:
        mask = np.asanyarray(mask)
        marginal_mask = [np.sum(mask, axis=i).astype(np.bool) for i in [0, 1]]
    else:
        marginal_mask = [None, None]

    marginal_data = [np.sum(data, axis=i) for i in [0, 1]]

    return marginal_data, marginal_error, marginal_mask


def centroid_1dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting 1D Gaussians to the
    marginal x and y distributions of the array.

    Parameters
    ----------
    data : array_like
        The 2D data array.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    xcen, ycen : float
        (x, y) coordinates of the centroid.
    """

    data = np.ma.masked_invalid(data)
    if mask is not None:
        mask = np.asanyarray(mask)
        data.mask |= mask

    if error is not None:
        error = np.ma.masked_invalid(error)
        data.mask |= error.mask
    else:
        error = np.ma.masked_array(np.ones_like(data))

    yx_data = np.array([np.ma.sum(data, axis=i) for i in [0, 1]])

    error.mask = data.mask
    error.fill_value = 1.e5
    error = error.filled()
    yx_error = np.array([np.sqrt(np.ma.sum(error**2, axis=i))
                         for i in [0, 1]])

    yx_weights = [(1.0 / yx_error[i].clip(min=1.e-30)) for i in [0, 1]]

    constant_init = np.min(data)
    centroid = []
    for (data_i, weights_i) in zip(yx_data, yx_weights):
        params_init = gaussian1d_moments(data_i)
        g_init = _GaussianConst1D(constant_init, *params_init)
        fitter = LevMarLSQFitter()
        x = np.arange(data_i.size)
        g_fit = fitter(g_init, x, data_i.data, weights=weights_i)
        centroid.append(g_fit.mean_1.value)

    return np.array(centroid)


def centroid_2dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting a 2D Gaussian (plus
    a constant) to the array.

    Parameters
    ----------
    data : array_like
        The 2D data array.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    xcen, ycen : float
        (x, y) coordinates of the centroid.
    """

    gfit = fit_2dgaussian(data, error=error, mask=mask)
    return gfit.x_mean_1.value, gfit.y_mean_1.value


def fit_2dgaussian(data, error=None, mask=None):
    """
    Fit a 2D Gaussian plus a constant to a 2D image.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    result : A `GaussianConst2D` model instance.
        The best-fitting Gaussian 2D model.
    """

    if data.size < 7:
        warnings.warn('data array must have a least 7 values to fit a 2D '
                      'Gaussian plus a constant', AstropyUserWarning)
        return None

    if error is not None:
        weights = 1.0 / error
    else:
        weights = None

    if mask is not None:
        mask = np.asanyarray(mask)
        if weights is None:
            weights = np.ones_like(data)
        # down-weight masked pixels
        weights[mask] = 1.e-30

    # Subtract the minimum of the data as a crude background estimate.
    # This will also make the data values positive, preventing issues with
    # the moment estimation in data_properties (moments from negative data
    # values can yield undefined Gaussian parameters, e.g. x/y_stddev).
    shift = np.min(data)
    data = np.copy(data) - shift
    props = data_properties(data, mask=mask)
    init_values = np.array([props.xcentroid.value, props.ycentroid.value,
                            props.semimajor_axis_sigma.value,
                            props.semiminor_axis_sigma.value,
                            props.orientation.value])

    init_const = 0.    # subtracted data minimum above
    init_amplitude = np.nanmax(data) - np.nanmin(data)
    g_init = GaussianConst2D(init_const, init_amplitude, *init_values)
    fitter = LevMarLSQFitter()
    y, x = np.indices(data.shape)
    gfit = fitter(g_init, x, y, data, weights=weights)
    gfit.amplitude_0 = gfit.amplitude_0 + shift
    return gfit
