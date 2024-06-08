# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides utilities for PSF-fitting photometry.
"""

import re

import numpy as np
from astropy.modeling import CompoundModel
from astropy.modeling.models import Const2D, Identity, Shift
from astropy.nddata import NDData

__all__ = ['make_psf_model', 'grid_from_epsfs']


def _interpolate_missing_data(data, mask, method='cubic'):
    """
    Interpolate missing data as identified by the ``mask`` keyword.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        An array containing the 2D image.

    mask : 2D bool `~numpy.ndarray`
        A 2D boolean mask array with the same shape as the input
        ``data``, where a `True` value indicates the corresponding
        element of ``data`` is masked. The masked data points are those
        that will be interpolated.

    method : {'cubic', 'nearest'}, optional
        The method of used to interpolate the missing data:

        * ``'cubic'``:  Masked data are interpolated using 2D cubic
            splines. This is the default.

        * ``'nearest'``:  Masked data are interpolated using
            nearest-neighbor interpolation.

    Returns
    -------
    data_interp : 2D `~numpy.ndarray`
        The interpolated 2D image.
    """
    from scipy import interpolate

    data_interp = np.array(data, copy=True)

    if len(data_interp.shape) != 2:
        raise ValueError("'data' must be a 2D array.")

    if mask.shape != data.shape:
        raise ValueError("'mask' and 'data' must have the same shape.")

    y, x = np.indices(data_interp.shape)
    xy = np.dstack((x[~mask].ravel(), y[~mask].ravel()))[0]
    z = data_interp[~mask].ravel()

    if method == 'nearest':
        interpol = interpolate.NearestNDInterpolator(xy, z)
    elif method == 'cubic':
        interpol = interpolate.CloughTocher2DInterpolator(xy, z)
    else:
        raise ValueError('Unsupported interpolation method.')

    xy_missing = np.dstack((x[mask].ravel(), y[mask].ravel()))[0]
    data_interp[mask] = interpol(xy_missing)

    return data_interp


class _InverseShift(Shift):
    """
    A model that is the inverse of the normal
    `astropy.modeling.functional_models.Shift` model.
    """

    @staticmethod
    def evaluate(x, offset):
        return x - offset

    @staticmethod
    def fit_deriv(x, *params):
        """
        One dimensional Shift model derivative with respect to parameter.
        """
        d_offset = -np.ones_like(x)
        return [d_offset]


def _integrate_model(model, x_name=None, y_name=None, dx=50, dy=50,
                     subsample=100, use_dblquad=False):
    """
    Integrate a model over a 2D grid.

    By default, the model is discretized on a grid of size ``dx``
    x ``dy`` from the model center with a subsampling factor of
    ``subsample``. The model is then integrated over the grid using
    trapezoidal integration.

    If the ``use_dblquad`` keyword is set to `True`, then the model is
    integrated using `scipy.integrate.dblquad`. This is *much* slower
    than the default integration of the evaluated model, but it is more
    accurate. Also, note that the ``dblquad`` integration can sometimes
    fail, e.g., return zero for a non-zero model. This can happen when
    the model function is sharply localized relative to the size of the
    integration interval.

    Parameters
    ----------
    model : `~astropy.modeling.Fittable2DModel`
        The Astropy 2D model.

    x_name : str or `None`, optional
        The name of the ``model`` parameter that corresponds to the
        x-axis center of the PSF. This parameter is required if
        ``use_dblquad`` is `False` and ignored if ``use_dblquad`` is
        `True`.

    y_name : str or `None`, optional
        The name of the ``model`` parameter that corresponds to the
        y-axis center of the PSF. This parameter is required if
        ``use_dblquad`` is `False` and ignored if ``use_dblquad`` is
        `True`.

    dx, dy : odd int, optional
        The size of the integration grid in x and y. Must be odd.
        These keywords are ignored if ``use_dblquad`` is `True`.

    subsample : int, optional
        The subsampling factor for the integration grid along each axis.
        Each pixel will be sampled ``subsample`` x ``subsample`` times.
        This keyword is ignored if ``use_dblquad`` is `True`.

    use_dblquad : bool, optional
        If `True`, then use `scipy.integrate.dblquad` to integrate the
        model. This is *much* slower than the default integration of
        the evaluated model, but it is more accurate.

    Returns
    -------
    integral : float
        The integral of the model over the 2D grid.
    """
    if use_dblquad:
        from scipy.integrate import dblquad

        return dblquad(model, -np.inf, np.inf, -np.inf, np.inf)[0]

    from scipy.integrate import trapezoid

    if dx <= 0 or dy <= 0:
        raise ValueError('dx and dy must be > 0')
    if subsample < 1:
        raise ValueError('subsample must be >= 1')

    xc = getattr(model, x_name)
    yc = getattr(model, y_name)

    if np.any(~np.isfinite((xc.value, yc.value))):
        raise ValueError('model x and y positions must be finite')

    hx = (dx - 1) / 2
    hy = (dy - 1) / 2
    nxpts = int(dx * subsample)
    nypts = int(dy * subsample)
    xvals = np.linspace(xc - hx, xc + hx, nxpts)
    yvals = np.linspace(yc - hy, yc + hy, nypts)

    # evaluate the model on the subsampled grid
    data = model(xvals.reshape(-1, 1), yvals.reshape(1, -1))

    # now integrate over the subsampled grid (first over x, then over y)
    int_func = trapezoid

    return int_func([int_func(row, xvals) for row in data], yvals)


def _shift_model_param(model, param_name, shift=2):
    if isinstance(model, CompoundModel):
        # for CompoundModel, add "shift" to the parameter suffix
        out = re.search(r'(.*)_([\d]*)$', param_name)
        new_name = out.groups()[0] + '_' + str(int(out.groups()[1]) + 2)
    else:
        # simply add the shift to the parameter name
        new_name = param_name + '_' + str(shift)

    return new_name


def make_psf_model(model, *, x_name=None, y_name=None, flux_name=None,
                   normalize=True, dx=50, dy=50, subsample=100,
                   use_dblquad=False):
    """
    Make a PSF model that can be used with the PSF photometry classes
    (`PSFPhotometry` or `IterativePSFPhotometry`) from an Astropy
    fittable 2D model.

    If the ``x_name``, ``y_name``, or ``flux_name`` keywords are input,
    this function will map those ``model`` parameter names to ``x_0``,
    ``y_0``, or ``flux``, respectively.

    If any of the ``x_name``, ``y_name``, or ``flux_name`` keywords
    are `None`, then a new parameter will be added to the model
    corresponding to the missing parameter. Any new position parameters
    will be set to a default value of 0, and any new flux parameter will
    be set to a default value of 1.

    The output PSF model will have ``x_name``, ``y_name``, and
    ``flux_name`` attributes that contain the name of the corresponding
    model parameter.

    .. note::

        This function is needed only in cases where the 2D PSF model
        does not have ``x_0``, ``y_0``, and ``flux`` parameters.

        It is *not* needed for any of the PSF models provided
        by Photutils (e.g., `~photutils.psf.GriddedPSFModel`,
        `~photutils.psf.IntegratedGaussianPRF`).

    Parameters
    ----------
    model : `~astropy.modeling.Fittable2DModel`
        An Astropy fittable 2D model to use as a PSF.

    x_name : `str` or `None`, optional
        The name of the ``model`` parameter that corresponds to the x
        center of the PSF. If `None`, the model will be assumed to be
        centered at x=0, and a new model parameter called ``xpos_0``
        will be added for the x position.

    y_name : `str` or `None`, optional
        The name of the ``model`` parameter that corresponds to the
        y center of the PSF. If `None`, the model will be assumed
        to be centered at y=0, and a new parameter called ``ypos_1``
        will be added for the y position.

    flux_name : `str` or `None`, optional
        The name of the ``model`` parameter that corresponds to the
        total flux of a source. If `None`, a new model parameter called
        ``flux_3`` will be added for model flux.

    normalize : bool, optional
        If `True`, the input ``model`` will be integrated and rescaled
        so that its sum integrates to 1. This normalization occurs only
        once for the input ``model``. If the total flux of ``model``
        somehow depends on (x, y) position, then one will need to
        correct the fitted model fluxes for this effect.

    dx, dy : odd int, optional
        The size of the integration grid in x and y for normalization.
        Must be odd. These keywords are ignored if ``normalize`` is
        `False` or ``use_dblquad`` is `True`.

    subsample : int, optional
        The subsampling factor for the integration grid along each axis
        for normalization. Each pixel will be sampled ``subsample`` x
        ``subsample`` times. This keyword is ignored if ``normalize`` is
        `False` or ``use_dblquad`` is `True`.

    use_dblquad : bool, optional
        If `True`, then use `scipy.integrate.dblquad` to integrate the
        model for normalization. This is *much* slower than the default
        integration of the evaluated model, but it is more accurate.
        This keyword is ignored if ``normalize`` is `False`.

    Returns
    -------
    result : `~astropy.modeling.CompoundModel`
        A PSF model that can be used with the PSF photometry classes.
        The returned model will always be an Astropy compound model.

    Notes
    -----
    To normalize the model, by default it is discretized on a grid of
    size ``dx`` x ``dy`` from the model center with a subsampling factor
    of ``subsample``. The model is then integrated over the grid using
    trapezoidal integration.

    If the ``use_dblquad`` keyword is set to `True`, then the model is
    integrated using `scipy.integrate.dblquad`. This is *much* slower
    than the default integration of the evaluated model, but it is more
    accurate. Also, note that the ``dblquad`` integration can sometimes
    fail, e.g., return zero for a non-zero model. This can happen when
    the model function is sharply localized relative to the size of the
    integration interval.
    """
    input_model = model.copy()

    if x_name is None:
        x_model = _InverseShift(0, name='x_position')
        # "offset" is the _InverseShift parameter name;
        # the x inverse shift model is always the first submodel
        x_name = 'offset_0'
    else:
        if x_name not in input_model.param_names:
            raise ValueError(f'{x_name!r} parameter name not found in the '
                             'input model.')

        x_model = Identity(1)
        x_name = _shift_model_param(input_model, x_name, shift=2)

    if y_name is None:
        y_model = _InverseShift(0, name='y_position')
        # "offset" is the _InverseShift parameter name;
        # the y inverse shift model is always the second submodel
        y_name = 'offset_1'
    else:
        if y_name not in input_model.param_names:
            raise ValueError(f'{y_name!r} parameter name not found in the '
                             'input model.')

        y_model = Identity(1)
        y_name = _shift_model_param(input_model, y_name, shift=2)

    x_model.fittable = True
    y_model.fittable = True
    psf_model = (x_model & y_model) | input_model

    if flux_name is None:
        psf_model *= Const2D(1.0, name='flux')
        # "amplitude" is the Const2D parameter name;
        # the flux scaling is always the last component
        flux_name = psf_model.param_names[-1]
    else:
        flux_name = _shift_model_param(input_model, flux_name, shift=2)

    if normalize:
        integral = _integrate_model(psf_model, x_name=x_name, y_name=y_name,
                                    dx=dx, dy=dy, subsample=subsample,
                                    use_dblquad=use_dblquad)

        if integral == 0:
            raise ValueError('Cannot normalize the model because the '
                             'integrated flux is zero.')

        psf_model *= Const2D(1.0 / integral, name='normalization_scaling')

    # fix all the output model parameters that are not x, y, or flux
    for name in psf_model.param_names:
        psf_model.fixed[name] = name not in (x_name, y_name, flux_name)

    # final check that the x, y, and flux parameter names are in the
    # output model
    names = (x_name, y_name, flux_name)
    for name in names:
        if name not in psf_model.param_names:
            raise ValueError(f'{name!r} parameter name not found in the '
                             'output model.')

    # set the parameter names for the PSF photometry classes
    psf_model.x_name = x_name
    psf_model.y_name = y_name
    psf_model.flux_name = flux_name

    # set aliases
    psf_model.x_0 = getattr(psf_model, x_name)
    psf_model.y_0 = getattr(psf_model, y_name)
    psf_model.flux = getattr(psf_model, flux_name)

    return psf_model


def grid_from_epsfs(epsfs, grid_xypos=None, meta=None):
    """
    Create a GriddedPSFModel from a list of EPSFModels.

    Given a list of EPSFModels, this function will return a
    GriddedPSFModel. The fiducial points for each input EPSFModel can
    either be set on each individual model by setting the 'x_0' and
    'y_0' attributes, or provided as a list of tuples (``grid_xypos``).
    If a ``grid_xypos`` list is provided, it must match the length of
    input EPSFs. In either case, the fiducial points must be on a grid.

    Optionally, a ``meta`` dictionary may be provided for the output
    GriddedPSFModel. If this dictionary contains the keys 'grid_xypos',
    'oversampling', or 'fill_value', they will be overridden.

    Note: If set on the input EPSFModel (x_0, y_0), then ``origin``
    must be the same for each input EPSF. Additionally data units and
    dimensions must be for each input EPSF, and values for ``flux`` and
    ``oversampling``, and ``fill_value`` must match as well.

    Parameters
    ----------
    epsfs : list of `photutils.psf.models.EPSFModel`
        A list of EPSFModels representing the individual PSFs.
    grid_xypos : list, optional
        A list of fiducial points (x_0, y_0) for each PSF. If not
        provided, the x_0 and y_0 of each input EPSF will be considered
        the fiducial point for that PSF. Default is None.
    meta : dict, optional
        Additional metadata for the GriddedPSFModel. Note that, if
        they exist in the supplied ``meta``, any values under the keys
        ``grid_xypos`` , ``oversampling``, or ``fill_value`` will be
        overridden. Default is None.

    Returns
    -------
    GriddedPSFModel: `photutils.psf.GriddedPSFModel`
        The gridded PSF model created from the input EPSFs.
    """
    # prevent circular imports
    from photutils.psf import EPSFModel, GriddedPSFModel

    # optional, to store fiducial from input if `grid_xypos` is None
    x_0s = []
    y_0s = []
    data_arrs = []
    oversampling = None
    fill_value = None
    dat_unit = None
    origin = None
    flux = None

    # make sure, if provided, that ``grid_xypos`` is the same length as
    # ``epsfs``
    if grid_xypos is not None:
        if len(grid_xypos) != len(epsfs):
            raise ValueError('``grid_xypos`` must be the same length as '
                             '``epsfs``.')

    # loop over input once
    for i, epsf in enumerate(epsfs):

        # check input type
        if not isinstance(epsf, EPSFModel):
            raise ValueError('All input `epsfs` must be of type '
                             '`photutils.psf.models.EPSFModel`.')

        # get data array from EPSF
        data_arrs.append(epsf.data)

        if i == 0:
            oversampling = epsf.oversampling

            # same for fill value and flux, grid will have a single value
            # so it should be the same for all input, and error if not.
            fill_value = epsf.fill_value

            # check that origins are the same
            if grid_xypos is None:
                origin = epsf.origin

            flux = epsf.flux

            # if there's a unit, those should also all be the same
            try:
                dat_unit = epsf.data.unit
            except AttributeError:
                pass  # just keep as None

        else:
            if np.any(epsf.oversampling != oversampling):
                raise ValueError('All input EPSFModels must have the same '
                                 'value for ``oversampling``.')

            if epsf.fill_value != fill_value:
                raise ValueError('All input EPSFModels must have the same '
                                 'value for ``fill_value``.')

            if epsf.data.ndim != data_arrs[0].ndim:
                raise ValueError('All input EPSFModels must have data with '
                                 'the same dimensions.')

            try:
                unitt = epsf.data_unit
                if unitt != dat_unit:
                    raise ValueError('All input data must have the same unit.')
            except AttributeError as exc:
                if dat_unit is not None:
                    raise ValueError('All input data must have the same '
                                     'unit.') from exc

            if epsf.flux != flux:
                raise ValueError('All input EPSFModels must have the same '
                                 'value for ``flux``.')

        if grid_xypos is None:  # get gridxy_pos from x_0, y_0 if not provided
            x_0s.append(epsf.x_0.value)
            y_0s.append(epsf.y_0.value)

            # also check that origin is the same, if using x_0s and y_0s
            # from input
            if epsf.origin != origin:
                raise ValueError('If using ``x_0``, ``y_0`` as fiducial point,'
                                 '``origin`` must match for each input EPSF.')

    # if not supplied, use from x_0, y_0 of input EPSFs as fiducuals
    # these are checked when GriddedPSFModel is created to make sure they
    # are actually on a grid.
    if grid_xypos is None:
        grid_xypos = list(zip(x_0s, y_0s))

    data_cube = np.stack(data_arrs, axis=0)

    if meta is None:
        meta = {}
    # add required keywords to meta
    meta['grid_xypos'] = grid_xypos
    meta['oversampling'] = oversampling
    meta['fill_value'] = fill_value

    data = NDData(data_cube, meta=meta)

    grid = GriddedPSFModel(data, fill_value=fill_value)

    return grid
