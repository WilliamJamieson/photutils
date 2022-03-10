# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines rectangular and rectangular-annulus apertures in
both pixel and sky coordinates.
"""

import math

import astropy.units as u
import numpy as np

from .attributes import (ScalarAngle, PixelPositions, PositiveScalar, Scalar,
                         SkyCoordPositions, ScalarAngleOrPixel)
from .core import PixelAperture, SkyAperture
from .mask import ApertureMask
from ..geometry import rectangular_overlap_grid

__all__ = ['RectangularMaskMixin', 'RectangularAperture',
           'RectangularAnnulus', 'SkyRectangularAperture',
           'SkyRectangularAnnulus']


class RectangularMaskMixin:
    """
    Mixin class to create masks for rectangular or rectangular-annulus
    aperture objects.
    """

    def to_mask(self, method='exact', subpixels=5):
        """
        Return a mask for the aperture.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture on
            the pixel grid.  Not all options are available for all
            aperture types.  Note that the more precise methods are
            generally slower.  The following methods are available:

                * ``'exact'`` (default):
                  The the exact fractional overlap of the aperture and
                  each pixel is calculated.  The returned mask will
                  contain values between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out
                  of the aperture.  The returned mask will contain
                  values only of 0 (out) and 1 (in).

                * ``'subpixel'``:
                  A pixel is divided into subpixels (see the
                  ``subpixels`` keyword), each of which are considered
                  to be entirely in or out of the aperture depending on
                  whether its center is in or out of the aperture.  If
                  ``subpixels=1``, this method is equivalent to
                  ``'center'``.  The returned mask will contain values
                  between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this factor
            in each dimension.  That is, each pixel is divided into
            ``subpixels ** 2`` subpixels.

        Returns
        -------
        mask : `~photutils.aperture.ApertureMask` or list of `~photutils.aperture.ApertureMask`
            A mask for the aperture.  If the aperture is scalar then a
            single `~photutils.aperture.ApertureMask` is returned,
            otherwise a list of `~photutils.aperture.ApertureMask` is
            returned.
        """
        _, subpixels = self._translate_mask_mode(method, subpixels,
                                                 rectangle=True)

        if hasattr(self, 'w'):
            w = self.w
            h = self.h
        elif hasattr(self, 'w_out'):  # annulus
            w = self.w_out
            h = self.h_out
        else:
            raise ValueError('Cannot determine the aperture radius.')

        masks = []
        for bbox, edges in zip(np.atleast_1d(self.bbox),
                               self._centered_edges):
            ny, nx = bbox.shape
            mask = rectangular_overlap_grid(edges[0], edges[1], edges[2],
                                            edges[3], nx, ny, w, h,
                                            self.theta, 0, subpixels)

            # subtract the inner circle for an annulus
            if hasattr(self, 'w_in'):
                mask -= rectangular_overlap_grid(edges[0], edges[1], edges[2],
                                                 edges[3], nx, ny, self.w_in,
                                                 self.h_in, self.theta, 0,
                                                 subpixels)

            masks.append(ApertureMask(mask, bbox))

        if self.isscalar:
            return masks[0]
        else:
            return masks

    @staticmethod
    def _calc_extents(width, height, theta):
        """
        Calculate half of the bounding box extents of an ellipse.
        """
        half_width = width / 2.
        half_height = height / 2.
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        x_extent1 = abs((half_width * cos_theta) - (half_height * sin_theta))
        x_extent2 = abs((half_width * cos_theta) + (half_height * sin_theta))
        y_extent1 = abs((half_width * sin_theta) + (half_height * cos_theta))
        y_extent2 = abs((half_width * sin_theta) - (half_height * cos_theta))
        x_extent = max(x_extent1, x_extent2)
        y_extent = max(y_extent1, y_extent2)

        return x_extent, y_extent

    @staticmethod
    def _lower_left_positions(positions, width, height, theta):
        """
        Calculate lower-left positions from the input center positions.

        Used for creating `~matplotlib.patches.Rectangle` patch for the
        aperture.
        """
        half_width = width / 2.
        half_height = height / 2.
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        xshift = (half_height * sin_theta) - (half_width * cos_theta)
        yshift = -(half_height * cos_theta) - (half_width * sin_theta)

        return np.atleast_2d(positions) + np.array([xshift, yshift])


class RectangularAperture(RectangularMaskMixin, PixelAperture):
    """
    A rectangular aperture defined in pixel coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

            * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
            * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs
            * `~astropy.units.Quantity` instance of ``(x, y)`` pairs in
              pixel units

    w : float
        The full width of the rectangle in pixels.  For ``theta=0`` the
        width side is along the ``x`` axis.

    h : float
        The full height of the rectangle in pixels.  For ``theta=0`` the
        height side is along the ``y`` axis.

    theta : float, optional
        The rotation angle in radians of the rectangle "width" side from
        the positive ``x`` axis.  The rotation angle increases
        counterclockwise.  The default is 0.

    Raises
    ------
    ValueError : `ValueError`
        If either width (``w``) or height (``h``) is negative.

    Examples
    --------
    >>> from photutils.aperture import RectangularAperture
    >>> aper = RectangularAperture([10., 20.], 5., 3.)
    >>> aper = RectangularAperture((10., 20.), 5., 3., theta=np.pi)

    >>> pos1 = (10., 20.)  # (x, y)
    >>> pos2 = (30., 40.)
    >>> pos3 = (50., 60.)
    >>> aper = RectangularAperture([pos1, pos2, pos3], 5., 3.)
    >>> aper = RectangularAperture((pos1, pos2, pos3), 5., 3., theta=np.pi)
    """

    _shape_params = ('w', 'h', 'theta')
    positions = PixelPositions('The center pixel position(s).')
    w = PositiveScalar('The full width in pixels.')
    h = PositiveScalar('The full height in pixels.')
    theta = Scalar('The counterclockwise rotation angle in radians of the '
                   'rectangle "width" side from the positive x axis.')

    def __init__(self, positions, w, h, theta=0.):
        self.positions = positions
        self.w = w
        self.h = h
        self.theta = theta

    @property
    def _xy_extents(self):
        return self._calc_extents(self.w, self.h, self.theta)

    @property
    def area(self):
        return self.w * self.h

    def _to_patch(self, origin=(0, 0), **kwargs):
        """
        Return a `~matplotlib.patches.patch` for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : `dict`
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `~matplotlib.patches.patch` or list of `~matplotlib.patches.patch`
            A patch for the aperture.  If the aperture is scalar then a
            single `~matplotlib.patches.patch` is returned, otherwise a
            list of `~matplotlib.patches.patch` is returned.
        """
        import matplotlib.patches as mpatches

        xy_positions, patch_kwargs = self._define_patch_params(origin=origin,
                                                               **kwargs)
        xy_positions = self._lower_left_positions(xy_positions, self.w,
                                                  self.h, self.theta)

        patches = []
        theta_deg = self.theta * 180. / np.pi
        for xy_position in xy_positions:
            patches.append(mpatches.Rectangle(xy_position, self.w, self.h,
                                              theta_deg, **patch_kwargs))

        if self.isscalar:
            return patches[0]
        else:
            return patches

    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyRectangularAperture` object
        defined in celestial coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `SkyRectangularAperture` object
            A `SkyRectangularAperture` object.
        """
        return SkyRectangularAperture(**self._to_sky_params(wcs))


class RectangularAnnulus(RectangularMaskMixin, PixelAperture):
    r"""
    A rectangular annulus aperture defined in pixel coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

            * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
            * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs
            * `~astropy.units.Quantity` instance of ``(x, y)`` pairs in
              pixel units

    w_in : float
        The inner full width of the rectangular annulus in pixels.  For
        ``theta=0`` the width side is along the ``x`` axis.

    w_out : float
        The outer full width of the rectangular annulus in pixels.  For
        ``theta=0`` the width side is along the ``x`` axis.

    h_out : float
        The outer full height of the rectangular annulus in pixels.

    h_in : `None` or float
        The inner full height of the rectangular annulus in pixels.
        If `None`, then the inner full height is calculated as:

            .. math:: h_{in} = h_{out}
                \left(\frac{w_{in}}{w_{out}}\right)

        For ``theta=0`` the height side is along the ``y`` axis.

    theta : float, optional
        The rotation angle in radians of the rectangle "width" side from
        the positive ``x`` axis.  The rotation angle increases
        counterclockwise.  The default is 0.

    Raises
    ------
    ValueError : `ValueError`
        If inner width (``w_in``) is greater than outer width
        (``w_out``).

    ValueError : `ValueError`
        If either the inner width (``w_in``) or the outer height
        (``h_out``) is negative.

    Examples
    --------
    >>> from photutils.aperture import RectangularAnnulus
    >>> aper = RectangularAnnulus([10., 20.], 3., 8., 5.)
    >>> aper = RectangularAnnulus((10., 20.), 3., 8., 5., theta=np.pi)

    >>> pos1 = (10., 20.)  # (x, y)
    >>> pos2 = (30., 40.)
    >>> pos3 = (50., 60.)
    >>> aper = RectangularAnnulus([pos1, pos2, pos3], 3., 8., 5.)
    >>> aper = RectangularAnnulus((pos1, pos2, pos3), 3., 8., 5., theta=np.pi)
    """

    _shape_params = ('w_in', 'w_out', 'h_in', 'h_out', 'theta')
    positions = PixelPositions('The center pixel position(s).')
    w_in = PositiveScalar('The inner full width in pixels.')
    w_out = PositiveScalar('The outer full width in pixels.')
    h_in = PositiveScalar('The inner full height in pixels.')
    h_out = PositiveScalar('The outer full height in pixels.')
    theta = Scalar('The counterclockwise rotation angle in radians of the '
                   'rectangle "width" side from the positive x axis.')

    def __init__(self, positions, w_in, w_out, h_out, h_in=None, theta=0.):
        if not w_out > w_in:
            raise ValueError('"w_out" must be greater than "w_in"')

        self.positions = positions
        self.w_in = w_in
        self.w_out = w_out
        self.h_out = h_out

        if h_in is None:
            h_in = self.w_in * self.h_out / self.w_out
        else:
            if not h_out > h_in:
                raise ValueError('"h_out" must be greater than "h_in"')
        self.h_in = h_in

        self.theta = theta

    @property
    def _xy_extents(self):
        return self._calc_extents(self.w_out, self.h_out, self.theta)

    @property
    def area(self):
        return self.w_out * self.h_out - self.w_in * self.h_in

    def _to_patch(self, origin=(0, 0), **kwargs):
        """
        Return a `~matplotlib.patches.patch` for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : `dict`
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `~matplotlib.patches.patch` or list of `~matplotlib.patches.patch`
            A patch for the aperture.  If the aperture is scalar then a
            single `~matplotlib.patches.patch` is returned, otherwise a
            list of `~matplotlib.patches.patch` is returned.
        """
        import matplotlib.patches as mpatches

        xy_positions, patch_kwargs = self._define_patch_params(origin=origin,
                                                               **kwargs)
        inner_xy_positions = self._lower_left_positions(xy_positions,
                                                        self.w_in, self.h_in,
                                                        self.theta)
        outer_xy_positions = self._lower_left_positions(xy_positions,
                                                        self.w_out,
                                                        self.h_out,
                                                        self.theta)

        patches = []
        theta_deg = self.theta * 180. / np.pi
        for xy_in, xy_out in zip(inner_xy_positions, outer_xy_positions):
            patch_inner = mpatches.Rectangle(xy_in, self.w_in, self.h_in,
                                             theta_deg)
            patch_outer = mpatches.Rectangle(xy_out, self.w_out, self.h_out,
                                             theta_deg)
            path = self._make_annulus_path(patch_inner, patch_outer)
            patches.append(mpatches.PathPatch(path, **patch_kwargs))

        if self.isscalar:
            return patches[0]
        else:
            return patches

    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyRectangularAnnulus` object
        defined in celestial coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `SkyRectangularAnnulus` object
            A `SkyRectangularAnnulus` object.
        """
        return SkyRectangularAnnulus(**self._to_sky_params(wcs))


class SkyRectangularAperture(SkyAperture):
    """
    A rectangular aperture defined in sky coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        The celestial coordinates of the aperture center(s). This can be
        either scalar coordinates or an array of coordinates.

    w : scalar `~astropy.units.Quantity`
        The full width of the rectangle, either in angular or pixel
        units.  For ``theta=0`` the width side is along the North-South
        axis.

    h :  scalar `~astropy.units.Quantity`
        The full height of the rectangle, either in angular or pixel
        units.  For ``theta=0`` the height side is along the East-West
        axis.

    theta : scalar `~astropy.units.Quantity`, optional
        The position angle (in angular units) of the rectangle "width"
        side.  For a right-handed world coordinate system, the position
        angle increases counterclockwise from North (PA=0).  The default
        is 0 degrees.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from photutils.aperture import SkyRectangularAperture
    >>> positions = SkyCoord(ra=[10., 20.], dec=[30., 40.], unit='deg')
    >>> aper = SkyRectangularAperture(positions, 1.0*u.arcsec, 0.5*u.arcsec)
    """

    _shape_params = ('w', 'h', 'theta')
    positions = SkyCoordPositions('The center position(s) in sky coordinates.')
    w = ScalarAngleOrPixel('The full width, in angular or pixel units.')
    h = ScalarAngleOrPixel('The full height, in angular or pixel units.')
    theta = ScalarAngle('The position angle (in angular units) of the '
                        'rectangle "width" side.')

    def __init__(self, positions, w, h, theta=0.*u.deg):
        if w.unit.physical_type != h.unit.physical_type:
            raise ValueError('"w" and "h" should either both be angles or '
                             'in pixels')

        self.positions = positions
        self.w = w
        self.h = h
        self.theta = theta

    def to_pixel(self, wcs):
        """
        Convert the aperture to a `RectangularAperture` object defined
        in pixel coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `RectangularAperture` object
            A `RectangularAperture` object.
        """
        return RectangularAperture(**self._to_pixel_params(wcs))


class SkyRectangularAnnulus(SkyAperture):
    r"""
    A rectangular annulus aperture defined in sky coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        The celestial coordinates of the aperture center(s). This can be
        either scalar coordinates or an array of coordinates.

    w_in : scalar `~astropy.units.Quantity`
        The inner full width of the rectangular annulus, either in
        angular or pixel units.  For ``theta=0`` the width side is along
        the North-South axis.

    w_out : scalar `~astropy.units.Quantity`
        The outer full width of the rectangular annulus, either in
        angular or pixel units.  For ``theta=0`` the width side is along
        the North-South axis.

    h_out : scalar `~astropy.units.Quantity`
        The outer full height of the rectangular annulus, either in
        angular or pixel units.

    h_in : `None` or scalar `~astropy.units.Quantity`
        The outer full height of the rectangular annulus, either in
        angular or pixel units.  If `None`, then the inner full height
        is calculated as:

            .. math:: h_{in} = h_{out}
                \left(\frac{w_{in}}{w_{out}}\right)

        For ``theta=0`` the height side is along the East-West axis.

    theta : scalar `~astropy.units.Quantity`, optional
        The position angle (in angular units) of the rectangle "width"
        side.  For a right-handed world coordinate system, the position
        angle increases counterclockwise from North (PA=0).  The default
        is 0 degrees.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from photutils.aperture import SkyRectangularAnnulus
    >>> positions = SkyCoord(ra=[10., 20.], dec=[30., 40.], unit='deg')
    >>> aper = SkyRectangularAnnulus(positions, 3.0*u.arcsec, 8.0*u.arcsec,
    ...                              5.0*u.arcsec)
    """

    _shape_params = ('w_in', 'w_out', 'h_in', 'h_out', 'theta')
    positions = SkyCoordPositions('The center position(s) in sky coordinates.')
    w_in = ScalarAngleOrPixel('The inner full width, in angular or pixel units.')
    w_out = ScalarAngleOrPixel('The outer full width, in angular or pixel units.')
    h_in = ScalarAngleOrPixel('The inner full height, in angular or pixel units.')
    h_out = ScalarAngleOrPixel('The outer full height, in angular or pixel '
                               'units.')
    theta = ScalarAngle('The position angle (in angular units) of the rectangle '
                        '"width" side.')

    def __init__(self, positions, w_in, w_out, h_out, h_in=None,
                 theta=0.*u.deg):
        if w_in.unit.physical_type != w_out.unit.physical_type:
            raise ValueError('w_in and w_out should either both be angles or '
                             'in pixels')
        if w_out.unit.physical_type != h_out.unit.physical_type:
            raise ValueError('w_out and h_out should either both be angles '
                             'or in pixels')
        if not w_out > w_in:
            raise ValueError('"w_out" must be greater than "w_in".')

        self.positions = positions
        self.w_in = w_in
        self.w_out = w_out
        self.h_out = h_out

        if h_in is None:
            h_in = self.w_in * self.h_out / self.w_out
        else:
            if h_in.unit.physical_type != h_out.unit.physical_type:
                raise ValueError('h_in and h_out should either both be '
                                 'angles or in pixels')
            if not h_out > h_in:
                raise ValueError('"h_out" must be greater than "h_in".')
        self.h_in = h_in

        self.theta = theta

    def to_pixel(self, wcs):
        """
        Convert the aperture to a `RectangularAnnulus` object defined in
        pixel coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `RectangularAnnulus` object
            A `RectangularAnnulus` object.
        """
        return RectangularAnnulus(**self._to_pixel_params(wcs))
