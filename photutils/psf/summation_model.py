"""
This module defines a helper model to define a summation of models.

Notes
-----
This should be upstreamed and generalized to astropy.modeling.
"""
from __future__ import annotations

import numpy as np
from astropy.modeling import Model, Parameter


class SummationModel(Model):
    """
    A model to sum a model set.

    Suppose that we have
        m_1, m_2, ..., m_n

    models, all with the same "base" model (think Gaussian2D for example). We
    then want to work with the model that is the summation of all these
    constituent models. Using the builtin modeling binary operators, in theory
        m = m_1 + m_2 + ... + m_n

    should work. However, for large n this can become intractable for several
    reasons such as memory usage and the fact that this creates a massively
    nested CompoundModel,
        CompoundModel(CompoundModel(CompoundModel(...)), m_n)

    Not only does this have tractibility issues, but it also will have
    performance issues because there will be no way to vectorize the evaluation
    of the common m_i form.

    This model takes this issue and pushes much of the logistical work onto the
    "model set" built into astropy.modeling. This model then wraps that model
    and sums all the individual results from the model set to accomplish the
    summation of the models.

    Parameters
    ----------
    model_set : Model
        An astropy.modeling.Model that is defined as a model set. The
        individual models in the set will act as the terms of the summation.
    bounds : dict, optional
        The xy_bounds passed to the psf fitting routine for the x/y values.
    names : list, optional
        The names that each term should be given as distinct models.
    """

    _param_names = ()

    def __init__(self,
                 model_set: Model,
                 bounds: dict[str, tuple] | None = None,
                 names: list[str] | None = None) -> None:
        self._model_set = model_set
        self._make_parameters(bounds)
        self._names = names

        super().__init__()

    def _term(self, index: int) -> Model:
        params = {
            name: getattr(self, f'{name}_{index}').value
            for name in self._model_set.param_names
        }
        mdl = self._model_set.__class__(**params)
        mdl.name = self._names[index] if self._names is not None else None

        return mdl

    @property
    def terms(self) -> list[Model]:
        """
        Construct the individual terms of the summation model as models.
        """
        return [self._term(index) for index in range(self.n_terms)]

    @property
    def fittable(self):
        """
        Whether the model is fittable or not.
        """
        return self._model_set.fittable

    @property
    def fit_deriv(self):
        """
        Define the fit_deriv method for the model.
        """
        # TODO: Implement this better
        return None

    @property
    def n_terms(self):
        """
        The number of terms in the summation.
        """
        return len(self._model_set)

    @property
    def param_names(self):
        """
        The name of the parameters in the model.
        """
        return self._param_names

    def _create_parameter(self, name: str, parameter: str, index: int) -> None:
        # Note: to allow parameters and attribute array to freely exchange
        #       values
        #   _getter forces reading value from attribute array
        #   _setter forces setting value to attribute array

        # The getter function needs to take in a `value` argument to match the
        # required signature for the Parameter class's getter. This is because
        # it is intended to perform an operation on what value the parameter
        # is storing rather than getting the value of the parameter itself.
        #
        # Here we are using it to directly get the value from inside the
        # model_set
        def getter(value):  # noqa: ARG001
            return getattr(self._model_set, parameter)[index]

        def setter(value):
            getattr(self._model_set, parameter)[index] = value

            return value

        set_param = getattr(self._model_set, parameter)
        value = set_param[index]

        param = Parameter(
            name,
            getter=getter,
            setter=setter,
            default=value,
            unit=set_param.unit,
            fixed=set_param.fixed,
            bounds=set_param.bounds,
            prior=set_param.prior,
            tied=set_param.tied,
        )
        param.model = self
        param.value = value

        return param

    def _make_parameters(self, bounds: dict[str, tuple] | None = None) -> None:
        self._param_names = []
        for parameter in self._model_set.param_names:
            for index in range(self.n_terms):
                name = f'{parameter}_{index}'
                param = self._create_parameter(name, parameter, index)

                setattr(self, name, param)
                self._parameters_[name] = param
                self._param_names.append(name)

        if bounds is not None:
            for name, bound in bounds.items():
                param = getattr(self, name)
                param.bounds = (param.value - bound[0], param.value + bound[1])

    @property
    def n_inputs(self):
        """
        The number of inputs to the summation model.
        """
        return self._model_set.n_inputs

    def evaluate(self, *args, **kwargs):
        """
        Evaluate the summation model.

        Parameters
        ----------
        *args : tuple
            The positional arguments to the model.
        **kwargs : dict
            The keyword arguments to the model.
        """
        input_args = args[:self.n_inputs]
        param_args = args[self.n_inputs:]

        set_args = [np.array([arg] * self.n_terms).T for arg in input_args]
        set_args.extend(
            [
                np.array(
                    param_args[index * self.n_terms:(index + 1) * self.n_terms]
                ) for index in range(len(self._model_set.param_names))
            ]
        )

        results = self._model_set.evaluate(*set_args, **kwargs).T

        return results.sum(axis=0)
