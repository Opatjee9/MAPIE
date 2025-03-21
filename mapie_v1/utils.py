import copy
from typing import Union, Tuple, cast, Optional, Iterable
from collections.abc import Iterable as IterableType

from mapie._typing import ArrayLike, NDArray
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import train_test_split
from decimal import Decimal


def transform_confidence_level_to_alpha(
    confidence_level: float,
) -> float:
    # Using decimals to avoid weird-looking float approximations
    # when computing alpha = 1 - confidence_level
    # Such approximations arise even with simple confidence levels like 0.9
    confidence_level_decimal = Decimal(str(confidence_level))
    alpha_decimal = Decimal("1") - confidence_level_decimal
    return float(alpha_decimal)


def transform_confidence_level_to_alpha_list(
    confidence_level: Union[float, Iterable[float]]
) -> Iterable[float]:
    if isinstance(confidence_level, IterableType):
        confidence_levels = confidence_level
    else:
        confidence_levels = [confidence_level]
    return [
        transform_confidence_level_to_alpha(confidence_level)
        for confidence_level in confidence_levels
    ]


# Could be replaced by using sklearn _validate_params (ideally wrapping it)
def check_if_param_in_allowed_values(
    param: str, param_name: str, allowed_values: list
) -> None:
    if param not in allowed_values:
        raise ValueError(
            f"'{param}' option not valid for parameter '{param_name}'"
            f"Available options are: {allowed_values}"
        )


def check_cv_not_string(cv: Union[int, str, BaseCrossValidator]) -> None:
    if isinstance(cv, str):
        raise ValueError(
            "'cv' string options not available in MAPIE >= v1.0.0"
        )


def cast_point_predictions_to_ndarray(
    point_predictions: Union[NDArray, Tuple[NDArray, NDArray]]
) -> NDArray:
    if isinstance(point_predictions, tuple):
        raise TypeError(
            "Developer error: use this function to cast point predictions only, "
            "not points + intervals."
        )
    return cast(NDArray, point_predictions)


def cast_predictions_to_ndarray_tuple(
    predictions: Union[NDArray, Tuple[NDArray, NDArray]]
) -> Tuple[NDArray, NDArray]:
    if not isinstance(predictions, tuple):
        raise TypeError(
            "Developer error: use this function to cast predictions containing points "
            "and intervals, not points only."
        )
    return cast(Tuple[NDArray, NDArray], predictions)


def prepare_params(params: Union[dict, None]) -> dict:
    return copy.deepcopy(params) if params else {}


def prepare_fit_params_and_sample_weight(
    fit_params: Union[dict, None]
) -> Tuple[dict, Optional[ArrayLike]]:
    fit_params_ = prepare_params(fit_params)
    sample_weight = fit_params_.pop("sample_weight", None)
    return fit_params_, sample_weight


def raise_error_if_previous_method_not_called(
    current_method_name: str,
    previous_method_name: str,
    was_previous_method_called: bool,
) -> None:
    if not was_previous_method_called:
        raise ValueError(
            f"Incorrect method order: call {previous_method_name} "
            f"before calling {current_method_name}."
        )


def raise_error_if_method_already_called(
    method_name: str,
    was_method_called: bool,
) -> None:
    if was_method_called:
        raise ValueError(
            f"{method_name} method already called. "
            f"MAPIE does not currently support calling {method_name} several times."
        )


def raise_error_if_fit_called_in_prefit_mode(
    is_mode_prefit: bool,
) -> None:
    if is_mode_prefit:
        raise ValueError(
            "The fit method must be skipped when the prefit parameter is set to True. "
            "Use the conformalize method directly after instanciation."
        )


def super_train_test_split(
    X: NDArray,
    y: NDArray,
    train_size: float = None,
    conformalize_size: float = None,
    test_size: float = None,
    random_state: int = None,
    shuffle: bool = True,
    stratify: list = None,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Split arrays or matrices into random train, conformalize and test subsets.

    Quick utility that wraps two calls to sklearn.model_selection.train_test_split
    for splitting data into 3 sets in one line.

    Parameters
    ----------
    X : indexable with same type and length / shape[0] than "y"
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.

    y : indexable with same type and length / shape[0] than "X"
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.
        Read more in the :ref:`User Guide <stratification>`.

    Returns
    -------
    X_train, X_conformalize, X_test, y_train, y_conformalize, y_test : 
        6 array-like splits of inputs.
        output types are the same as the input types.

    Examples
    --------

    """
    train_size, test_size = _set_proportions(
        train_size, conformalize_size, test_size
    )

    X_train, X_test_conformalize, y_train, y_test_conformalize = train_test_split(
        X, y, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify
    )
    X_conformalize, X_test, y_conformalize, y_test = train_test_split(
        X_test_conformalize, y_test_conformalize, test_size=test_size/(1-train_size), random_state=random_state,
        shuffle=shuffle, stratify=stratify
    )

    return X_train, X_conformalize, X_test, y_train, y_conformalize, y_test

# Tell nose that train_test_split is not a test.
# (Needed for external libraries that may use nose.)
# Use setattr to avoid mypy errors when monkeypatching.
setattr(train_test_split, "__test__", False)


# problème dans les proportions : faire en sorte qu'elles soient à un nombre de décimales fini
# En entrée : le train_size + la proportion conf/test sur les données restantes ?

def _set_proportions(
        train_size=None, conformalize_size=None, test_size=None):

    count_input_proportions = sum(
        x is not None for x in [test_size, train_size, conformalize_size])

    if count_input_proportions == 3:
        if round(test_size + train_size + conformalize_size, 10) != 1:
            raise ValueError("The sum of the 3 input proportions must be 1.")
        return train_size, test_size

    if sum(x for x in [test_size, train_size, conformalize_size] if x) >= 1:
        raise ValueError("The input proportion must be < 1.")

    if count_input_proportions == 0:
        train_size = 0.6
        test_size = 0.2

    elif count_input_proportions == 1:
        if test_size:
            train_size = 0.8 * (1 - test_size)
        elif train_size:
            test_size = 0.5 * (1 - train_size)
        elif conformalize_size:
            train_size = 0.8 * (1 - conformalize_size)
            test_size = 1 - train_size - conformalize_size

    elif count_input_proportions == 2:
        if not test_size:
            test_size = 1 - train_size - conformalize_size
        elif not train_size:
            train_size = 1 - test_size - conformalize_size

    return train_size, test_size
