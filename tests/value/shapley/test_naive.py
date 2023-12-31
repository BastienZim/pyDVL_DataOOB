import logging

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from pydvl.utils import GroupedDataset, MemcachedConfig, Utility
from pydvl.value.shapley.naive import (
    combinatorial_exact_shapley,
    permutation_exact_shapley,
)

from .. import check_total_value, check_values

log = logging.getLogger(__name__)


# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "num_samples, fun, rtol, total_atol",
    [
        (12, combinatorial_exact_shapley, 0.01, 1e-5),
        (6, permutation_exact_shapley, 0.01, 1e-5),
    ],
)
def test_analytic_exact_shapley(num_samples, analytic_shapley, fun, rtol, total_atol):
    """Compares the combinatorial exact shapley and permutation exact shapley with
    the analytic_shapley calculation for a dummy model.
    """
    u, exact_values = analytic_shapley
    values_p = fun(u, progress=False)
    check_total_value(u, values_p, atol=total_atol)
    check_values(values_p, exact_values, rtol=rtol)


@pytest.mark.parametrize(
    "a, b, num_points, scorer",
    [
        (2, 0, 20, "r2"),
        (2, 1, 20, "r2"),
        (2, 1, 20, "neg_median_absolute_error"),
        (2, 1, 20, "explained_variance"),
    ],
)
def test_linear(
    linear_dataset, memcache_client_config, scorer, rtol=0.01, total_atol=1e-5
):
    linear_utility = Utility(
        LinearRegression(),
        data=linear_dataset,
        scorer=scorer,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )

    values_combinatorial = combinatorial_exact_shapley(linear_utility, progress=False)
    check_total_value(linear_utility, values_combinatorial, atol=total_atol)

    values_permutation = permutation_exact_shapley(linear_utility, progress=False)
    check_total_value(linear_utility, values_permutation, atol=total_atol)

    check_values(values_combinatorial, values_permutation, rtol=rtol)


@pytest.mark.parametrize(
    "a, b, num_points, num_groups, scorer",
    [(2, 0, 50, 3, "r2"), (2, 1, 100, 5, "r2"), (2, 1, 100, 5, "explained_variance")],
)
def test_grouped_linear(
    linear_dataset,
    num_groups,
    memcache_client_config,
    scorer,
    rtol=0.01,
    total_atol=1e-5,
):
    # assign groups recursively
    data_groups = np.random.randint(0, num_groups, len(linear_dataset))

    grouped_linear_dataset = GroupedDataset.from_dataset(linear_dataset, data_groups)
    grouped_linear_utility = Utility(
        LinearRegression(),
        data=grouped_linear_dataset,
        scorer=scorer,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )
    values_combinatorial = combinatorial_exact_shapley(
        grouped_linear_utility, progress=False
    )
    check_total_value(grouped_linear_utility, values_combinatorial, atol=total_atol)

    values_permutation = permutation_exact_shapley(
        grouped_linear_utility, progress=False
    )
    check_total_value(grouped_linear_utility, values_permutation, atol=total_atol)

    check_values(values_combinatorial, values_permutation, rtol=rtol)


@pytest.mark.parametrize(
    "a, b, num_points, scorer",
    [
        (2, 1, 20, "explained_variance"),
        (2, 0, 20, "r2"),
        (2, 1, 20, "neg_median_absolute_error"),
        (2, 1, 20, "r2"),
    ],
)
def test_linear_with_outlier(
    linear_dataset, memcache_client_config, scorer, total_atol=1e-5
):
    outlier_idx = np.random.randint(len(linear_dataset.y_train))
    linear_dataset.y_train[outlier_idx] -= 100
    linear_utility = Utility(
        LinearRegression(),
        data=linear_dataset,
        scorer=scorer,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )
    values = permutation_exact_shapley(linear_utility, progress=False)
    values.sort()
    check_total_value(linear_utility, values, atol=total_atol)

    assert values.indices[0] == outlier_idx


@pytest.mark.parametrize(
    "coefficients, scorer",
    [
        (np.random.randint(-3, 3, size=3), "r2"),
        (np.random.randint(-3, 3, size=5), "neg_median_absolute_error"),
        (np.random.randint(-3, 3, size=7), "explained_variance"),
    ],
)
def test_polynomial(
    polynomial_dataset,
    polynomial_pipeline,
    memcache_client_config,
    scorer,
    rtol=0.01,
    total_atol=1e-5,
):
    dataset, _ = polynomial_dataset
    poly_utility = Utility(
        polynomial_pipeline,
        dataset,
        scorer=scorer,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )

    values_combinatorial = combinatorial_exact_shapley(poly_utility, progress=False)
    check_total_value(poly_utility, values_combinatorial, atol=total_atol)

    values_permutation = permutation_exact_shapley(poly_utility, progress=False)
    check_total_value(poly_utility, values_permutation, atol=total_atol)

    check_values(values_combinatorial, values_permutation, rtol=rtol)


@pytest.mark.parametrize(
    "coefficients, scorer",
    [
        (np.random.randint(-3, 3, size=3), "r2"),
        (np.random.randint(-3, 3, size=3), "neg_median_absolute_error"),
        (np.random.randint(-3, 3, size=3), "explained_variance"),
    ],
)
def test_polynomial_with_outlier(
    polynomial_dataset,
    polynomial_pipeline,
    memcache_client_config,
    scorer,
    total_atol=1e-5,
):
    dataset, _ = polynomial_dataset
    outlier_idx = np.random.randint(len(dataset.y_train))
    dataset.y_train[outlier_idx] *= 100
    poly_utility = Utility(
        polynomial_pipeline,
        dataset,
        scorer=scorer,
        cache_options=MemcachedConfig(client_config=memcache_client_config),
    )

    shapley_values = permutation_exact_shapley(poly_utility, progress=False)
    check_total_value(poly_utility, shapley_values, atol=total_atol)

    assert shapley_values[0].index == outlier_idx
