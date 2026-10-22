import numpy as np
import pandas as pd
import pytest

from aiq.dataset.processor import CSNeutralize
from aiq.utils.functional import neutralize


def make_frame(industry, cap, date="2024-01-02", **factors):
    frame = pd.DataFrame({"industry": industry, "cap": cap, **factors})
    frame.columns = pd.MultiIndex.from_product([["feature"], frame.columns])
    frame.index = pd.MultiIndex.from_product(
        [[date], range(len(frame))], names=["Date", "Instrument"]
    )
    return frame


def run(frame, **kwargs):
    return neutralize(
        frame,
        industry_col="industry",
        cap_col="cap",
        factor_cols=[r"^F\d$"],
        **kwargs,
    )


@pytest.mark.parametrize("n_samples", [99, 100])
def test_default_threshold_is_100_effective_samples(n_samples):
    industry = np.arange(n_samples) % 2
    cap = np.arange(n_samples, dtype=float) % 7
    frame = make_frame(industry, cap, F1=3 + industry + cap * 2 + np.sin(cap))

    result = run(frame)["feature", "F1"]

    if n_samples == 99:
        assert result.isna().all()
    else:
        assert np.isfinite(result).all()


def test_threshold_counts_only_reference_rows_with_valid_exposures_and_factor():
    industry = np.arange(105) % 2
    cap = np.arange(105, dtype=float) % 7
    frame = make_frame(industry, cap, F1=3 + industry + cap * 2)
    reference = pd.Series(True, index=frame.index[:102])
    frame.loc[frame.index[0], ("feature", "industry")] = np.nan
    frame.loc[frame.index[1], ("feature", "cap")] = np.inf
    frame.loc[frame.index[2], ("feature", "F1")] = np.nan

    # 102 reference rows minus three invalid rows: the fit must be rejected.
    assert run(frame, reference_mask=reference)["feature", "F1"].isna().all()

    frame.loc[frame.index[2], ("feature", "F1")] = 3 + industry[2] + cap[2] * 2
    result = run(frame, reference_mask=reference)["feature", "F1"]

    assert result.iloc[:2].isna().all()
    assert np.isfinite(result.iloc[2:]).all()


def test_reference_residuals_are_orthogonal_and_outside_values_do_not_affect_fit():
    industry = np.repeat([0, 1, 2], 10)
    cap = np.tile(np.arange(10, dtype=float), 3)
    frame = make_frame(industry, cap, F1=industry * 3 + cap * 2 + np.sin(np.arange(30)))
    reference = pd.Series(np.arange(30) % 10 != 9, index=frame.index)

    result = run(frame, min_samples=8, reference_mask=reference)
    residuals = result.loc[reference, ("feature", "F1")].to_numpy()
    for group in np.unique(industry):
        np.testing.assert_allclose(
            residuals[industry[reference] == group].sum(), 0, atol=1e-6
        )
    np.testing.assert_allclose(cap[reference] @ residuals, 0, atol=1e-6)

    changed = frame.copy()
    changed.loc[~reference, ("feature", "F1")] += 10000
    changed_result = run(changed, min_samples=8, reference_mask=reference)
    np.testing.assert_allclose(
        result.loc[reference, ("feature", "F1")],
        changed_result.loc[reference, ("feature", "F1")],
        atol=1e-6,
    )


@pytest.mark.parametrize("empty_series", [False, True])
def test_empty_reference_pool_returns_nan_instead_of_raw_values(empty_series):
    frame = make_frame([0] * 6, np.arange(6, dtype=float), F1=np.arange(6) + 10.0)
    reference = pd.Series(False, index=frame.index[:0] if empty_series else frame.index)

    result = run(frame, min_samples=3, reference_mask=reference)

    assert result["feature", "F1"].isna().all()


def test_invalid_exposures_and_factors_become_nan_instead_of_raw_values():
    frame = make_frame(np.arange(12) % 2, np.arange(12, dtype=float), F1=np.arange(12) + 10.0)
    frame.loc[frame.index[0], ("feature", "industry")] = np.nan
    frame.loc[frame.index[1:3], ("feature", "cap")] = [np.nan, np.inf]
    frame.loc[frame.index[3:6], ("feature", "F1")] = [np.nan, np.inf, -np.inf]

    result = run(frame, min_samples=4)

    assert result["feature", "F1"].iloc[:6].isna().all()
    assert np.isfinite(result["feature", "F1"].iloc[6:]).all()


@pytest.mark.parametrize("industry,cap", [([0, 0], [1.0, 2.0]), ([0, 1], [1.0, 1.0])])
def test_no_residual_degrees_of_freedom_returns_nan(industry, cap):
    frame = make_frame(industry, cap, F1=[10.0, 20.0])

    assert run(frame, min_samples=2)["feature", "F1"].isna().all()


def test_size_collinear_with_industry_is_rejected_instead_of_extrapolated():
    frame = make_frame(
        [0] * 3 + [1] * 3 + [0, 1],
        [1.0] * 3 + [2.0] * 3 + [3.0, 4.0],
        F1=[3.0, 4.0, 5.0, 10.0, 12.0, 14.0, 20.0, 30.0],
    )
    reference = pd.Series(True, index=frame.index[:6])

    result = run(frame, min_samples=3, reference_mask=reference)

    assert result["feature", "F1"].isna().all()


def test_unseen_industry_is_nan_but_supported_outside_rows_are_transformed():
    # The absent reference industry is the would-be dropped baseline category.
    industry = np.array([1] * 6 + [2] * 6 + [0, 1])
    cap = np.array(list(range(6)) * 2 + [3, 4], dtype=float)
    factor = 10 + industry * 5 + cap * 2
    factor[-2:] += 7
    frame = make_frame(industry, cap, F1=factor)
    reference = pd.Series(True, index=frame.index[:12])

    result = run(frame, min_samples=6, reference_mask=reference)["feature", "F1"]

    np.testing.assert_allclose(result.iloc[:12], 0, atol=1e-6)
    assert np.isnan(result.iloc[12])
    assert result.iloc[13] == pytest.approx(7)


@pytest.mark.parametrize("min_samples", [3, 4])
def test_suffix_preserves_input_and_original_factors_on_success_and_failure(min_samples):
    frame = make_frame([0] * 4, [1.0, 2.0, 3.0, 4.0], F1=[2.0, 3.0, 5.0, np.nan])
    original = frame.copy(deep=True)

    result = run(frame, add_suffix=True, suffix="_RES", min_samples=min_samples)

    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(result[original.columns], original)
    if min_samples == 4:
        assert result["feature", "F1_RES"].isna().all()
    else:
        np.testing.assert_allclose(
            result["feature", "F1_RES"].iloc[:3], [1 / 6, -1 / 3, 1 / 6], atol=1e-6
        )
        assert np.isnan(result["feature", "F1_RES"].iloc[3])


@pytest.mark.parametrize("min_samples", [3, 4])
def test_each_factor_uses_its_own_valid_reference_rows_and_supported_industries(min_samples):
    industry = np.repeat([0, 1], 3)
    cap = np.tile([1.0, 2.0, 3.0], 2)
    factor = 5 + industry * 10 + cap * 2
    sparse_factor = factor.copy()
    sparse_factor[3:5] = np.nan
    frame = make_frame(industry, cap, F1=factor, F2=sparse_factor)
    reference = pd.Series(
        [True, True, True, True, True, pd.NA], index=frame.index, dtype="boolean"
    )

    result = run(frame, min_samples=min_samples, reference_mask=reference)

    np.testing.assert_allclose(result["feature", "F1"], 0, atol=1e-6)
    if min_samples == 4:
        assert result["feature", "F2"].isna().all()
    else:
        np.testing.assert_allclose(result["feature", "F2"].iloc[:3], 0, atol=1e-6)
        assert result["feature", "F2"].iloc[3:].isna().all()


def test_csneutralize_applies_default_100_effective_reference_samples_per_date():
    first_cap = np.arange(101, dtype=float)
    first_day = make_frame(
        np.arange(101) % 2, first_cap, F1=3 + first_cap * 2 + np.sin(first_cap)
    )
    second_cap = np.arange(100, dtype=float)
    second_day = make_frame(
        np.arange(100) % 2, second_cap, date="2024-01-03",
        F1=103 + second_cap * 2 + np.sin(second_cap),
    )
    frame = pd.concat([first_day, second_day])
    reference = pd.Series(
        True, index=first_day.index[:100].append(second_day.index[:99])
    )
    processor = CSNeutralize(
        fields_group="feature", industry_col="industry", cap_col="cap",
        factor_cols=["^F1$"],
    )

    result = processor(frame, reference_mask=reference)

    pd.testing.assert_index_equal(result.index, frame.index)
    assert np.isfinite(result["feature", "F1"].iloc[:101]).all()
    np.testing.assert_allclose(result["feature", "F1"].iloc[:100].sum(), 0, atol=1e-6)
    assert result["feature", "F1"].iloc[101:].isna().all()


@pytest.mark.parametrize("industry_only", [False, True])
def test_single_exposure_paths_do_not_require_the_unused_exposure(industry_only):
    if industry_only:
        expected = np.array([-1.0, 0, 1, -2, 0, 2])
        industry = np.repeat([0, 1], 3)
        frame = make_frame(industry, [np.nan] * 6, F1=3 + industry * 5 + expected)
    else:
        expected = np.array([1.0, -2, 1, 1, -2, 1])
        cap = np.arange(6, dtype=float)
        frame = make_frame([np.nan] * 6, cap, F1=3 + cap * 2 + expected)

    result = neutralize(
        frame,
        industry_col="industry" if industry_only else None,
        cap_col=None if industry_only else "cap",
        factor_cols=["^F1$"],
        min_samples=3,
    )

    np.testing.assert_allclose(result["feature", "F1"], expected, atol=1e-6)
