"""Historical membership must determine cross-sectional transform parameters."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from aiq.dataset.dataset import TSDataset
from aiq.dataset.handler import Alpha158
from aiq.dataset.loader import DataLoader
from aiq.dataset.processor import (
    CrossSectionalProcessor, CSNeutralize, CSWinsorize, CSZScoreNorm,
    CSRankNorm, RobustZScoreNorm, Fillna, Dropna,
)


class HistoricalPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.dates = ["2023-01-03", "2023-01-04", "2023-01-05"]
        self.members = pd.DataFrame(
            [(date, inst) for date in self.dates for inst in ["A", "B"]]
            + [(self.dates[-1], "C")],
            columns=["Date", "Instrument"],
        )
        self.data = pd.DataFrame(
            {"FACTOR": [0.0, 10.0, 1000.0] * 3,
             "RET": [0.0, 0.1, 100.0] * 3},
            index=pd.MultiIndex.from_product(
                [self.dates, ["A", "B", "C"]], names=["Date", "Instrument"]
            ),
        )

    def make_handler(self, members=None, instruments="INDEX"):
        if members is None:
            members = self.members
        with patch.object(DataLoader, "load_instruments", return_value=members):
            with patch.object(DataLoader, "load_calendar", return_value=self.dates):
                return Alpha158(
                    data_dir="unused", instruments=instruments,
                    start_time=self.dates[0], end_time=self.dates[-1],
                    fit_start_time=self.dates[0], fit_end_time=self.dates[-1],
                )

    def process(self, handler, data, mode="train"):
        return handler.process(
            data.copy(), feature_names=["FACTOR"], label_names=["RET"],
            processors=[
                CSWinsorize("feature", lower_quantile=0.25, upper_quantile=0.75),
                CSWinsorize("label", lower_quantile=0.25, upper_quantile=0.75),
            ], mode=mode,
        )

    def test_future_member_does_not_change_past_features_or_labels(self):
        original_members = self.members.loc[self.members.Instrument.ne("C")]
        original_data = self.data.loc[
            self.data.index.get_level_values("Instrument") != "C"
        ]
        before = self.process(self.make_handler(original_members), original_data)
        after = self.process(self.make_handler(), self.data)
        past_keys = before.index[before.index.get_level_values("Date") < self.dates[-1]]

        pd.testing.assert_frame_equal(before.loc[past_keys], after.loc[past_keys])

    def test_pre_entry_rows_receive_the_historical_members_transform(self):
        result = self.process(self.make_handler(), self.data)
        self.assertEqual(result.index.tolist(), self.data.index.tolist())
        self.assertAlmostEqual(result.loc[(self.dates[0], "C"), "FACTOR"], 7.5)
        self.assertAlmostEqual(result.loc[(self.dates[0], "C"), "RET"], 0.075)

    def test_pre_entry_history_still_builds_a_complete_sequence(self):
        processed = self.process(self.make_handler(), self.data)
        with patch.object(DataLoader, "load_instruments", return_value=self.members):
            dataset = TSDataset(
                processed, segments={"train": (self.dates[0], self.dates[-1])},
                data_dir="unused", universe="INDEX", seq_len=3,
                feature_names=["FACTOR"], label_names=["RET"],
            )

        slices = dataset._daily_slices[dataset._daily_dates.index(self.dates[-1])]
        c_slice = next(s for s in slices if dataset._index[s.stop - 1][0] == "C")
        self.assertEqual(dataset._index[c_slice].tolist(), [("C", d) for d in self.dates])

    def test_day_without_members_keeps_history_without_using_future_members(self):
        members = self.members.loc[self.members.Date.ne(self.dates[0])]
        result = self.process(self.make_handler(members), self.data)
        pd.testing.assert_frame_equal(
            result.loc[self.dates[0]], self.data.loc[self.dates[0]]
        )

    def test_test_mode_still_leaves_raw_labels_untouched(self):
        result = self.process(self.make_handler(), self.data, mode="test")
        pd.testing.assert_series_equal(result["RET"], self.data["RET"])
        self.assertEqual(result.loc[(self.dates[0], "C"), "FACTOR"], 7.5)

    def test_static_instrument_lists_keep_the_existing_behavior(self):
        handler = self.make_handler(instruments=["A", "B", "C"])
        result = self.process(handler, self.data)
        self.assertEqual(result.loc[(self.dates[0], "C"), "FACTOR"], 505.0)

    def test_market_date_only_index_does_not_use_stock_membership(self):
        market = pd.DataFrame(
            {"FACTOR": [0.0, 10.0, 1000.0]},
            index=pd.Index([self.dates[0]] * 3, name="Date"),
        )
        result = self.make_handler().process(
            market, feature_names=["FACTOR"],
            processors=[CSWinsorize("feature", 0.25, 0.75)],
        )
        self.assertEqual(result["FACTOR"].tolist(), [5.0, 10.0, 505.0])

    def neutralization_data(self):
        cap = np.arange(12, dtype=float)
        frame = pd.DataFrame(
            {("feature", "CAP"): cap, ("feature", "FACTOR"): 2 * cap + 3},
            index=pd.MultiIndex.from_product(
                [[self.dates[0]], [f"A{i:02d}" for i in range(12)]],
                names=["Date", "Instrument"],
            ),
        )
        return frame

    def test_future_member_does_not_change_neutralization_coefficients(self):
        original = self.neutralization_data()
        full = pd.concat([
            original,
            pd.DataFrame(
                {("feature", "CAP"): [100.0], ("feature", "FACTOR"): [10000.0]},
                index=pd.MultiIndex.from_tuples(
                    [(self.dates[0], "FUTURE")], names=original.index.names
                ),
            ),
        ])
        processor = CSNeutralize("feature", cap_col="CAP", factor_cols=["FACTOR"])
        before = processor(original.copy())
        after = processor(
            full.copy(), reference_mask=pd.Series(True, index=original.index)
        )

        np.testing.assert_allclose(
            before[("feature", "FACTOR")],
            after.loc[original.index, ("feature", "FACTOR")], atol=1e-6,
        )
        self.assertAlmostEqual(
            after.loc[(self.dates[0], "FUTURE"), ("feature", "FACTOR")], 9797.0,
        )

    def test_neutralization_minimum_counts_only_historical_members(self):
        data = self.neutralization_data()
        processor = CSNeutralize("feature", cap_col="CAP", factor_cols=["FACTOR"])
        result = processor(
            data.copy(), reference_mask=pd.Series(True, index=data.index[:2])
        )
        pd.testing.assert_frame_equal(result, data)

    def test_handler_passes_membership_to_neutralization(self):
        handler = self.make_handler()
        data = self.neutralization_data()
        handler.instruments_index = data.index[:2]
        result = handler.process(
            data.droplevel(0, axis=1).copy(), feature_names=["CAP", "FACTOR"],
            processors=[CSNeutralize("feature", cap_col="CAP", factor_cols=["FACTOR"])],
        )
        pd.testing.assert_frame_equal(result, data.droplevel(0, axis=1))

    def test_all_cross_sectional_normalizers_use_historical_members(self):
        original_members = self.members.loc[self.members.Instrument.ne("C")]
        original_data = self.data.loc[
            self.data.index.get_level_values("Instrument") != "C"
        ]
        for processor in [
            CSZScoreNorm("feature"), CSZScoreNorm("feature", method="robust"),
            CSRankNorm("feature"),
        ]:
            with self.subTest(processor=type(processor).__name__):
                before = self.make_handler(original_members).process(
                    original_data.copy(), feature_names=["FACTOR"],
                    label_names=["RET"], processors=[processor],
                )
                after = self.make_handler().process(
                    self.data.copy(), feature_names=["FACTOR"],
                    label_names=["RET"], processors=[processor],
                )
                keys = before.index[before.index.get_level_values("Date") < self.dates[-1]]
                pd.testing.assert_frame_equal(before.loc[keys], after.loc[keys])

    def test_new_cross_sectional_processor_needs_no_handler_special_case(self):
        class CenterOnReference(CrossSectionalProcessor):
            def transform_group(self, df, reference):
                df[("feature", "FACTOR")] -= reference[("feature", "FACTOR")].mean()
                return df

        result = self.make_handler().process(
            self.data.copy(), feature_names=["FACTOR"], label_names=["RET"],
            processors=[CenterOnReference("feature")],
        )
        self.assertEqual(result.loc[(self.dates[0], "A"), "FACTOR"], -5.0)
        self.assertEqual(result.loc[(self.dates[0], "C"), "FACTOR"], 995.0)

    def test_global_normalizer_fits_only_training_period_historical_members(self):
        handler = self.make_handler()
        handler.fit_end_time = self.dates[1]
        processor = RobustZScoreNorm("feature", clip_outlier=False)
        result = handler.process(
            self.data.copy(), feature_names=["FACTOR"], label_names=["RET"],
            processors=[processor],
        )
        self.assertEqual(processor.mean_train.tolist(), [5.0])
        np.testing.assert_allclose(processor.std_train, [5.0 * 1.4826])
        self.assertEqual(len(result), len(self.data))

    def test_row_filters_keep_reference_alignment_for_later_processors(self):
        data = self.data.copy()
        data.loc[(self.dates[0], "A"), "RET"] = np.nan
        result = self.make_handler().process(
            data, feature_names=["FACTOR"], label_names=["RET"],
            processors=[Dropna("label"), CSZScoreNorm("feature")],
        )
        # B is the only remaining historical member on day one; C cannot
        # contribute to its center or variance despite still having a row.
        self.assertEqual(result.loc[(self.dates[0], "B"), "FACTOR"], 0.0)
        self.assertEqual(result.loc[(self.dates[0], "C"), "FACTOR"], 3.0)

    def test_row_wise_processor_still_transforms_non_member_history(self):
        data = self.data.copy()
        data.loc[(self.dates[0], "C"), "FACTOR"] = np.nan
        result = self.make_handler().process(
            data, feature_names=["FACTOR"], label_names=["RET"],
            processors=[Fillna("feature")],
        )
        self.assertEqual(result.loc[(self.dates[0], "C"), "FACTOR"], 0.0)
        self.assertEqual(len(result), len(self.data))

    def test_rank_normalization_preserves_legacy_ties_and_missing_values(self):
        data = pd.DataFrame(
            {("feature", "FACTOR"): [0.0, 0.0, 10.0, np.nan]},
            index=pd.MultiIndex.from_product(
                [[self.dates[0]], ["A", "B", "C", "D"]],
                names=["Date", "Instrument"],
            ),
        )
        ranks = data[("feature", "FACTOR")].rank(method="average")
        scaled = (ranks - 1) / (len(ranks) - 1)
        expected = (scaled - scaled.mean()) / scaled.std(ddof=0)
        result = CSRankNorm("feature")(data.copy())
        pd.testing.assert_series_equal(result[("feature", "FACTOR")], expected)

    def test_rank_outside_reference_uses_its_empirical_distribution(self):
        data = pd.DataFrame(
            {("feature", "FACTOR"): [0.0, 10.0, 5.0]},
            index=pd.MultiIndex.from_product(
                [[self.dates[0]], ["A", "B", "C"]],
                names=["Date", "Instrument"],
            ),
        )
        result = CSRankNorm("feature")(
            data, reference_mask=pd.Series(True, index=data.index[:2])
        )
        np.testing.assert_allclose(result[("feature", "FACTOR")], [-1.0, 1.0, 0.0])

    def test_fit_transform_and_existing_callable_api_share_the_same_transform(self):
        data = self.data.loc[[self.dates[0]]].copy()
        data.columns = pd.MultiIndex.from_tuples([("feature", "FACTOR"), ("label", "RET")])
        mask = pd.Series(True, index=data.index[:2])
        processor = RobustZScoreNorm("feature", clip_outlier=False)
        result = processor.fit_transform(data.copy(), reference_mask=mask)
        self.assertEqual(processor.mean_train.tolist(), [5.0])
        pd.testing.assert_frame_equal(result, processor(data.copy()))

    def test_new_industry_outside_reference_cannot_change_existing_residuals(self):
        original = self.neutralization_data()
        original[("feature", "IND")] = [1, 2] * 6
        original[("feature", "FACTOR")] += np.arange(12) % 3
        future = pd.DataFrame(
            {("feature", "CAP"): [100.0], ("feature", "FACTOR"): [10000.0],
             ("feature", "IND"): [0]},
            index=pd.MultiIndex.from_tuples(
                [(self.dates[0], "FUTURE")], names=original.index.names
            ),
        )
        processor = CSNeutralize(
            "feature", industry_col="IND", cap_col="CAP", factor_cols=["FACTOR"]
        )
        before = processor(original.copy())
        after = processor(
            pd.concat([original, future]),
            reference_mask=pd.Series(True, index=original.index),
        )
        np.testing.assert_allclose(
            before[("feature", "FACTOR")],
            after.loc[original.index, ("feature", "FACTOR")], atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
