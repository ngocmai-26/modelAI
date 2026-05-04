"""Tests for ml_clo.features.categorical_encoder (Phase 3)."""

import numpy as np
import pandas as pd
import pytest

from ml_clo.features.categorical_encoder import FrequencyEncoder, TargetEncoder


class TestFrequencyEncoder:
    def test_fit_transform_basic(self):
        x = pd.Series(["A", "B", "A", "C", "A"])
        enc = FrequencyEncoder()
        result = enc.fit_transform(x)
        assert result.tolist() == [3, 1, 3, 1, 3]

    def test_unseen_category_uses_default(self):
        enc = FrequencyEncoder().fit(pd.Series(["A", "A", "B"]))
        result = enc.transform(pd.Series(["A", "ZZ"]))
        assert result.iloc[0] == 2
        assert result.iloc[1] == 0  # default for unseen

    def test_missing_values_grouped(self):
        x = pd.Series(["A", None, "A", np.nan])
        enc = FrequencyEncoder()
        result = enc.fit_transform(x)
        # Both None and NaN map to "__MISSING__"
        assert result.iloc[1] == result.iloc[3]

    def test_state_picklable_via_dict(self):
        enc = FrequencyEncoder().fit(pd.Series(["X", "Y", "X"]))
        # Encoder state lives in plain dict — joblib-friendly
        assert isinstance(enc.counts, dict)
        assert "X" in enc.counts
        assert enc.counts["X"] == 2


class TestTargetEncoder:
    def test_constant_target_yields_global_mean(self):
        x = pd.Series(["A"] * 10 + ["B"] * 10)
        y = pd.Series([3.0] * 20)
        enc = TargetEncoder(n_folds=5, smoothing=10.0)
        result = enc.fit_transform(x, y)
        assert all(abs(v - 3.0) < 1e-6 for v in result)
        assert enc.global_mean == pytest.approx(3.0)

    def test_strong_signal_separated(self):
        # A always low, B always high — encoder should separate them
        x = pd.Series(["A"] * 50 + ["B"] * 50)
        y = pd.Series([1.0] * 50 + [5.0] * 50)
        enc = TargetEncoder(n_folds=5, smoothing=1.0)
        encoded = enc.fit_transform(x, y)
        a_mean = encoded.iloc[:50].mean()
        b_mean = encoded.iloc[50:].mean()
        assert a_mean < b_mean
        assert b_mean - a_mean > 1.0  # clear separation

    def test_smoothing_shrinks_rare_categories(self):
        # Rare category C with extreme target — should be shrunk toward global mean
        x = pd.Series(["A"] * 20 + ["B"] * 20 + ["C"])
        y = pd.Series([3.0] * 20 + [3.0] * 20 + [10.0])  # C is outlier
        enc_small_smooth = TargetEncoder(n_folds=5, smoothing=0.1)
        enc_big_smooth = TargetEncoder(n_folds=5, smoothing=100.0)
        enc_small_smooth.fit(x, y)
        enc_big_smooth.fit(x, y)
        # With high smoothing, C's encoded value pulled toward global ~3.17
        c_small = enc_small_smooth.means["C"]
        c_big = enc_big_smooth.means["C"]
        assert abs(c_big - enc_big_smooth.global_mean) < abs(c_small - enc_small_smooth.global_mean)

    def test_unseen_category_uses_global_mean(self):
        enc = TargetEncoder(n_folds=5, smoothing=10.0)
        enc.fit(pd.Series(["A", "B"] * 50), pd.Series(np.linspace(1, 5, 100)))
        result = enc.transform(pd.Series(["UNSEEN"]))
        assert result.iloc[0] == pytest.approx(enc.global_mean)

    def test_kfold_no_leakage_for_singleton(self):
        # If a category has only 1 sample, out-of-fold encoding cannot use that
        # sample (it would be in the val fold) — encoded value falls back to global.
        x = pd.Series(["X"] + ["A"] * 100)
        y = pd.Series([99.0] + [1.0] * 100)
        enc = TargetEncoder(n_folds=5, smoothing=0.0)
        encoded = enc.fit_transform(x, y)
        # X's out-of-fold encoded value is NOT 99 (would be leakage)
        # It should fall back to global mean since X never appears in train fold
        assert encoded.iloc[0] != 99.0

    def test_target_validates_length(self):
        with pytest.raises(ValueError):
            TargetEncoder().fit(pd.Series(["A"]), pd.Series([1.0, 2.0]))

    def test_target_requires_y_for_fit_transform(self):
        with pytest.raises(ValueError):
            TargetEncoder().fit_transform(pd.Series(["A", "B"]), pd.Series([1.0]))

    def test_state_picklable(self):
        enc = TargetEncoder()
        enc.fit(pd.Series(["A", "B", "A"]), pd.Series([1.0, 2.0, 3.0]))
        assert isinstance(enc.means, dict)
        assert isinstance(enc.global_mean, float)
