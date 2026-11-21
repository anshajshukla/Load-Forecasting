"""Paper evaluation protocol: 2026 never enters, and the significance test behaves."""
import numpy as np
import pandas as pd
import pytest

from research.common import DEV_END, load_dev, save
from research.evaluate import dm_test


def test_dev_data_stops_before_2026():
    assert load_dev().index.max() < pd.Timestamp(DEV_END)


def test_save_refuses_2026(tmp_path, monkeypatch):
    import research.common as c
    monkeypatch.setattr(c, "OUT", tmp_path)
    with pytest.raises(AssertionError):
        save("x", pd.Series([1.0], index=pd.to_datetime(["2026-01-01"])))


def test_dm_sign_and_power():
    rng = np.random.default_rng(0)
    a = np.abs(rng.normal(0, 1, 2000))
    stat, p = dm_test(a, a + 0.2)          # a has lower loss
    assert stat < 0 and p < 1e-6
    _, p_same = dm_test(a, rng.permutation(a))
    assert p_same > 0.01
