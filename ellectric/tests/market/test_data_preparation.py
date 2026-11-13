from __future__ import annotations

import hashlib
import json

import duckdb
import pandas as pd
import pytest

from ellectric.pipeline.market.data import (
    _normalise_jinan_weather,
    _normalise_market,
    build_market_dataset,
)


def _power_rows() -> pd.DataFrame:
    rows = []
    for day in ("2025-12-31", "2026-01-01"):
        for slot in range(96):
            minutes = (slot + 1) * 15
            label = "24:00" if minutes == 1440 else f"{minutes // 60:02d}:{minutes % 60:02d}"
            rows.append(
                {
                    "日期": day,
                    "时刻": label,
                    "是否节假日": "是" if day == "2026-01-01" else "否",
                    "是否周末休息日": "否",
                    "直调负荷(预测)": 1000,
                    "联络线受电负荷(预测)": 100,
                    "风电总加(预测)": 200,
                    "光伏总加(预测)": -5 if slot == 0 else 50,
                    "非市场化核电总加(预测)": None if day == "2026-01-01" and slot == 1 else 100,
                    "自备机组总加(预测)": 50,
                    "地方电厂发电总加(预测)": 50,
                    "直调负荷(实际)": 1100,
                    "联络线受电负荷(实际)": 100,
                    "风电总加(实际)": 200,
                    "光伏总加(实际)": -30 if slot == 0 else 60,
                    "抽蓄(实际)": 10,
                    "地方电厂发电总加(实际)": 40,
                    "非市场化核电总加(实际)": 100,
                    "自备机组总加(实际)": 50,
                    "实时价格": None if day == "2025-12-31" and slot in (30, 31, 35) else (-20 if slot == 0 else 300),
                    "日前价格": "must-not-be-read",
                }
            )
    return pd.DataFrame(rows)


def _weather_frame() -> pd.DataFrame:
    index = pd.date_range("2025-12-31", periods=3, freq="h", tz="UTC", name="timestamp")
    values = {name: [1.0, None, 3.0] for name in (
        "temp_jinan", "ghi_jinan", "wind_speed_jinan", "precip_jinan", "humidity_jinan", "cloud_jinan"
    )}
    values.update({name: [99.0] * 3 for name in (
        "temp_qingdao", "ghi_qingdao", "wind_speed_qingdao", "precip_qingdao", "humidity_qingdao", "cloud_qingdao"
    )})
    return pd.DataFrame(values, index=index)


def test_market_normalisation_uses_trade_date_slot_and_keeps_label_semantics_unknown():
    market, quality = _normalise_market(_power_rows())

    first = market.iloc[0]
    last = market.iloc[95]
    assert first["trade_date"].isoformat() == "2025-12-31"
    assert first["slot_id"] == 0
    assert first["time_label"] == "00:15"
    assert str(first["label_at_local"]) == "2025-12-31 00:15:00+08:00"
    assert last["slot_id"] == 95
    assert last["time_label"] == "24:00"
    assert str(last["label_at_local"]) == "2026-01-01 00:00:00+08:00"
    assert quality["time_semantics"]["interval_boundary"] == "unknown"
    assert not {"ts_start", "ts_end"} & set(market.columns)
    assert "日前价格" not in market.columns


def test_missing_targets_stay_null_and_incomplete_bid_space_is_not_partially_summed():
    market, quality = _normalise_market(_power_rows())
    target = market.loc[
        (market["trade_date"] == pd.Timestamp("2026-01-01").date())
        & (market["slot_id"] == 1)
    ].iloc[0]
    assert pd.isna(target["forecast_bid_space_mw"])
    assert quality["forecast_bid_space_missing_count"] == 1
    assert quality["price_missing_segments"]
    assert quality["negative_rt_price_count"] == 2


def test_solar_raw_value_is_kept_and_clipped_value_is_separate():
    market, _ = _normalise_market(_power_rows())
    first = market.iloc[0]
    assert first["solar_actual_mw"] == -30
    assert first["solar_actual_clipped_mw"] == 0
    assert first["actual_bid_space_mw"] == 610


def test_weather_is_jinan_only_and_timestamp_values_are_not_shifted_or_resampled(tmp_path):
    path = tmp_path / "weather.parquet"
    _weather_frame().to_parquet(path)
    weather, quality = _normalise_jinan_weather(path)

    assert len(weather) == 3
    assert "temp_qingdao" not in weather.columns
    assert weather["timestamp_raw"].iloc[0] == pd.Timestamp("2025-12-31", tz="UTC")
    assert quality["timestamp_timezone_basis"] == "unknown"
    assert quality["units_basis"] == "unknown"
    assert quality["resampled_or_shifted"] is False


def test_build_writes_expected_tables_and_refuses_unrequested_overwrite(tmp_path):
    power = tmp_path / "power.csv"
    weather = tmp_path / "weather.parquet"
    output = tmp_path / "processed"
    source = _power_rows()
    source.to_csv(power, index=False, encoding="utf-8-sig")
    _weather_frame().to_parquet(weather)
    original_hash = hashlib.sha256(power.read_bytes()).hexdigest()

    result = build_market_dataset(power, weather, output)

    assert result["market_rows"] == 192
    assert hashlib.sha256(power.read_bytes()).hexdigest() == original_hash
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scope"]["day_ahead_price_processed"] is False
    assert manifest["market_time"]["interval_boundary"] == "unknown"
    with duckdb.connect(str(output / "sd_market.duckdb"), read_only=True) as connection:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        assert tables == {"market_15min", "weather_actual_hourly", "data_quality"}
        columns = {row[1] for row in connection.execute("PRAGMA table_info('market_15min')").fetchall()}
        assert "日前价格" not in columns
        assert connection.execute("SELECT count(*) FROM weather_actual_hourly").fetchone()[0] == 3

    with pytest.raises(FileExistsError):
        build_market_dataset(power, weather, output)
