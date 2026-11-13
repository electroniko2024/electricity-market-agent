"""Market tool contracts preserve source keys and fail closed on unresolved evidence."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import duckdb
from pydantic import ValidationError

from ellectric.service import market_tools


def _make_database(path: Path) -> None:
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE market_15min (
            trade_date DATE,
            time_label VARCHAR,
            slot_id SMALLINT,
            is_holiday BOOLEAN,
            is_weekend BOOLEAN,
            rt_price DOUBLE,
            load_actual_mw DOUBLE,
            load_forecast_mw DOUBLE,
            tie_line_actual_mw DOUBLE,
            tie_line_forecast_mw DOUBLE,
            wind_actual_mw DOUBLE,
            wind_forecast_mw DOUBLE,
            solar_actual_mw DOUBLE,
            solar_forecast_mw DOUBLE,
            nuclear_actual_mw DOUBLE,
            nuclear_forecast_mw DOUBLE,
            captive_actual_mw DOUBLE,
            captive_forecast_mw DOUBLE,
            local_gen_actual_mw DOUBLE,
            local_gen_forecast_mw DOUBLE,
            solar_actual_clipped_mw DOUBLE,
            solar_forecast_clipped_mw DOUBLE,
            actual_bid_space_mw DOUBLE,
            forecast_bid_space_mw DOUBLE
        )
        """
    )
    rows = [
        (date(2025, 10, 1), "00:15", 0, False, False, -10.0, 1100, 1000, 100, 90, 200, 180, 50, 40, 10, 10, 20, 20, 30, 30, 50, 40, 710, 630),
        (date(2025, 10, 1), "00:30", 1, False, False, -20.0, 1110, 1000, 100, 90, 200, 180, 50, 40, 10, 10, 20, 20, 30, 30, 50, 40, 720, 630),
        (date(2025, 10, 1), "00:45", 2, False, False, None, 1120, 1000, 100, 90, 200, 180, 50, 40, 10, 10, 20, 20, 30, 30, 50, 40, 730, 630),
        (date(2025, 10, 1), "01:00", 3, False, False, 900.0, 1130, 1000, 100, 90, 200, 180, 50, 40, 10, 10, 20, 20, 30, 30, 50, 40, 740, 630),
        (date(2025, 11, 1), "00:15", 0, False, False, 50.0, 1100, 1000, 100, 90, 200, 180, 50, 40, 10, 10, 20, 20, 30, 30, 50, 40, 710, 630),
    ]
    placeholders = ",".join("?" for _ in rows[0])
    connection.executemany(f"INSERT INTO market_15min VALUES ({placeholders})", rows)
    connection.execute(
        """
        CREATE TABLE weather_actual_hourly (
            timestamp_raw TIMESTAMPTZ,
            temp_jinan DOUBLE,
            ghi_jinan DOUBLE,
            wind_speed_jinan DOUBLE,
            precip_jinan DOUBLE,
            humidity_jinan BIGINT,
            cloud_jinan BIGINT
        )
        """
    )
    connection.execute(
        "INSERT INTO weather_actual_hourly VALUES ('2025-10-01 00:00:00+00', 18, 100, 2, 0, 50, 30)"
    )
    connection.close()


class MarketToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "market.duckdb"
        _make_database(self.db_path)
        self.database_patch = patch.object(market_tools, "DATABASE_PATH", self.db_path)
        self.database_patch.start()

    def tearDown(self) -> None:
        self.database_patch.stop()
        self.temporary.cleanup()

    def test_data_catalog_reads_coverage_from_the_database_and_stops_at_as_of(self) -> None:
        from datetime import datetime

        market_tools._catalog_coverage.cache_clear()
        catalog = market_tools.data_catalog(datetime.fromisoformat("2025-10-20T09:00:00+08:00"))
        market = catalog["tables"]["market_15min"]
        self.assertEqual(market["first_date"], "2025-10-01")
        self.assertEqual(market["last_available_date"], "2025-10-19")  # the database goes on to 2025-11-01
        self.assertIn("rt_price", market["columns"])
        self.assertNotIn("slot_id", market["columns"])
        self.assertEqual(catalog["forecast_target_date"], "2025-10-20")
        self.assertIn("没有其他省份", catalog["region"])
        market_tools._catalog_coverage.cache_clear()

    def test_time_label_selects_the_slot(self) -> None:
        period = {"start_date": "2025-10-01", "end_date": "2025-10-01"}
        self.assertEqual(market_tools.PriceQueryInput(**period, time_label="19:45").slot_id, 78)
        self.assertEqual(market_tools.PriceQueryInput(**period, time_label="24:00").slot_id, 95)
        rows = market_tools.query_price(market_tools.PriceQueryInput(**period, time_label="00:30"))["data"]["rows"]
        self.assertEqual([(row["slot_id"], row["time_label"]) for row in rows], [(1, "00:30")])
        for bad in ("19:40", "00:00", "7点"):
            with self.assertRaises(ValidationError):
                market_tools.PriceQueryInput(**period, time_label=bad)
        with self.assertRaises(ValidationError):
            market_tools.PriceQueryInput(**period, time_label="19:45", slot_id=79)

    def test_price_load_weather_and_stats_keep_quality_and_timezone_limits(self) -> None:
        period = {"start_date": "2025-10-01", "end_date": "2025-10-01"}
        prices = market_tools.query_price(market_tools.PriceQueryInput(**period, slot_id=0))
        stats = market_tools.query_stats(
            market_tools.StatsInput(**period, metric="rt_price", group_by="day")
        )
        load = market_tools.query_load(market_tools.PriceQueryInput(**period, limit=2))
        weather = market_tools.query_weather(market_tools.WeatherQueryInput(**period))

        self.assertEqual(prices["data"]["rows"][0]["time_label"], "00:15")
        self.assertTrue(prices["data"]["rows"][0]["target_observed"])
        self.assertEqual(stats["data"]["missing_prices"], 1)
        self.assertEqual(stats["data"]["negative_price_slots"], 2)
        self.assertEqual(stats["data"]["groups"][0]["group_value"], "2025-10-01")
        self.assertEqual(stats["data"]["groups"][0]["negative_price_share"], 2 / 3)
        self.assertIsNone(stats["data"]["verified_floor_price_count"])
        self.assertTrue(load["data"]["truncated"])
        self.assertIn("timestamp basis", weather["limitations"][0])
        self.assertEqual(weather["data"]["rows"][0]["temp_jinan"], 18.0)

    def test_stats_support_day_month_and_slot_groupings(self) -> None:
        period = {"start_date": "2025-10-01", "end_date": "2025-11-01"}

        monthly = market_tools.query_stats(
            market_tools.StatsInput(**period, metric="rt_price", group_by="month")
        )["data"]["groups"]
        by_slot = market_tools.query_stats(
            market_tools.StatsInput(**period, metric="rt_price", group_by="slot")
        )["data"]["groups"]

        self.assertEqual([group["group_value"] for group in monthly], ["2025-10", "2025-11"])
        self.assertEqual(monthly[0]["negative_price_share"], 2 / 3)
        self.assertEqual(monthly[1]["negative_price_share"], 0.0)
        self.assertTrue(all(group["verified_floor_price_count"] is None for group in monthly))
        self.assertEqual([group["group_value"] for group in by_slot], [0, 1, 2, 3])
        self.assertEqual(by_slot[0]["negative_price_share"], 0.5)
        self.assertIsNone(by_slot[2]["negative_price_share"])
        self.assertTrue(all(group["floor_price_count_status"] == "unavailable_unverified_policy_limits" for group in by_slot))
        self.assertEqual([group["time_label"] for group in by_slot], ["00:15", "00:30", "00:45", "01:00"])
        # Month groups say which of their days fall inside the requested range.
        self.assertEqual([(g["first_date"], g["last_date"]) for g in monthly],
                         [("2025-10-01", "2025-10-31"), ("2025-11-01", "2025-11-01")])

    def test_group_extremes_rank_groups_with_ties(self) -> None:
        stats = market_tools.query_stats(
            market_tools.StatsInput(start_date="2025-10-01", end_date="2025-11-01", group_by="month")
        )["data"]
        share = stats["group_extremes"]["negative_price_slots"]
        self.assertEqual((share["max"]["group_value"], share["max"]["first_date"], share["max"]["tied_groups"]),
                         ("2025-10", "2025-10-01", 1))
        self.assertEqual(share["max"]["value"], max(g["negative_price_slots"] for g in stats["groups"]))
        groups = [{"group_value": "a", "mean_value": 2.0}, {"group_value": "b", "mean_value": 2.0},
                  {"group_value": "c", "mean_value": None}]
        ranked = market_tools._group_extremes(groups)["mean_value"]
        self.assertEqual((ranked["max"]["group_value"], ranked["max"]["tied_groups"]), ("a", 2))

    def test_month_bounds_expand_to_the_whole_month(self) -> None:
        # A month group value ("2025-10") can be passed straight back as a range bound.
        request = market_tools.StatsInput(start_date="2025-10", end_date="2025-10")
        self.assertEqual((request.start_date.isoformat(), request.end_date.isoformat()), ("2025-10-01", "2025-10-31"))
        request = market_tools.StatsInput(start_date="2024-02", end_date="2024-02")
        self.assertEqual(request.end_date.isoformat(), "2024-02-29")
        # A month still running at as_of ends at the last available trade date instead of failing.
        running = market_tools.StatsInput(start_date="2025-03", end_date="2025-03", as_of="2025-03-17T09:00:00+08:00")
        self.assertEqual(running.end_date.isoformat(), "2025-03-16")
        with self.assertRaises(ValidationError):
            market_tools.StatsInput(start_date="2025-1", end_date="2025-10-02")

    def test_stats_cover_long_ranges_and_locate_extremes(self) -> None:
        # Aggregates are not capped at 32 days (detail tools are), and each extreme names its source row.
        stats = market_tools.query_stats(
            market_tools.StatsInput(start_date="2025-09-01", end_date="2025-11-30", group_by="month")
        )["data"]
        self.assertEqual(stats["max_at"], {"trade_date": "2025-10-01", "slot_id": 3, "time_label": "01:00", "rt_price": 900.0})
        self.assertEqual(stats["min_at"]["slot_id"], 1)
        self.assertEqual(stats["groups"][0]["max_at"]["rt_price"], 900.0)
        self.assertIn("spike_price_slots", stats["groups"][0])
        load = market_tools.query_stats(
            market_tools.StatsInput(start_date="2025-10-01", end_date="2025-10-01", metric="load_actual_mw")
        )["data"]
        self.assertEqual((load["max_value"], load["unit"]), (1130, "MW"))
        self.assertEqual(load["groups"][0]["max_at"], {"trade_date": "2025-10-01", "slot_id": 3, "time_label": "01:00", "load_actual_mw": 1130})
        weather = market_tools.query_stats(
            market_tools.StatsInput(start_date="2025-10-01", end_date="2025-10-01", metric="temp_jinan")
        )["data"]
        self.assertEqual(weather["max_at"]["temp_jinan"], 18.0)
        self.assertIn("timestamp_raw", weather["max_at"])

    def test_span_limits_point_to_the_aggregate_tool(self) -> None:
        with self.assertRaisesRegex(ValidationError, "query_stats"):
            market_tools.PriceQueryInput(start_date="2025-01-01", end_date="2025-03-01")
        with self.assertRaises(ValidationError):
            market_tools.StatsInput(start_date="2024-01-01", end_date="2025-06-01")
        with self.assertRaises(ValidationError):
            market_tools.StatsInput(start_date="2025-10-01", end_date="2025-10-01", metric="temp_jinan", group_by="slot")

    def test_anomaly_events_do_not_bridge_missing_price_slots(self) -> None:
        result = market_tools.detect_anomaly(
            market_tools.AnomalyInput(
                start_date="2025-10-01", end_date="2025-10-01"
            )
        )
        events = result["data"]["events"]
        negative_runs = [event for event in events if event["event_type"] == "negative_price_run"]
        isolated = [event for event in events if event["event_type"] == "isolated_negative_price"]
        spikes = [event for event in events if event["event_type"] == "high_price_candidate"]
        quality = [event for event in events if event["event_type"].startswith("data_quality_")]

        self.assertEqual(len(negative_runs), 1)
        self.assertEqual(negative_runs[0]["points"], 2)
        self.assertEqual(negative_runs[0]["start"]["slot_id"], 0)
        self.assertTrue(negative_runs[0]["event_id"].startswith("evt_"))
        self.assertEqual(len(isolated), 0)
        self.assertEqual(len(spikes), 1)
        self.assertEqual(spikes[0]["threshold_status"], "training_p99_candidate_not_legal_price_limit")
        self.assertTrue(any("rt_price" in event.get("missing_fields", []) for event in quality))
        self.assertTrue(any(event["event_type"] == "data_quality_missing_slots" for event in quality))
        self.assertIn("FLOOR_PRICE_NOT_DETECTED", " ".join(result["limitations"]))

    def test_supply_decomposition_and_policy_search_fail_closed(self) -> None:
        anomaly = market_tools.detect_anomaly(
            market_tools.AnomalyInput(
                start_date="2025-10-01",
                end_date="2025-10-01",
                methods=["negative_price"],
            )
        )
        event_id = anomaly["data"]["events"][0]["event_id"]
        with patch.object(market_tools, "FIXED_SNAPSHOT_DIR", Path(self.temporary.name) / "no-model"), patch.object(
            market_tools, "POLICY_METADATA_PATH", Path(self.temporary.name) / "no-metadata.json"
        ):
            drivers = market_tools.explain_drivers(
                market_tools.DriverInput(
                    start_date="2025-10-01",
                    end_date="2025-10-01",
                    event_ids=[event_id],
                )
            )
        with patch.object(market_tools, "POLICY_METADATA_PATH", Path(self.temporary.name) / "no-metadata.json"):
            policy = market_tools.search_policy(
                market_tools.PolicySearchInput(query="价格上限", policy_at="2025-10-01")
            )

        event = drivers["data"]["events"][0]
        point = event["representative_market_point"]
        self.assertTrue(point["supply_error_complete"])
        self.assertEqual(
            point["reconstructed_actual_minus_forecast_bid_space_mw"], 60.0
        )
        self.assertEqual(point["calendar"], {"is_holiday": False, "is_weekend": False})
        self.assertEqual(event["weather_context"]["time_alignment"], "not_performed")
        self.assertEqual(event["policy_evidence"]["status"], "unverified_candidates")
        self.assertEqual(policy["status"], "unverified_candidates")
        self.assertIn("POLICY_METADATA_PENDING_USER_CONFIRMATION", policy["limitations"][0])
        self.assertTrue(policy["data"]["evidence"])
        self.assertFalse(policy["data"]["evidence"][0]["evidence_eligible"])

    def _calibration_files(self) -> tuple[Path, Path]:
        import json

        import pandas as pd

        calibration = Path(self.temporary.name) / "calibration.json"
        calibration.write_text(
            json.dumps(
                {
                    "calibration_period": ["2025-07-01", "2025-09-30"],
                    "divergence": {
                        "high_threshold_yuan_per_mwh": 50.0,
                        "validation_ensemble_mae_high_divergence": 100.0,
                        "validation_ensemble_mae_other": 50.0,
                    },
                    "forecast_residual": {"threshold_yuan_per_mwh": 300.0},
                }
            ),
            encoding="utf-8",
        )
        forecasts = Path(self.temporary.name) / "forecasts.parquet"
        pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2025-10-01"] * 4),
                "time_label": ["00:15", "00:30", "00:45", "01:00"],
                "slot_id": [0, 1, 2, 3],
                "LightGBM": [400.0, 300.0, 0.0, 100.0],
                "XGBoost": [400.0, 300.0, 0.0, 100.0],
                "PatchTST": [400.0, 300.0, 0.0, 100.0],
            }
        ).to_parquet(forecasts)
        return calibration, forecasts

    def test_forecast_residual_events_use_saved_forecasts_and_calibrated_threshold(self) -> None:
        calibration, forecasts = self._calibration_files()
        with patch.object(market_tools, "CALIBRATION_PATH", calibration), patch.object(
            market_tools, "FIXED_TEST_PREDICTIONS_PATH", forecasts
        ):
            result = market_tools.detect_anomaly(
                market_tools.AnomalyInput(
                    start_date="2025-10-01", end_date="2025-10-01", methods=["forecast_residual"]
                )
            )
        events = result["data"]["events"]
        # Slots 0-1: actual -10/-20 vs forecast 400/300 -> one low run; slot 2 has no price;
        # slot 3: 900 vs 100 -> high.
        self.assertEqual([(e["event_type"], e["points"]) for e in events],
                         [("forecast_residual_low", 2), ("forecast_residual_high", 1)])
        self.assertEqual(events[0]["peak_residual_yuan_per_mwh"], -410.0)
        self.assertEqual(events[1]["threshold_yuan_per_mwh"], 300.0)

    def test_forecast_residual_outside_saved_forecasts_fails_closed(self) -> None:
        calibration, forecasts = self._calibration_files()
        with patch.object(market_tools, "CALIBRATION_PATH", calibration), patch.object(
            market_tools, "FIXED_TEST_PREDICTIONS_PATH", forecasts
        ):
            result = market_tools.detect_anomaly(
                market_tools.AnomalyInput(
                    start_date="2025-09-01", end_date="2025-09-02", methods=["forecast_residual"]
                )
            )
        self.assertEqual(result["data"]["event_count"], 0)
        self.assertIn("FORECAST_RESIDUAL_NOT_DETECTED", " ".join(result["limitations"]))

    def test_floor_price_events_need_confirmed_limits(self) -> None:
        import json

        limits = Path(self.temporary.name) / "limits.json"
        period = {"from": "2025-10-01", "to": "2025-10-02", "rt_clearing_floor_yuan_per_mwh": -20.0, "source": "test"}
        for status, expected in (("pending_user_confirmation", 0), ("user_confirmed", 1)):
            limits.write_text(json.dumps({"verification_status": status, "periods": [period]}), encoding="utf-8")
            market_tools._cached_json.cache_clear()
            with patch.object(market_tools, "PRICE_LIMITS_PATH", limits):
                result = market_tools.detect_anomaly(market_tools.AnomalyInput(
                    start_date="2025-10-01", end_date="2025-10-01", methods=["floor_price"]))
            floor = [e for e in result["data"]["events"] if e["event_type"] == "floor_price"]
            self.assertEqual(len(floor), expected)
        self.assertEqual(floor[0]["start"]["slot_id"], 1)  # the -20 slot
        self.assertTrue(floor[0]["event_id"].startswith("evt_"))

    def test_divergence_summary_flags_slots_above_threshold(self) -> None:
        rows = [
            {"slot_id": 0, "time_label": "00:15", "model_std": 10.0, "high_divergence": False},
            {"slot_id": 1, "time_label": "00:30", "model_std": 60.0, "high_divergence": True},
        ]
        calibration = {
            "calibration_period": ["2025-07-01", "2025-09-30"],
            "divergence": {
                "high_threshold_yuan_per_mwh": 50.0,
                "validation_ensemble_mae_high_divergence": 100.0,
                "validation_ensemble_mae_other": 50.0,
            },
        }
        summary = market_tools._divergence_summary(rows, calibration)
        self.assertEqual(summary["high_divergence_slot_count"], 1)
        self.assertEqual(summary["max_model_std_slot"]["slot_id"], 1)
        self.assertEqual(market_tools._divergence_summary(rows, None)["status"], "uncalibrated")

    def test_as_of_cutoff_rejects_unfinished_trade_dates(self) -> None:
        # 2025-10-02 00:30 Beijing: 2025-10-01 is complete, 2025-10-02 is not.
        as_of = "2025-10-01T16:30:00+00:00"
        self.assertEqual(
            market_tools.latest_available_trade_date(
                market_tools.PriceQueryInput(
                    start_date="2025-10-01", end_date="2025-10-01", as_of=as_of
                ).as_of
            ),
            date(2025, 10, 1),
        )
        with self.assertRaisesRegex(ValidationError, "DATA_AFTER_AS_OF"):
            market_tools.PriceQueryInput(start_date="2025-10-01", end_date="2025-10-02", as_of=as_of)
        with self.assertRaisesRegex(ValidationError, "POLICY_AT_AFTER_AS_OF"):
            market_tools.PolicySearchInput(query="价格上限", policy_at="2025-10-03", as_of=as_of)
        market_tools.PolicySearchInput(query="价格上限", policy_at="2025-10-02", as_of=as_of)

    def test_tool_inputs_reject_extra_fields_and_invalid_slots(self) -> None:
        with self.assertRaises(ValidationError):
            market_tools.PriceQueryInput(
                start_date="2025-10-01",
                end_date="2025-10-01",
                slot_id=96,
            )
        with self.assertRaises(ValidationError):
            market_tools.DateRangeInput(
                start_date="2025-10-02",
                end_date="2025-10-01",
                hidden="ignored",
            )
        with self.assertRaises(ValidationError):
            market_tools.StatsInput(
                start_date="2025-10-01",
                end_date="2025-10-01",
                group_by="quarter",
            )
        with self.assertRaises(ValidationError):
            market_tools.StatsInput(
                start_date="2025-10-01",
                end_date="2025-10-01",
                metric="da_price",
            )


if __name__ == "__main__":
    unittest.main()
