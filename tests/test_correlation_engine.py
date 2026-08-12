"""Alert correlation: grouping, root-cause matching, and what happens to
alerts that match nothing."""

from __future__ import annotations

import pandas as pd
import pytest

from correlation_engine import (
    DEFAULT_RULES,
    AlertCorrelationEngine,
    CorrelationRule,
    summarise,
)

BASE = pd.Timestamp("2026-01-01 00:00:00")


def alerts(*rows) -> pd.DataFrame:
    """rows are (offset_seconds, message, severity)."""
    return pd.DataFrame(
        {
            "timestamp": [BASE + pd.Timedelta(seconds=r[0]) for r in rows],
            "message": [r[1] for r in rows],
            "severity": [r[2] if len(r) > 2 else "medium" for r in rows],
        }
    )


class TestGrouping:
    def test_a_burst_becomes_one_incident(self):
        frame = alerts(
            (0, "interface_down on Gi0/1"),
            (30, "bgp_session_down peer 10.0.0.1"),
            (60, "tunnel flap"),
        )
        assert len(AlertCorrelationEngine().correlate_alerts(frame)) == 1

    def test_alerts_separated_by_a_long_gap_are_separate_incidents(self):
        frame = alerts((0, "interface_down"), (3600, "interface_down"))
        assert len(AlertCorrelationEngine().correlate_alerts(frame)) == 2

    def test_a_long_storm_stays_one_incident(self):
        """Clustering on the gap between alerts, not on distance from the first
        one, is what keeps a ten-minute storm from being split into chunks."""
        frame = alerts(*[(i * 60, "interface_down") for i in range(11)])
        assert len(AlertCorrelationEngine(time_window=120).correlate_alerts(frame)) == 1

    def test_out_of_order_input_is_sorted_first(self):
        frame = alerts((120, "b"), (0, "a"), (60, "c"))
        incidents = AlertCorrelationEngine().correlate_alerts(frame)
        assert incidents.iloc[0]["first_seen"] == BASE

    def test_the_window_is_configurable(self):
        frame = alerts((0, "x"), (400, "y"))
        assert len(AlertCorrelationEngine(time_window=300).correlate_alerts(frame)) == 2
        assert len(AlertCorrelationEngine(time_window=600).correlate_alerts(frame)) == 1

    def test_string_timestamps_are_accepted(self):
        frame = pd.DataFrame(
            {"timestamp": ["2026-01-01T00:00:00", "2026-01-01T00:00:10"], "message": ["a", "b"]}
        )
        assert len(AlertCorrelationEngine().correlate_alerts(frame)) == 1


class TestRootCause:
    def test_interface_plus_bgp_is_a_path_failure(self):
        frame = alerts((0, "interface_down Gi0/1"), (10, "bgp_session_down peer 10.0.0.1"))
        incident = AlertCorrelationEngine().correlate_alerts(frame).iloc[0]
        assert incident["root_cause"] == "SDWAN_PATH_FAILURE"
        assert incident["severity"] == "critical"

    def test_bandwidth_plus_latency_is_saturation(self):
        frame = alerts((0, "high_bandwidth on WAN1"), (10, "high_latency to 8.8.8.8"))
        assert (
            AlertCorrelationEngine().correlate_alerts(frame).iloc[0]["root_cause"]
            == "BANDWIDTH_SATURATION"
        )

    def test_one_symptom_alone_does_not_fire_a_two_symptom_rule(self):
        """Otherwise a single interface flap gets reported as a path failure."""
        frame = alerts((0, "interface_down Gi0/1"))
        assert (
            AlertCorrelationEngine().correlate_alerts(frame).iloc[0]["root_cause"]
            == "UNCORRELATED"
        )

    def test_a_partial_match_fires_when_the_rule_allows_it(self):
        """The AAA rule needs two of its three symptoms."""
        frame = alerts((0, "radius_timeout from 10.1.1.5"), (5, "auth_failure user jdoe"))
        assert (
            AlertCorrelationEngine().correlate_alerts(frame).iloc[0]["root_cause"]
            == "AAA_SERVICE_DEGRADED"
        )

    def test_matching_ignores_case(self):
        frame = alerts((0, "INTERFACE_DOWN Gi0/1"), (10, "BGP_SESSION_DOWN"))
        assert (
            AlertCorrelationEngine().correlate_alerts(frame).iloc[0]["root_cause"]
            == "SDWAN_PATH_FAILURE"
        )

    def test_rules_are_evaluated_in_declaration_order(self):
        first = CorrelationRule("specific", ("disk",), "DISK_FULL", severity="high")
        second = CorrelationRule("generic", ("disk",), "SOMETHING_ELSE")
        engine = AlertCorrelationEngine(rules=(first, second))
        assert engine.correlate_alerts(alerts((0, "disk usage 95%"))).iloc[0]["rule"] == "specific"

    def test_a_custom_rule_can_be_supplied(self):
        rule = CorrelationRule("power", ("psu_failure", "fan_failure"), "HARDWARE_FAULT")
        engine = AlertCorrelationEngine(rules=(rule,))
        frame = alerts((0, "psu_failure slot 1"), (5, "fan_failure slot 1"))
        assert engine.correlate_alerts(frame).iloc[0]["root_cause"] == "HARDWARE_FAULT"

    @pytest.mark.parametrize("rule", DEFAULT_RULES, ids=lambda r: r.name)
    def test_every_shipped_rule_needs_at_least_one_symptom(self, rule):
        assert rule.required_matches() >= 1
        assert rule.required_matches() <= len(rule.patterns)


class TestUncorrelatedAlerts:
    def test_they_are_returned_not_dropped(self):
        """An alert that silently disappears is how an outage gets missed."""
        frame = alerts((0, "disk usage 81% on switch-04"))
        incidents = AlertCorrelationEngine().correlate_alerts(frame)
        assert len(incidents) == 1
        assert incidents.iloc[0]["root_cause"] == "UNCORRELATED"

    def test_they_keep_the_worst_severity_in_the_group(self):
        frame = alerts((0, "thing a", "low"), (10, "thing b", "high"))
        assert AlertCorrelationEngine().correlate_alerts(frame).iloc[0]["severity"] == "high"

    def test_a_missing_severity_column_is_tolerated(self):
        frame = pd.DataFrame({"timestamp": [BASE], "message": ["odd thing"]})
        assert AlertCorrelationEngine().correlate_alerts(frame).iloc[0]["severity"] == "info"

    def test_no_alert_is_ever_lost(self):
        frame = alerts(
            (0, "interface_down"), (10, "bgp_session_down"), (5000, "disk usage 81%")
        )
        incidents = AlertCorrelationEngine().correlate_alerts(frame)
        assert incidents["alert_count"].sum() == len(frame)


class TestEdgeCases:
    def test_an_empty_frame_returns_an_empty_result_with_the_right_columns(self):
        result = AlertCorrelationEngine().correlate_alerts(pd.DataFrame())
        assert result.empty
        assert "root_cause" in result.columns

    def test_none_is_tolerated(self):
        assert AlertCorrelationEngine().correlate_alerts(None).empty

    def test_a_missing_message_column_raises_a_clear_error(self):
        frame = pd.DataFrame({"timestamp": [BASE]})
        with pytest.raises(KeyError, match="message"):
            AlertCorrelationEngine().correlate_alerts(frame)

    def test_the_input_frame_is_not_mutated(self):
        frame = alerts((0, "interface_down"))
        before = frame.copy(deep=True)
        AlertCorrelationEngine().correlate_alerts(frame)
        pd.testing.assert_frame_equal(frame, before)


class TestReduction:
    def test_a_pure_storm_collapses_to_one_incident(self):
        frame = alerts(*[(i * 10, "interface_down") for i in range(20)])
        assert AlertCorrelationEngine().reduction_ratio(frame) == pytest.approx(0.95)

    def test_unrelated_alerts_do_not_reduce(self):
        frame = alerts(*[(i * 3600, "unique thing") for i in range(5)])
        assert AlertCorrelationEngine().reduction_ratio(frame) == 0.0

    def test_an_empty_stream_reduces_nothing(self):
        assert AlertCorrelationEngine().reduction_ratio(pd.DataFrame()) == 0.0


def test_summarise_renders_one_line_per_incident():
    frame = alerts((0, "interface_down"), (10, "bgp_session_down"))
    incidents = AlertCorrelationEngine().correlate_alerts(frame)
    text = summarise(incidents.to_dict("records"))
    assert text.count("\n") == 0
    assert "SDWAN_PATH_FAILURE" in text
