"""Alert correlation.

A single network fault produces dozens of alerts: the interface drops, BGP tears
down, the tunnel flaps, the monitoring system pages three times. An operator
should see one incident, not forty lines.

This engine groups alerts that arrive close together in time and then labels a
group with a root cause when its alert types match a known failure signature.
It is deliberately rule-based and dependency-light: pandas and the standard
library, nothing else. An earlier version loaded a 90 MB sentence-transformer
model on construction and never used it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pandas as pd

DEFAULT_TIME_WINDOW = 300  # seconds


@dataclass(frozen=True)
class CorrelationRule:
    """A failure signature.

    ``patterns`` are substrings matched case-insensitively against the alert
    message. ``min_match_ratio`` is the fraction of them that must be present
    for the rule to fire, so a rule can tolerate one missing symptom without
    matching on a single coincidence.
    """

    name: str
    patterns: Sequence[str]
    root_cause: str
    severity: str = "high"
    time_window: int = DEFAULT_TIME_WINDOW
    min_match_ratio: float = 1.0

    def required_matches(self) -> int:
        import math

        return max(1, math.ceil(len(self.patterns) * self.min_match_ratio))


DEFAULT_RULES: tuple[CorrelationRule, ...] = (
    CorrelationRule(
        name="cascading_failure",
        patterns=("interface_down", "bgp_session_down"),
        root_cause="SDWAN_PATH_FAILURE",
        severity="critical",
        time_window=300,
    ),
    CorrelationRule(
        name="capacity_issue",
        patterns=("high_bandwidth", "high_latency"),
        root_cause="BANDWIDTH_SATURATION",
        severity="high",
        time_window=600,
    ),
    CorrelationRule(
        name="auth_infrastructure",
        patterns=("radius_timeout", "auth_failure", "dot1x_failure"),
        root_cause="AAA_SERVICE_DEGRADED",
        severity="high",
        time_window=300,
        # Two of the three symptoms are enough: a client may never reach dot1x
        # if RADIUS is already timing out.
        min_match_ratio=0.66,
    ),
)


@dataclass
class AlertCorrelationEngine:
    """Group alerts into incidents and attach a probable root cause."""

    rules: tuple[CorrelationRule, ...] = DEFAULT_RULES
    time_window: int = DEFAULT_TIME_WINDOW
    message_column: str = "message"
    timestamp_column: str = "timestamp"
    _last_groups: list[list] = field(default_factory=list, repr=False)

    # -- public API ---------------------------------------------------------

    def correlate_alerts(self, alerts: pd.DataFrame) -> pd.DataFrame:
        """Return one row per correlated incident.

        Alerts that match no rule are not dropped: they are returned as
        single-alert incidents with an ``UNCORRELATED`` root cause, because an
        alert that silently disappears is how outages get missed.
        """
        columns = [
            "group_id",
            "rule",
            "root_cause",
            "severity",
            "alert_count",
            "first_seen",
            "last_seen",
            "alert_ids",
        ]
        if alerts is None or alerts.empty:
            return pd.DataFrame(columns=columns)

        self._require_columns(alerts)
        frame = alerts.copy()
        frame[self.timestamp_column] = pd.to_datetime(frame[self.timestamp_column])

        incidents = []
        for position, indices in enumerate(self.group_by_time(frame)):
            group = frame.loc[indices]
            rule = self.match_rule(group)
            incidents.append(
                {
                    "group_id": f"corr_{position:04d}",
                    "rule": rule.name if rule else "uncorrelated",
                    "root_cause": rule.root_cause if rule else "UNCORRELATED",
                    "severity": rule.severity if rule else self._fallback_severity(group),
                    "alert_count": len(indices),
                    "first_seen": group[self.timestamp_column].min(),
                    "last_seen": group[self.timestamp_column].max(),
                    "alert_ids": list(indices),
                }
            )

        return pd.DataFrame(incidents, columns=columns)

    def reduction_ratio(self, alerts: pd.DataFrame) -> float:
        """How much less an operator has to read: 1 - incidents/alerts.

        Reported rather than asserted. Run it on your own alert stream; the
        figure depends entirely on how bursty your alerts are.
        """
        if alerts is None or alerts.empty:
            return 0.0
        incidents = self.correlate_alerts(alerts)
        return 1.0 - (len(incidents) / len(alerts))

    # -- internals ----------------------------------------------------------

    def _require_columns(self, alerts: pd.DataFrame) -> None:
        missing = {self.timestamp_column, self.message_column} - set(alerts.columns)
        if missing:
            raise KeyError(f"alerts frame is missing required column(s): {sorted(missing)}")

    def group_by_time(self, frame: pd.DataFrame) -> list[list]:
        """Cluster alerts by gap, not by distance from the first alert.

        A storm lasting ten minutes is one incident even though its last alert
        is far from its first. Splitting on the gap between consecutive alerts
        keeps that storm together while still separating unrelated bursts.
        """
        ordered = frame.sort_values(self.timestamp_column)
        groups: list[list] = []
        current: list = []
        previous = None

        for index, row in ordered.iterrows():
            moment = row[self.timestamp_column]
            if previous is not None and (moment - previous).total_seconds() > self.time_window:
                groups.append(current)
                current = []
            current.append(index)
            previous = moment

        if current:
            groups.append(current)

        self._last_groups = groups
        return groups

    def match_rule(self, group: pd.DataFrame) -> CorrelationRule | None:
        """The first rule whose signature the group satisfies.

        Rules are evaluated in declaration order, so put the most specific one
        first.
        """
        messages = group[self.message_column].astype(str).str.lower()
        for rule in self.rules:
            hits = sum(1 for pattern in rule.patterns if messages.str.contains(pattern.lower()).any())
            if hits >= rule.required_matches():
                return rule
        return None

    @staticmethod
    def _fallback_severity(group: pd.DataFrame) -> str:
        """Preserve the worst severity an uncorrelated group carried."""
        if "severity" not in group.columns:
            return "info"
        order = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        values = [str(v).lower() for v in group["severity"].dropna()]
        if not values:
            return "info"
        return max(values, key=lambda v: order.get(v, 0))


def summarise(incidents: Iterable[dict]) -> str:
    """One line per incident, for a terminal or a chat message."""
    return "\n".join(
        f"{i['group_id']}  {i['root_cause']:<24} {i['alert_count']:>3} alerts  [{i['severity']}]"
        for i in incidents
    )
