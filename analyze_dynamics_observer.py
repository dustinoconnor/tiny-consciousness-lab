#!/usr/bin/env python3
"""Evaluate passive synchrony and criticality proxies against Unity events."""

import argparse
import json
import math
from collections import Counter, deque
from pathlib import Path


METRICS = [
    "sync_coherence",
    "sync_bus_pressure",
    "criticality_score",
    "criticality_propagation_ratio",
    "criticality_recommended_gain",
]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sample_std(values):
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def standardized_difference(positive, negative):
    if len(positive) < 2 or len(negative) < 2:
        return 0.0
    variance = ((len(positive) - 1) * sample_std(positive) ** 2 + (len(negative) - 1) * sample_std(negative) ** 2) / max(1, len(positive) + len(negative) - 2)
    return (mean(positive) - mean(negative)) / math.sqrt(variance) if variance > 1e-12 else 0.0


def roc_auc(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positive_count = sum(bool(label) for _, label in pairs)
    negative_count = len(pairs) - positive_count
    if not positive_count or not negative_count:
        return 0.5
    positive_rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = 0.5 * ((index + 1) + end)
        positive_rank_sum += average_rank * sum(
            bool(label) for _, label in pairs[index:end]
        )
        index = end
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def friction(row):
    return bool(
        row.get("blocked")
        or row.get("body_collision")
        or row.get("stuck")
        or row.get("fallback_active")
        or float(row.get("trap_pressure", 0.0) or 0.0) >= 0.35
    )


def forward_labels(rows, horizon):
    labels = []
    for index in range(len(rows)):
        episode = rows[index].get("trap_episode")
        future = rows[index + 1 : index + horizon + 1]
        labels.append(any(friction(row) for row in future if row.get("trap_episode") == episode))
    return labels


def metric_summary(rows, metric, labels):
    values = [float(row.get(metric, 0.0) or 0.0) for row in rows]
    positive = [value for value, label in zip(values, labels) if label]
    negative = [value for value, label in zip(values, labels) if not label]
    auc = roc_auc(values, labels)
    inverse_auc = roc_auc([-value for value in values], labels)
    return {
        "event_mean": mean(positive),
        "control_mean": mean(negative),
        "standardized_difference": standardized_difference(positive, negative),
        "predictive_auc": max(auc, inverse_auc),
        "predictive_direction": "higher" if auc >= inverse_auc else "lower",
        "event_frames": len(positive),
        "control_frames": len(negative),
    }


def analyze(path, hz=5.0):
    horizon = max(1, int(round(3.0 * hz)))
    pending = deque()
    current_labels = []
    future_labels = []
    current_values = {metric: [] for metric in METRICS}
    future_values = {metric: [] for metric in METRICS}
    courses = {}
    course_counts = Counter()
    regime_counts = Counter()
    binding_ready = 0
    frames = 0
    first_time = None
    last_time = None

    def finalize_future(item, following):
        label = any(
            later["friction"] and later["episode"] == item["episode"]
            for later in following
        )
        future_labels.append(label)
        for metric in METRICS:
            future_values[metric].append(item["metrics"][metric])

    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frames += 1
            timestamp = float(row.get("time", frames / hz))
            first_time = timestamp if first_time is None else first_time
            last_time = timestamp
            event = friction(row)
            episode = int(row.get("trap_episode", 0) or 0)
            metrics = {
                metric: float(row.get(metric, 0.0) or 0.0) for metric in METRICS
            }
            current_labels.append(event)
            for metric, value in metrics.items():
                current_values[metric].append(value)
            pending.append({"episode": episode, "friction": event, "metrics": metrics})
            if len(pending) > horizon:
                oldest = pending.popleft()
                finalize_future(oldest, pending)

            course_name = row.get("trap_course")
            course_counts[course_name] += 1
            course = courses.setdefault(
                str(episode),
                {
                    "course": course_name,
                    "frames": 0,
                    "success": False,
                    "timeout": False,
                    "friction_frames": 0,
                    "fallback_frames": 0,
                },
            )
            course["frames"] += 1
            course["success"] |= row.get("trap_outcome") == "success"
            course["timeout"] |= row.get("trap_outcome") == "timeout"
            course["friction_frames"] += event
            course["fallback_frames"] += bool(row.get("fallback_active"))
            binding_ready += bool(row.get("sync_binding_ready"))
            regime_counts[row.get("criticality_regime")] += 1

    while pending:
        oldest = pending.popleft()
        finalize_future(oldest, pending)

    duration = (
        max(0.0, last_time - first_time)
        if first_time is not None and last_time is not None
        else frames / hz
    )
    return {
        "source": str(path),
        "frames": frames,
        "duration_seconds": duration,
        "courses": courses,
        "course_counts": dict(course_counts),
        "current_friction_rate": mean(current_labels),
        "future_3s_friction_rate": mean(future_labels),
        "current_event_association": {
            metric: metric_summary(
                [{metric: value} for value in current_values[metric]],
                metric,
                current_labels,
            )
            for metric in METRICS
        },
        "future_3s_prediction": {
            metric: metric_summary(
                [{metric: value} for value in future_values[metric]],
                metric,
                future_labels,
            )
            for metric in METRICS
        },
        "binding_ready_rate": binding_ready / frames if frames else 0.0,
        "regime_counts": dict(regime_counts),
        "interpretation_rule": {
            "promising_auc": 0.65,
            "promising_absolute_standardized_difference": 0.35,
            "note": "These exploratory thresholds nominate an intervention test; they do not establish causal benefit.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.log, args.hz)
    output = args.output or Path("outputs/dynamics_observer_course_analysis.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"frames={result['frames']} duration={result['duration_seconds']:.1f}s")
    for metric, values in result["future_3s_prediction"].items():
        print(f"{metric:36s} AUC={values['predictive_auc']:.3f} direction={values['predictive_direction']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
