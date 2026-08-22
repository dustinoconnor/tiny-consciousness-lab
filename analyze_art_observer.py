#!/usr/bin/env python3
"""Summarize passive ART observer behavior from Unity JSONL telemetry."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze_rows(rows):
    category_counts = Counter()
    evidence_counts = Counter()
    category_evidence = defaultdict(Counter)
    course_categories = defaultdict(Counter)
    frames = 0
    resonant = 0
    novel = 0
    unknown = 0
    label_agreement = 0
    first_time = None
    last_time = None
    mismatch_total = 0
    switches = 0
    successes = 0
    failures = 0

    for row in rows:
        category = row.get("art_category")
        if not category:
            continue
        frames += 1
        timestamp = row.get("time")
        if isinstance(timestamp, (int, float)):
            first_time = timestamp if first_time is None else min(first_time, timestamp)
            last_time = timestamp if last_time is None else max(last_time, timestamp)
        evidence = str(row.get("art_evidence_label", "unobserved"))
        category_label = str(row.get("art_category_label", "unlearned"))
        course = str(row.get("trap_course", "natural_terrain"))
        category_counts[category] += 1
        evidence_counts[evidence] += 1
        category_evidence[category][evidence] += 1
        course_categories[course][category] += 1
        resonant += bool(row.get("art_resonance", False))
        novel += bool(row.get("art_novel", False))
        unknown += bool(row.get("art_unknown", False))
        label_agreement += category_label == evidence
        mismatch_total = max(mismatch_total, int(row.get("art_mismatch_resets_total", 0) or 0))
        switches = max(switches, int(row.get("art_category_switches", 0) or 0))
        successes = max(successes, int(row.get("trap_successes", 0) or 0))
        failures = max(failures, int(row.get("trap_failures", 0) or 0))

    duration_seconds = max(0.0, (last_time or 0.0) - (first_time or 0.0))
    category_summary = {}
    purity_mass = 0
    for category, count in category_counts.items():
        evidence = category_evidence[category]
        dominant_label, dominant_count = evidence.most_common(1)[0]
        purity = dominant_count / count
        purity_mass += dominant_count
        category_summary[category] = {
            "frames": count,
            "dominant_evidence": dominant_label,
            "purity": round(purity, 6),
            "evidence_counts": dict(evidence),
        }

    return {
        "frames": frames,
        "duration_minutes": round(duration_seconds / 60.0, 4),
        "category_count": len(category_counts),
        "weighted_category_purity": round(purity_mass / max(frames, 1), 6),
        "current_label_agreement": round(label_agreement / max(frames, 1), 6),
        "resonance_rate": round(resonant / max(frames, 1), 6),
        "novelty_rate": round(novel / max(frames, 1), 6),
        "unknown_rate": round(unknown / max(frames, 1), 6),
        "mismatch_resets_total": mismatch_total,
        "category_switches": switches,
        "category_switches_per_minute": round(switches / max(duration_seconds / 60.0, 1e-8), 4),
        "trap_successes": successes,
        "trap_timeouts": failures,
        "evidence_counts": dict(evidence_counts),
        "categories": category_summary,
        "course_categories": {
            course: dict(counts)
            for course, counts in sorted(course_categories.items())
        },
        "claim_boundary": (
            "This summarizes a passive online Fuzzy-ART software observer. Category stability "
            "or semantic coherence does not establish biological ART, consciousness, or qualia."
        ),
    }


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    metrics = analyze_rows(iter_jsonl(args.log))
    output = args.output or args.log.with_name(f"{args.log.stem}_art_analysis.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
