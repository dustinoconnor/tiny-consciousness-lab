#!/usr/bin/env python3
"""Replay Unity telemetry through a revised passive resource-memory policy."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from analyze_resource_memory_unity import load_rows, summarize
from terrain_resource_memory import PassiveTerrainResourceMemory


def body_packet(row):
    position = row.get("position", [0.0, 0.0, 0.0])
    return {
        "x": position[0],
        "z": position[2],
        "mushroom_pickups_total": row.get("mushroom_pickups_total", 0),
        "food_visible": row.get("food_visible", False),
    }


def replay(memory, rows, capture=False):
    snapshots = []
    for row in rows:
        memory.update(body_packet(row), row.get("hunger", 0.0))
        if capture:
            snapshots.append(
                {
                    "time": row["time"],
                    "position": row.get("position"),
                    "hunger": row.get("hunger", 0.0),
                    "food_visible": row.get("food_visible", False),
                    "mushroom_pickups_total": row.get(
                        "mushroom_pickups_total", 0
                    ),
                    "resource_memory_enabled": memory.enabled,
                    "resource_memory_hunger_gate": memory.hunger_gate,
                    "resource_memory_active": memory.active,
                    "resource_memory_recommendation": memory.recommendation,
                    "resource_memory_target": (
                        list(memory.target)
                        if memory.target is not None
                        else None
                    ),
                    "resource_memory_distance": memory.distance,
                    "resource_memory_regions": len(memory.entries),
                    "resource_memory_encodings": memory.encodings,
                    "resource_memory_queries": memory.queries,
                    "resource_memory_recommendations": memory.recommendations,
                    "resource_memory_pickups_after_recommendation": (
                        memory.pickups_after_recommendation
                    ),
                    "resource_memory_stale_arrivals": (
                        memory.counterfactual_stale_arrivals
                    ),
                    "resource_memory_release_reason": memory.release_reason,
                    "resource_memory_action_influence": memory.action_influence,
                }
            )
    return snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("warmup", help="Recording used to construct initial memory.")
    parser.add_argument("evaluation", help="Recording replayed for recommendations.")
    parser.add_argument(
        "--output",
        default="outputs/resource_memory_policy_replay.json",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        memory_path = Path(directory) / "resource_memory.json"
        warmup_memory = PassiveTerrainResourceMemory(memory_path)
        replay(warmup_memory, load_rows(Path(args.warmup)))
        memory = PassiveTerrainResourceMemory(memory_path)
        snapshots = replay(
            memory,
            load_rows(Path(args.evaluation)),
            capture=True,
        )
    result = summarize(snapshots)
    result["warmup_recording"] = args.warmup
    result["evaluation_recording"] = args.evaluation
    result["replay_only"] = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
