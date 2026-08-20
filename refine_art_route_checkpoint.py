#!/usr/bin/env python3
"""Refine ART route precedents using their recorded terminal approach geometry."""

import argparse
import json
import math
from pathlib import Path

import torch


def turn_angle(first, second, third):
    incoming = (second[0] - first[0], second[1] - first[1])
    outgoing = (third[0] - second[0], third[1] - second[1])
    first_length = math.hypot(*incoming)
    second_length = math.hypot(*outgoing)
    if first_length <= 1e-6 or second_length <= 1e-6:
        return 0.0
    cosine = max(
        -1.0,
        min(
            1.0,
            (incoming[0] * outgoing[0] + incoming[1] * outgoing[1])
            / (first_length * second_length),
        ),
    )
    return math.degrees(math.acos(cosine))


def prune_reversal_waypoints(waypoints, threshold_degrees=120.0):
    points = [tuple(map(float, point)) for point in waypoints]
    removed = 0
    changed = True
    while changed and len(points) >= 3:
        changed = False
        for index in range(1, len(points) - 1):
            if turn_angle(points[index - 1], points[index], points[index + 1]) <= threshold_degrees:
                continue
            points.pop(index)
            removed += 1
            changed = True
            break
    return [list(point) for point in points], removed


def terminal_alignment(route):
    source = Path(route["source_recording"])
    episode = int(route["source_episode"])
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    visible = next(
        (
            row
            for row in rows
            if int(row.get("trap_episode", 0) or 0) == episode
            and bool(row.get("food_visible"))
        ),
        None,
    )
    points = route.get("waypoints", [])
    if visible is None or len(points) < 2:
        return 0.0
    heading = (
        float(points[-1][0]) - float(points[-2][0]),
        float(points[-1][1]) - float(points[-2][1]),
    )
    food = visible.get("food_world", [0.0, 0.0])
    heading_length = math.hypot(*heading)
    food_length = math.hypot(float(food[0]), float(food[1]))
    if heading_length <= 1e-6 or food_length <= 1e-6:
        return 0.0
    return max(
        -1.0,
        min(
            1.0,
            (heading[0] * float(food[0]) + heading[1] * float(food[1]))
            / (heading_length * food_length),
        ),
    )


def refine(payload):
    output = dict(payload)
    routes = []
    for source_route in payload.get("routes", []):
        route = dict(source_route)
        route["terminal_alignment"] = terminal_alignment(route)
        route["original_waypoint_count"] = len(route.get("waypoints", []))
        route["waypoints"], route["pruned_reversals"] = prune_reversal_waypoints(
            route.get("waypoints", [])
        )
        routes.append(route)
    output["routes"] = routes
    output["refinement"] = "terminal_alignment_and_reversal_pruning_v1"
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    refined = refine(payload)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(refined, args.destination)
    for route in refined["routes"]:
        print(
            f"{route['route_id']}: alignment={route['terminal_alignment']:.3f} "
            f"waypoints={route['original_waypoint_count']}->{len(route['waypoints'])}"
        )


if __name__ == "__main__":
    main()
