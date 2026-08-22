#!/usr/bin/env python3
"""Run resumable, reset-isolated PGNW active/passive Unity pairs."""

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_seeds(value):
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError("seed_range_must_ascend")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("at_least_one_seed_required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate_seed")
    return seeds


def condition_order(seed, active_mode="bounded"):
    """Counterbalance order: odd bounded-first, even passive-first."""
    return (active_mode, "passive") if seed % 2 else ("passive", active_mode)


def recording_path(output_dir, mode, seed, tag):
    return output_dir / f"pgnw_{mode}_science_seed{seed}_{tag}.jsonl"


@dataclass
class RecordingStatus:
    exists: bool = False
    complete: bool = False
    elapsed_seconds: float = 0.0
    rows: int = 0
    mode: str = "unknown"
    seed: int = -1
    initial_pickups: int = -1
    error: str = "none"


def inspect_recording(
    path,
    expected_mode,
    expected_seed,
    duration,
    expected_initial_position=None,
    position_tolerance=0.25,
):
    if not path.exists():
        return RecordingStatus()
    try:
        with path.open(encoding="utf-8") as handle:
            first = json.loads(next(handle))
            last = first
            rows = 1
            for line in handle:
                if line.strip():
                    last = json.loads(line)
                    rows += 1
        elapsed = float(last["time"]) - float(first["time"])
        mode = str(last["pgnw_experiment"]["mode"])
        seed = int(last.get("controller_seed", -1))
        initial_pickups = int(first.get("mushroom_pickups_total", -1))
        initial_position = tuple(float(item) for item in first.get("position", ()))
        position_matches = expected_initial_position is None or (
            len(initial_position) == 3
            and all(
                abs(actual - expected) <= position_tolerance
                for actual, expected in zip(initial_position, expected_initial_position)
            )
        )
        tolerance = max(30.0, 0.03 * float(duration))
        complete = (
            elapsed >= float(duration) - tolerance
            and mode == expected_mode
            and seed == expected_seed
            and initial_pickups == 0
            and position_matches
        )
        return RecordingStatus(
            exists=True,
            complete=complete,
            elapsed_seconds=elapsed,
            rows=rows,
            mode=mode,
            seed=seed,
            initial_pickups=initial_pickups,
        )
    except (OSError, StopIteration, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return RecordingStatus(exists=True, error=f"{type(exc).__name__}:{exc}")


def build_command(python, duration, seed, mode, output_path):
    return [
        python,
        "embodied_unity_loop.py",
        "--duration",
        str(duration),
        "--seed",
        str(seed),
        "--shadow-policy",
        "checkpoints/unity_geometry_posttrained/best.pt",
        "--shadow-control",
        "terrain",
        "--shadow-control-confidence",
        "0.0",
        "--shadow-mpc",
        "--systemic-conductor-checkpoint",
        "checkpoints/four_context_conductor/best.json",
        "--systemic-conductor-control",
        "recurrent_mpc_air",
        "--adaptive-gnw-control",
        "bounded",
        "--tiny-scientist-experiment-control",
        mode,
        "--tiny-scientist-experiment-guidance-weight",
        "0.03",
        "--shadow-log",
        str(output_path),
    ]


def reset_unity(host, command_port, listen_port, timeout_seconds, expected_position, position_tolerance):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        receiver.bind(("127.0.0.1", listen_port))
        receiver.settimeout(0.35)
        deadline = time.monotonic() + timeout_seconds
        next_send = 0.0
        payload = json.dumps({"action": "experiment_reset", "mode": "wake"}).encode("utf-8")
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                sender.sendto(payload, (host, command_port))
                next_send = now + 0.5
            try:
                data, _address = receiver.recvfrom(16384)
            except socket.timeout:
                continue
            try:
                state = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if (
                state.get("action") == "experiment_reset"
                and int(state.get("mushroom_pickups_total", -1)) == 0
                and int(state.get("red_mushroom_pickups_total", -1)) == 0
                and abs(float(state.get("x", float("inf"))) - expected_position[0]) <= position_tolerance
                and abs(float(state.get("y", float("inf"))) - expected_position[1]) <= position_tolerance
                and abs(float(state.get("z", float("inf"))) - expected_position[2]) <= position_tolerance
            ):
                return state
    finally:
        receiver.close()
        sender.close()
    raise RuntimeError("unity_experiment_reset_timeout")


def append_manifest(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", required=True, help="Comma list or inclusive range, e.g. 143-147")
    parser.add_argument("--duration", type=float, default=1200.0)
    parser.add_argument(
        "--active-mode",
        choices=("bounded", "committed"),
        default="bounded",
        help="Active PGNW condition paired against passive control.",
    )
    parser.add_argument("--tag", default=time.strftime("%Y%m%d"))
    parser.add_argument("--output-dir", default="outputs/unity_shadow")
    parser.add_argument("--minimum-free-gib", type=float, default=5.0)
    parser.add_argument("--inter-run-delay", type=float, default=3.0)
    parser.add_argument("--reset-timeout", type=float, default=12.0)
    parser.add_argument("--unity-host", default="127.0.0.1")
    parser.add_argument("--unity-port", type=int, default=5055)
    parser.add_argument("--listen-port", type=int, default=5056)
    parser.add_argument("--expected-reset-position", default="0,-0.008,0")
    parser.add_argument(
        "--expected-recorded-start-position",
        default="0,-2.856,0",
        help="Required first telemetry position for this unattended protocol.",
    )
    parser.add_argument("--reset-position-tolerance", type=float, default=0.25)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=0,
        help="Stop after this many newly executed runs; zero means no limit.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    expected_position = tuple(float(item) for item in args.expected_reset_position.split(","))
    if len(expected_position) != 3:
        raise ValueError("expected_reset_position_requires_x_y_z")
    expected_recorded_position = tuple(
        float(item) for item in args.expected_recorded_start_position.split(",")
    )
    if len(expected_recorded_position) != 3:
        raise ValueError("expected_recorded_start_position_requires_x_y_z")
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / f"pgnw_batch_{seeds[0]}-{seeds[-1]}_{args.tag}.jsonl"
    schedule = [
        (seed, mode)
        for seed in seeds
        for mode in condition_order(seed, args.active_mode)
    ]
    print(f"PGNW batch schedule: {schedule}", flush=True)
    new_runs = 0

    for index, (seed, mode) in enumerate(schedule, 1):
        output = recording_path(output_dir, mode, seed, args.tag)
        status = inspect_recording(
            output,
            mode,
            seed,
            args.duration,
            expected_recorded_position,
            args.reset_position_tolerance,
        )
        if status.complete:
            print(f"[{index}/{len(schedule)}] skip complete {mode} seed {seed}: {status.elapsed_seconds:.1f}s", flush=True)
            continue
        if status.exists:
            raise RuntimeError(f"refusing_to_overwrite_incomplete_recording:{output}:{status.error}")
        if args.max_new_runs > 0 and new_runs >= args.max_new_runs:
            print(f"Checkpoint reached after {new_runs} new run(s).", flush=True)
            break

        command = build_command(args.python, args.duration, seed, mode, output)
        if args.dry_run:
            print(f"[{index}/{len(schedule)}] would run: {' '.join(command)}", flush=True)
            new_runs += 1
            continue

        free_gib = shutil.disk_usage(ROOT).free / 1024**3
        if free_gib < args.minimum_free_gib:
            raise RuntimeError(f"free_disk_below_limit:{free_gib:.2f}GiB")
        print(f"[{index}/{len(schedule)}] reset Unity for {mode} seed {seed} ({free_gib:.2f} GiB free)", flush=True)
        reset_unity(
            args.unity_host,
            args.unity_port,
            args.listen_port,
            args.reset_timeout,
            expected_position,
            args.reset_position_tolerance,
        )
        time.sleep(0.5)
        started = time.time()
        print(f"[{index}/{len(schedule)}] start {mode} seed {seed}", flush=True)
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                timeout=args.duration + 180.0,
            )
        except subprocess.TimeoutExpired as exc:
            append_manifest(manifest, {"seed": seed, "mode": mode, "status": "timeout"})
            raise RuntimeError(f"run_timeout:{mode}:seed{seed}") from exc
        finished = inspect_recording(
            output,
            mode,
            seed,
            args.duration,
            expected_recorded_position,
            args.reset_position_tolerance,
        )
        event = {
            "seed": seed,
            "mode": mode,
            "returncode": result.returncode,
            "wall_seconds": round(time.time() - started, 3),
            "recorded_seconds": round(finished.elapsed_seconds, 3),
            "rows": finished.rows,
            "complete": finished.complete,
            "output": str(output.relative_to(ROOT)),
        }
        append_manifest(manifest, event)
        if result.returncode != 0 or not finished.complete:
            raise RuntimeError(f"incomplete_run:{mode}:seed{seed}:{event}")
        new_runs += 1
        print(f"[{index}/{len(schedule)}] complete {mode} seed {seed}: {finished.elapsed_seconds:.1f}s", flush=True)
        time.sleep(max(0.0, args.inter_run_delay))

    print(f"Batch complete. Manifest: {manifest.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
