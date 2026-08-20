#!/usr/bin/env python3
"""Run deterministic six-course Unity trials across fixed controller seeds."""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


DEFAULT_SEEDS = (11, 23, 37, 53, 71)


def completed_episodes(path):
    completed = []
    previous = None
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            state = (
                int(row.get("trap_episode", 0) or 0),
                str(row.get("trap_course", "inactive")),
                str(row.get("trap_outcome", "inactive")),
            )
            if (
                previous is not None
                and state[0] == previous[0]
                and previous[2] == "running"
                and state[2] in {"success", "timeout"}
            ):
                completed.append({"episode": state[0], "course": state[1], "outcome": state[2]})
            previous = state
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--episodes-per-seed", type=int, default=6)
    parser.add_argument("--checkpoint", default="checkpoints/unity_geometry_posttrained/best.pt")
    parser.add_argument(
        "--hidden-goal-adapter",
        default=None,
        help="Optional frozen-GRU hidden-goal adapter checkpoint forwarded to each course run.",
    )
    parser.add_argument(
        "--hidden-goal-adapter-confidence",
        type=float,
        default=0.30,
        help="Minimum hidden-goal adapter confidence forwarded to each course run.",
    )
    parser.add_argument(
        "--hidden-goal-adapter-commit-seconds",
        type=float,
        default=0.4,
        help="Safe-action commitment duration forwarded to each course run.",
    )
    parser.add_argument(
        "--hidden-goal-adapter-lateral-bias",
        type=float,
        default=0.0,
        help="Soft opposite-side logit penalty forwarded to each course run.",
    )
    parser.add_argument(
        "--passive-conductor",
        action="store_true",
        help="Enable the passive embodied conductor observer for each run.",
    )
    parser.add_argument(
        "--conductor-checkpoint",
        default=None,
        help="Optional offline conductor checkpoint forwarded to each run.",
    )
    parser.add_argument(
        "--conductor-control",
        choices=["passive", "familiar_hidden_goal"],
        default="passive",
        help="Bounded conductor control mode forwarded to each run.",
    )
    parser.add_argument("--minimum-success-rate", type=float, default=0.90)
    parser.add_argument("--per-seed-timeout", type=float, default=720.0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"outputs/unity_shadow/fixed_course_benchmark_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []

    for index, seed in enumerate(args.seeds, 1):
        telemetry = output_dir / f"seed_{seed}.jsonl"
        terminal = output_dir / f"seed_{seed}_terminal.log"
        command = [
            sys.executable,
            "embodied_unity_loop.py",
            "--seed",
            str(seed),
            "--course-episodes",
            str(args.episodes_per_seed),
            "--shadow-policy",
            args.checkpoint,
            "--shadow-control",
            "course",
            "--shadow-control-confidence",
            "0.0",
            "--shadow-mpc",
            "--shadow-log",
            str(telemetry),
        ]
        if args.hidden_goal_adapter:
            command.extend(
                [
                    "--hidden-goal-adapter",
                    args.hidden_goal_adapter,
                    "--hidden-goal-adapter-confidence",
                    str(args.hidden_goal_adapter_confidence),
                    "--hidden-goal-adapter-commit-seconds",
                    str(args.hidden_goal_adapter_commit_seconds),
                    "--hidden-goal-adapter-lateral-bias",
                    str(args.hidden_goal_adapter_lateral_bias),
                ]
            )
        if args.passive_conductor:
            command.append("--passive-conductor")
        if args.conductor_checkpoint:
            command.extend(["--conductor-checkpoint", args.conductor_checkpoint])
        if args.conductor_control != "passive":
            command.extend(["--conductor-control", args.conductor_control])
        print(f"[{index}/{len(args.seeds)}] seed {seed}: starting {args.episodes_per_seed} episodes", flush=True)
        started = time.time()
        returncode = None
        error = None
        try:
            with terminal.open("w", encoding="utf-8") as output:
                result = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parent,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=args.per_seed_timeout,
                    check=False,
                )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            error = "seed_timeout"

        episodes = completed_episodes(telemetry) if telemetry.exists() else []
        successes = sum(item["outcome"] == "success" for item in episodes)
        timeouts = sum(item["outcome"] == "timeout" for item in episodes)
        run = {
            "seed": seed,
            "returncode": returncode,
            "error": error,
            "duration_seconds": round(time.time() - started, 3),
            "successes": successes,
            "timeouts": timeouts,
            "success_rate": round(successes / len(episodes), 6) if episodes else 0.0,
            "episodes": episodes,
            "telemetry": str(telemetry),
            "terminal_log": str(terminal),
        }
        runs.append(run)
        print(f"    result: {successes} wins / {timeouts} timeouts", flush=True)
        time.sleep(3.0)

    total_episodes = sum(len(run["episodes"]) for run in runs)
    total_successes = sum(run["successes"] for run in runs)
    total_timeouts = sum(run["timeouts"] for run in runs)
    course_results = Counter(
        (episode["course"], episode["outcome"])
        for run in runs
        for episode in run["episodes"]
    )
    success_rate = total_successes / total_episodes if total_episodes else 0.0
    expected_episodes = len(args.seeds) * args.episodes_per_seed
    report = {
        "protocol": {
            "seeds": args.seeds,
            "episodes_per_seed": args.episodes_per_seed,
            "checkpoint": args.checkpoint,
            "hidden_goal_adapter": args.hidden_goal_adapter,
            "hidden_goal_adapter_confidence": args.hidden_goal_adapter_confidence,
            "hidden_goal_adapter_commit_seconds": args.hidden_goal_adapter_commit_seconds,
            "hidden_goal_adapter_lateral_bias": args.hidden_goal_adapter_lateral_bias,
            "passive_conductor": args.passive_conductor,
            "conductor_checkpoint": args.conductor_checkpoint,
            "conductor_control": args.conductor_control,
            "adaptive_stochastic_mpc": True,
            "art_observer": "passive",
            "minimum_success_rate": args.minimum_success_rate,
        },
        "runs": runs,
        "aggregate": {
            "expected_episodes": expected_episodes,
            "completed_episodes": total_episodes,
            "successes": total_successes,
            "timeouts": total_timeouts,
            "success_rate": round(success_rate, 6),
            "course_results": {
                f"{course}:{outcome}": count
                for (course, outcome), count in sorted(course_results.items())
            },
            "complete": total_episodes == expected_episodes,
            "passed": total_episodes == expected_episodes and success_rate >= args.minimum_success_rate,
        },
        "claim_boundary": (
            "This fixed-seed Unity regression estimates controller reliability. "
            "The passive ART observer cannot improve or degrade action selection."
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Wrote {report_path}")
    return 0 if report["aggregate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
