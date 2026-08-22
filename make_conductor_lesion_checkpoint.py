#!/usr/bin/env python3
"""Create a reproducible specialist-selection lesion from a conductor checkpoint."""

import argparse
import copy
import json
from pathlib import Path


def force_recurrent(payload, context="familiar_hidden_goal"):
    lesion = copy.deepcopy(payload)
    values = lesion["q_values"][context]
    values["recurrent"], values["episodic"] = (
        float(values["episodic"]),
        float(values["recurrent"]),
    )
    lesion["lesion"] = {
        "type": "forced_recurrent_specialist_selection",
        "context": context,
    }
    return lesion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.source.read_text(encoding="utf-8"))
    lesion = force_recurrent(payload)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(
        json.dumps(lesion, indent=2) + "\n",
        encoding="utf-8",
    )
    values = lesion["q_values"]["familiar_hidden_goal"]
    print(
        f"forced recurrent lesion: recurrent={values['recurrent']:.6f} "
        f"episodic={values['episodic']:.6f}"
    )


if __name__ == "__main__":
    main()
