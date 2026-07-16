#!/usr/bin/env python3
"""Export trained PyTorch MIDI policies to the compact NumPy app format."""

from pathlib import Path

import numpy as np
import torch


CHECKPOINTS = {
    Path("checkpoints/midi_transfer/recurrent_valence.pt"): Path("checkpoints/midi_transfer/recurrent_valence_runtime.npz"),
    Path("checkpoints/midi_transfer/learned_rhythm.pt"): Path("checkpoints/midi_transfer/learned_rhythm_runtime.npz"),
}


def export(source, destination):
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        weight_ih=state["cell.weight_ih"].numpy(),
        weight_hh=state["cell.weight_hh"].numpy(),
        bias_ih=state["cell.bias_ih"].numpy(),
        bias_hh=state["cell.bias_hh"].numpy(),
        head_weight=state["head.weight"].numpy(),
        head_bias=state["head.bias"].numpy(),
    )
    print(destination)


def main():
    for source, destination in CHECKPOINTS.items():
        export(source, destination)


if __name__ == "__main__":
    main()
