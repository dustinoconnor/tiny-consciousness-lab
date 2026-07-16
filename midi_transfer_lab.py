#!/usr/bin/env python3
"""Cross-domain recurrent/valence transfer test using procedural MIDI phrases.

This lab compares feedforward and recurrent sequence models on the same
procedural motif curriculum.  Valence is a grounded scalar derived from the
previous note's harmonic fit and cadence resolution; it is an observation, not
a label claiming subjective feeling.

The generated music is deliberately small and symbolic so the full experiment
runs on CPU and does not require a MIDI package or an external music corpus.
"""

import argparse
import json
import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


OUT = Path("outputs")
MIDI_OUT = OUT / "midi_transfer"
METRICS_PATH = OUT / "midi_transfer_metrics.json"
PLOT_PATH = OUT / "midi_transfer_comparison.png"
CHECKPOINT_DIR = Path("checkpoints/midi_transfer")

DEGREES = 7
STEPS = 64
MOTIF = 8
CONTEXT = 8
CHORDS = ((0, 2, 4), (3, 5, 0), (4, 6, 1))  # I, IV, V in scale degrees
PROGRESSION = (0, 0, 1, 2, 0, 1, 2, 0)
SECTION = (0, 1, 2, 2, 0, 1, 3, 0)
SCALE = (0, 2, 4, 5, 7, 9, 11)
FEATURE_DIM = DEGREES + 8 + 3 + 4 + 1


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def motif_from_rng(rng):
    motif = [int(rng.integers(0, DEGREES))]
    for _ in range(MOTIF - 1):
        step = int(rng.choice((-2, -1, 1, 2), p=(0.12, 0.38, 0.38, 0.12)))
        motif.append(int(np.clip(motif[-1] + step, 0, DEGREES - 1)))
    return np.asarray(motif, dtype=np.int64)


def fit_to_chord(notes, chord_index, rng, strength=0.72):
    result = notes.copy()
    chord = CHORDS[chord_index]
    for index, note in enumerate(result):
        if rng.random() < strength and int(note) not in chord:
            result[index] = min(chord, key=lambda candidate: abs(candidate - int(note)))
    return result


def make_phrase(rng):
    """Create A/A'/B/B'/A/A'/cadence/A form with a delayed motif return."""
    a = motif_from_rng(rng)
    av = a.copy()
    change = int(rng.integers(1, MOTIF - 1))
    av[change] = int(np.clip(av[change] + rng.choice((-1, 1)), 0, DEGREES - 1))
    b = fit_to_chord(motif_from_rng(rng), 1, rng)
    bv = np.roll(b, 1)
    transposed = np.clip(a + int(rng.choice((-1, 1))), 0, DEGREES - 1)
    cadence = fit_to_chord(motif_from_rng(rng), 2, rng, strength=0.90)
    cadence[-2:] = (4, 6)
    final_a = a.copy()
    final_a[-1] = 0
    return np.concatenate((a, av, b, bv, a, transposed, cadence, final_a))


def grounded_valence(previous_note, step):
    """Harmonic fit plus an explicit V-to-I resolution signal in [-1, 1]."""
    if step <= 0:
        return 0.0
    previous_bar = min(7, (step - 1) // MOTIF)
    chord_index = PROGRESSION[previous_bar]
    value = 0.35 if int(previous_note) in CHORDS[chord_index] else -0.35
    if step % MOTIF == 0:
        next_chord = PROGRESSION[min(7, step // MOTIF)]
        if chord_index == 2 and next_chord == 0:
            value += 0.45 if int(previous_note) in (4, 6) else -0.25
    return float(np.clip(value, -1.0, 1.0))


def feature(previous_note, step, valence_enabled):
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)
    vector[int(previous_note)] = 1.0
    offset = DEGREES
    vector[offset + step % 8] = 1.0
    offset += 8
    bar = min(7, step // MOTIF)
    vector[offset + PROGRESSION[bar]] = 1.0
    offset += 3
    vector[offset + SECTION[bar]] = 1.0
    if valence_enabled:
        vector[-1] = grounded_valence(previous_note, step)
    return vector


def encode_phrases(phrases, valence_enabled):
    rows = []
    for phrase in phrases:
        inputs = np.zeros((STEPS, FEATURE_DIM), dtype=np.float32)
        previous = 0
        for step in range(STEPS):
            inputs[step] = feature(previous, step, valence_enabled)
            previous = int(phrase[step])
        rows.append(inputs)
    return torch.tensor(np.stack(rows)), torch.tensor(np.stack(phrases))


class WindowMLP(nn.Module):
    def __init__(self, hidden=67):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM * CONTEXT, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, DEGREES),
        )

    def forward(self, x, reset_memory=False):
        del reset_memory
        padded = F.pad(x.transpose(1, 2), (CONTEXT - 1, 0))
        windows = padded.unfold(2, CONTEXT, 1).permute(0, 2, 3, 1)
        return self.net(windows.reshape(x.shape[0], x.shape[1], -1))


class RecurrentPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(FEATURE_DIM, hidden)
        self.head = nn.Linear(hidden, DEGREES)

    def forward(self, x, reset_memory=False):
        hidden = torch.zeros(x.shape[0], self.hidden, device=x.device)
        outputs = []
        for step in range(x.shape[1]):
            if reset_memory:
                hidden = torch.zeros_like(hidden)
            hidden = self.cell(x[:, step], hidden)
            outputs.append(self.head(hidden))
        return torch.stack(outputs, dim=1)


@dataclass
class Condition:
    name: str
    recurrent: bool
    valence: bool


CONDITIONS = (
    Condition("feedforward", False, False),
    Condition("feedforward_valence", False, True),
    Condition("recurrent", True, False),
    Condition("recurrent_valence", True, True),
)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def train_model(condition, train_phrases, epochs, seed):
    seed_all(seed)
    model = RecurrentPolicy() if condition.recurrent else WindowMLP()
    x, y = encode_phrases(train_phrases, condition.valence)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-3)
    generator = torch.Generator().manual_seed(seed + 91)
    batch_size = 64
    losses = []
    for _ in range(epochs):
        permutation = torch.randperm(len(x), generator=generator)
        epoch_losses = []
        for start in range(0, len(x), batch_size):
            indexes = permutation[start : start + batch_size]
            logits = model(x[indexes])
            loss = F.cross_entropy(logits.reshape(-1, DEGREES), y[indexes].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    return model.eval(), losses


def teacher_forced_metrics(model, condition, phrases, reset_memory=False):
    x, y = encode_phrases(phrases, condition.valence)
    with torch.no_grad():
        logits = model(x, reset_memory=reset_memory)
        prediction = logits.argmax(dim=-1)
        loss = F.cross_entropy(logits.reshape(-1, DEGREES), y.reshape(-1))
    correct = prediction.eq(y)
    return {
        "cross_entropy": float(loss),
        "next_note_accuracy": float(correct.float().mean()),
        "delayed_motif_accuracy": float(correct[:, 32:40].float().mean()),
        "final_return_accuracy": float(correct[:, 56:64].float().mean()),
        "cadence_tonic_accuracy": float(prediction[:, -1].eq(0).float().mean()),
    }


def generate(model, condition, temperature=0.75, seed=0):
    seed_all(seed)
    generated = []
    history = []
    hidden = None
    if condition.recurrent:
        hidden = torch.zeros(1, model.hidden)
    previous = 0
    for step in range(STEPS):
        current = torch.tensor(feature(previous, step, condition.valence)).view(1, 1, -1)
        if condition.recurrent:
            hidden = model.cell(current[:, 0], hidden)
            logits = model.head(hidden)[0]
        else:
            history.append(current[0, 0])
            window = history[-CONTEXT:]
            if len(window) < CONTEXT:
                window = [torch.zeros(FEATURE_DIM)] * (CONTEXT - len(window)) + window
            logits = model.net(torch.cat(window))
        probabilities = torch.softmax(logits / temperature, dim=-1)
        note = int(torch.multinomial(probabilities, 1))
        generated.append(note)
        previous = note
    return generated


def generation_metrics(notes):
    notes = np.asarray(notes, dtype=np.int64)
    motif_similarity = float(np.mean(notes[32:40] == notes[:8]))
    final_similarity = float(np.mean(notes[56:64] == notes[:8]))
    transitions = list(zip(notes[:-1].tolist(), notes[1:].tolist()))
    repetition = 1.0 - len(set(transitions)) / max(1, len(transitions))
    harmonic_fit = []
    for step, note in enumerate(notes):
        harmonic_fit.append(int(note) in CHORDS[PROGRESSION[min(7, step // MOTIF)]])
    return {
        "self_motif_return": motif_similarity,
        "self_final_return": final_similarity,
        "ends_on_tonic": float(notes[-1] == 0),
        "harmonic_fit": float(np.mean(harmonic_fit)),
        "unique_degrees": int(len(set(notes.tolist()))),
        "bigram_repetition_ratio": float(repetition),
    }


def variable_length(value):
    buffer = [value & 0x7F]
    value >>= 7
    while value:
        buffer.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(buffer))


def write_midi(path, degrees, root=60, bpm=104):
    """Write a minimal standards-compliant type-0 MIDI file."""
    ticks = 480
    step_ticks = ticks // 2
    tempo = round(60_000_000 / bpm)
    track = bytearray()
    track.extend(b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big"))
    track.extend(b"\x00\xc0\x00")
    previous_pitch = None
    for degree in degrees:
        pitch = root + SCALE[int(degree)]
        if previous_pitch is not None:
            track.extend(b"\x00\x80" + bytes((previous_pitch, 48)))
        track.extend((variable_length(0 if previous_pitch is not None else 0)))
        track.extend(b"\x90" + bytes((pitch, 82)))
        track.extend(variable_length(step_ticks))
        previous_pitch = pitch
    if previous_pitch is not None:
        track.extend(b"\x80" + bytes((previous_pitch, 48)))
    track.extend(b"\x00\xff\x2f\x00")
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks)
    path.write_bytes(header + b"MTrk" + struct.pack(">I", len(track)) + track)


def write_wav(path, degrees, root=60, bpm=104, sample_rate=22_050):
    """Render a simple melody and quiet chord bed for immediate listening."""
    step_seconds = 30.0 / bpm
    samples_per_step = int(round(step_seconds * sample_rate))
    rendered = []
    for step, degree in enumerate(degrees):
        time = np.arange(samples_per_step, dtype=np.float32) / sample_rate
        pitch = root + SCALE[int(degree)]
        frequency = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
        attack = np.clip(time / 0.025, 0.0, 1.0)
        release = np.clip((step_seconds - time) / 0.075, 0.0, 1.0)
        envelope = np.minimum(attack, release)
        melody = 0.30 * np.sin(2.0 * np.pi * frequency * time)
        melody += 0.08 * np.sin(4.0 * np.pi * frequency * time)
        chord = CHORDS[PROGRESSION[min(7, step // MOTIF)]]
        bed = np.zeros_like(time)
        for chord_degree in chord:
            chord_pitch = root - 12 + SCALE[chord_degree]
            chord_frequency = 440.0 * 2.0 ** ((chord_pitch - 69) / 12.0)
            bed += 0.035 * np.sin(2.0 * np.pi * chord_frequency * time)
        rendered.append((melody * envelope + bed * 0.65).astype(np.float32))
    audio = np.concatenate(rendered)
    peak = max(1e-6, float(np.max(np.abs(audio))))
    pcm = np.asarray(np.clip(audio / peak * 0.88, -1.0, 1.0) * 32767, dtype="<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def aggregate(rows):
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def plot_results(results):
    names = [condition.name for condition in CONDITIONS]
    accuracy = [results[name]["evaluation"]["next_note_accuracy"] for name in names]
    delayed = [results[name]["evaluation"]["delayed_motif_accuracy"] for name in names]
    reset = [results[name].get("memory_reset", {}).get("next_note_accuracy", np.nan) for name in names]
    colors = ["#606C76", "#4C956C", "#D9822B", "#1976A3"]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width, accuracy, width, label="all held-out notes", color=colors, alpha=0.65)
    ax.bar(x, delayed, width, label="delayed motif return", color=colors)
    ax.bar(x + width, reset, width, label="hidden reset", color="#B9C0C5")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x, [name.replace("_", "\n") for name in names])
    ax.set_title("Feedforward, Recurrence, and Grounded Valence on Held-Out Motifs")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)


def run(args):
    OUT.mkdir(exist_ok=True)
    MIDI_OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    train_rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 100_000)
    train_phrases = [make_phrase(train_rng) for _ in range(args.train_sequences)]
    eval_phrases = [make_phrase(eval_rng) for _ in range(args.eval_sequences)]
    results = {
        "design": {
            "train_sequences": args.train_sequences,
            "held_out_sequences": args.eval_sequences,
            "seeds": args.seeds,
            "context_steps_for_feedforward": CONTEXT,
            "sequence_steps": STEPS,
            "grounded_valence_definition": "previous-note harmonic fit plus V-to-I cadence resolution",
            "claim_boundary": "Tests architecture transfer to a procedural symbolic-music task; it does not transfer navigation weights or establish general intelligence.",
        }
    }
    trained = {}
    for condition in CONDITIONS:
        seed_rows = []
        reset_rows = []
        losses = []
        for run_seed in range(args.seeds):
            model, curve = train_model(condition, train_phrases, args.epochs, args.seed + 1009 * run_seed)
            row = teacher_forced_metrics(model, condition, eval_phrases)
            seed_rows.append(row)
            if condition.recurrent:
                reset_rows.append(teacher_forced_metrics(model, condition, eval_phrases, True))
            losses.append(curve[-1])
            if run_seed == 0:
                trained[condition.name] = model
        summary = aggregate(seed_rows)
        results[condition.name] = {
            "parameters": parameter_count(trained[condition.name]),
            "final_training_loss": float(np.mean(losses)),
            "evaluation": summary,
            "seed_evaluations": seed_rows,
        }
        if condition.recurrent:
            results[condition.name]["memory_reset"] = aggregate(reset_rows)

        generations = []
        for sample_index in range(args.samples):
            notes = generate(trained[condition.name], condition, seed=args.seed + sample_index * 31)
            generations.append(generation_metrics(notes))
            write_midi(MIDI_OUT / f"{condition.name}_{sample_index + 1}.mid", notes, root=60 + sample_index * 2)
            if sample_index == 0:
                write_wav(MIDI_OUT / f"{condition.name}_preview.wav", notes)
        results[condition.name]["generation"] = aggregate(generations)

        torch.save(
            {
                "state_dict": trained[condition.name].state_dict(),
                "condition": condition.name,
                "feature_dim": FEATURE_DIM,
                "degrees": DEGREES,
                "sequence_steps": STEPS,
            },
            CHECKPOINT_DIR / f"{condition.name}.pt",
        )

    METRICS_PATH.write_text(json.dumps(results, indent=2))
    plot_results(results)
    print("MIDI transfer lab complete")
    for condition in CONDITIONS:
        row = results[condition.name]
        evaluation = row["evaluation"]
        reset = row.get("memory_reset", {})
        print(
            f"{condition.name:22s} params={row['parameters']:6d} "
            f"accuracy={evaluation['next_note_accuracy']:.3f} "
            f"delayed={evaluation['delayed_motif_accuracy']:.3f} "
            f"reset={reset.get('next_note_accuracy', float('nan')):.3f}"
        )
    print(f"Metrics: {METRICS_PATH}")
    print(f"MIDI examples: {MIDI_OUT}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--train-sequences", type=int, default=768)
    parser.add_argument("--eval-sequences", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.train_sequences = 192
        args.eval_sequences = 64
        args.epochs = 5
        args.seeds = 1
        args.samples = 1
    return args


if __name__ == "__main__":
    run(parse_args())
