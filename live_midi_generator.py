#!/usr/bin/env python3
"""Live stochastic recurrent MIDI generator for the macOS IAC bus."""

import argparse
import queue
import signal
import threading
import time
import traceback
from collections import Counter, deque
from pathlib import Path

import mido
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk

from midi_transfer_lab import DEGREES, MOTIF, RecurrentPolicy, STEPS, feature
from midi_rhythm_learning_lab import RhythmPolicy, rhythm_observation


CHECKPOINT = Path("checkpoints/midi_transfer/recurrent_valence.pt")
RHYTHM_CHECKPOINT = Path("checkpoints/midi_transfer/learned_rhythm.pt")
ROOTS = {
    "C": 0,
    "C# / Db": 1,
    "D": 2,
    "D# / Eb": 3,
    "E": 4,
    "F": 5,
    "F# / Gb": 6,
    "G": 7,
    "G# / Ab": 8,
    "A": 9,
    "A# / Bb": 10,
    "B": 11,
}
SCALES = {
    "Minor pentatonic": (0, 3, 5, 7, 10),
    "Major pentatonic": (0, 2, 4, 7, 9),
    "Natural minor": (0, 2, 3, 5, 7, 8, 10),
    "Major": (0, 2, 4, 5, 7, 9, 11),
    "Dorian": (0, 2, 3, 5, 7, 9, 10),
    "Phrygian": (0, 1, 3, 5, 7, 8, 10),
    "Mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "Whole tone": (0, 2, 4, 6, 8, 10),
    "Chromatic": tuple(range(12)),
}
RHYTHM_MODES = ("Learned rhythm", "Stochastic phrase", "Random", "1/2", "1/4", "1/8", "1/16")
FIXED_BEATS = {"1/2": 2.0, "1/4": 1.0, "1/8": 0.5, "1/16": 0.25}
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class LiveGenerator:
    def __init__(self, checkpoint, rhythm_checkpoint=RHYTHM_CHECKPOINT):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("condition") != "recurrent_valence":
            raise ValueError(f"Expected recurrent_valence checkpoint, found {payload.get('condition')}")
        self.model = RecurrentPolicy().eval()
        self.model.load_state_dict(payload["state_dict"])
        rhythm_payload = torch.load(rhythm_checkpoint, map_location="cpu", weights_only=False)
        self.rhythm_policy = RhythmPolicy(hidden=int(rhythm_payload["hidden_size"])).eval()
        self.rhythm_policy.load_state_dict(rhythm_payload["state_dict"])
        self.hidden = torch.zeros(1, self.model.hidden)
        self.rhythm_hidden = self.rhythm_policy.initial_state()
        self.rhythm_previous_action = 1
        self.rhythm_elapsed = 0.0
        self.previous_token = 0
        self.step = 0
        self.recent = deque(maxlen=16)
        self.motif = []
        self.rhythm_motif = []
        self.last_duration = 0.5
        self.rng = np.random.default_rng()

    def reset_memory(self):
        self.hidden.zero_()
        self.previous_token = 0
        self.step = 0
        self.recent.clear()
        self.motif.clear()
        self.rhythm_motif.clear()
        self.last_duration = 0.5
        self.rhythm_hidden = self.rhythm_policy.initial_state()
        self.rhythm_previous_action = 1
        self.rhythm_elapsed = 0.0

    def next_token(self, temperature, novelty, motif_memory):
        phrase_step = self.step % STEPS
        current = torch.tensor(feature(self.previous_token, phrase_step, True)).view(1, -1)
        with torch.no_grad():
            self.hidden = self.model.cell(current, self.hidden)
            logits = self.model.head(self.hidden)[0].clone()

        counts = Counter(self.recent)
        for token, count in counts.items():
            logits[token] -= float(novelty) * count / max(1, len(self.recent)) * 4.0

        motif_position = phrase_step % MOTIF
        if phrase_step in range(32, 40) or phrase_step in range(56, 64):
            if motif_position < len(self.motif):
                logits[self.motif[motif_position]] += float(motif_memory) * 2.5

        probabilities = torch.softmax(logits / max(0.05, float(temperature)), dim=-1)
        token = int(torch.multinomial(probabilities, 1))
        if phrase_step < MOTIF:
            if phrase_step == 0:
                self.motif.clear()
            self.motif.append(token)
        self.previous_token = token
        self.recent.append(token)
        self.step += 1
        return token, float(probabilities[token])

    def next_duration(self, mode, confidence, motif_memory):
        phrase_step = (self.step - 1) % STEPS
        if mode == "Learned rhythm":
            obs = rhythm_observation(
                phrase_step,
                self.previous_token,
                confidence,
                self.rhythm_previous_action,
                self.rhythm_elapsed,
            )
            with torch.no_grad():
                logits, self.rhythm_hidden = self.rhythm_policy.step(torch.tensor(obs).view(1, -1), self.rhythm_hidden)
                action = int(torch.distributions.Categorical(logits=logits[0]).sample())
            duration = float((0.25, 0.5, 1.0, 2.0)[action])
            self.rhythm_previous_action = action
            self.rhythm_elapsed += duration
            if (phrase_step + 1) % MOTIF == 0:
                self.rhythm_elapsed = 0.0
        elif mode in FIXED_BEATS:
            duration = FIXED_BEATS[mode]
        elif mode == "Random":
            duration = float(self.rng.choice((0.25, 0.5, 1.0, 2.0), p=(0.30, 0.35, 0.25, 0.10)))
        else:
            motif_position = phrase_step % MOTIF
            recall_probability = float(np.clip(0.20 + 0.42 * motif_memory, 0.0, 0.92))
            returning = phrase_step in range(32, 40) or phrase_step in range(56, 64)
            if returning and motif_position < len(self.rhythm_motif) and self.rng.random() < recall_probability:
                duration = self.rhythm_motif[motif_position]
            elif self.rng.random() < 0.24:
                duration = self.last_duration
            else:
                weights = np.asarray((0.18, 0.52, 0.24, 0.06), dtype=np.float64)
                if confidence < 0.25:
                    weights += np.asarray((0.10, 0.04, -0.08, -0.06))
                elif confidence > 0.60:
                    weights += np.asarray((-0.06, -0.06, 0.08, 0.04))
                if motif_position == MOTIF - 1:
                    weights += np.asarray((-0.12, -0.12, 0.14, 0.10))
                weights = np.clip(weights, 0.01, None)
                weights /= weights.sum()
                duration = float(self.rng.choice((0.25, 0.5, 1.0, 2.0), p=weights))
            if phrase_step < MOTIF:
                if phrase_step == 0:
                    self.rhythm_motif.clear()
                self.rhythm_motif.append(duration)
        self.last_duration = duration
        return duration


class MidiApp:
    def __init__(self, root, checkpoint, rhythm_checkpoint=RHYTHM_CHECKPOINT):
        self.root = root
        self.root.title("Recurrent MIDI Lab")
        self.root.resizable(False, False)
        self.generator = LiveGenerator(checkpoint, rhythm_checkpoint)
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.settings_lock = threading.Lock()
        self.generator_lock = threading.Lock()
        self.midi_lock = threading.Lock()
        self.worker = None
        self.output = None
        self.closing = False

        ports = mido.get_output_names()
        preferred = next((name for name in ports if "IAC" in name), ports[0] if ports else "")
        self.port = tk.StringVar(value=preferred)
        self.root_note = tk.StringVar(value="A")
        self.scale = tk.StringVar(value="Minor pentatonic")
        self.octave = tk.IntVar(value=4)
        self.bpm = tk.DoubleVar(value=120)
        self.velocity = tk.IntVar(value=111)
        self.channel = tk.IntVar(value=1)
        self.temperature = tk.DoubleVar(value=0.85)
        self.novelty = tk.DoubleVar(value=0.65)
        self.motif_memory = tk.DoubleVar(value=0.55)
        self.rhythm_mode = tk.StringVar(value="Stochastic phrase")
        self.gate_variation = tk.DoubleVar(value=0.45)
        self.chords = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Stopped")
        self.note_status = tk.StringVar(value="Note: --")
        self.current_settings = {}

        frame = ttk.Frame(root, padding=14)
        frame.grid()
        self._combo(frame, 0, "MIDI output", self.port, ports, width=28)
        self._combo(frame, 1, "Root", self.root_note, list(ROOTS), width=15)
        self._combo(frame, 2, "Scale", self.scale, list(SCALES), width=20)
        self._spin(frame, 3, "Octave", self.octave, 1, 7, 1)
        self._spin(frame, 4, "BPM", self.bpm, 30, 240, 1)
        self._spin(frame, 5, "Velocity", self.velocity, 1, 127, 1)
        self._spin(frame, 6, "MIDI channel", self.channel, 1, 16, 1)
        self._scale(frame, 7, "Temperature", self.temperature, 0.15, 2.0)
        self._scale(frame, 8, "Novelty", self.novelty, 0.0, 1.5)
        self._scale(frame, 9, "Motif memory", self.motif_memory, 0.0, 1.5)
        self._combo(frame, 10, "Rhythm", self.rhythm_mode, RHYTHM_MODES, width=20)
        self._scale(frame, 11, "Gate variation", self.gate_variation, 0.0, 1.0)

        mode_row = ttk.Frame(frame)
        mode_row.grid(row=12, column=0, columnspan=3, pady=(8, 2), sticky="ew")
        ttk.Label(mode_row, text="Voicing").pack(side="left")
        ttk.Checkbutton(mode_row, text="Chords", variable=self.chords, command=self.chords_changed).pack(side="right")

        controls = ttk.Frame(frame)
        controls.grid(row=13, column=0, columnspan=3, pady=(14, 8), sticky="ew")
        self.start_button = ttk.Button(controls, text="Start", command=self.start)
        self.start_button.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", expand=True, fill="x", padx=5)
        ttk.Button(controls, text="Reset memory", command=self.reset_memory).pack(side="left", expand=True, fill="x", padx=5)
        ttk.Button(controls, text="Panic", command=self.panic).pack(side="left", expand=True, fill="x", padx=(5, 0))

        ttk.Separator(frame).grid(row=14, column=0, columnspan=3, pady=6, sticky="ew")
        ttk.Label(frame, textvariable=self.status).grid(row=15, column=0, columnspan=2, sticky="w")
        ttk.Label(frame, textvariable=self.note_status).grid(row=15, column=2, sticky="e")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.current_settings = self.snapshot()
        self.root.after(50, self.poll_events)

    def _combo(self, parent, row, label, variable, values, width):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        box = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=width)
        box.grid(row=row, column=1, columnspan=2, sticky="ew", pady=3)

    def _spin(self, parent, row, label, variable, low, high, increment):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Spinbox(parent, textvariable=variable, from_=low, to=high, increment=increment, width=10).grid(
            row=row, column=1, sticky="w", pady=3
        )

    def _scale(self, parent, row, label, variable, low, high):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Scale(parent, variable=variable, from_=low, to=high, orient="horizontal", length=220).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        ttk.Label(parent, textvariable=variable, width=5).grid(row=row, column=2, sticky="e")

    def snapshot(self):
        previous = self.current_settings

        def numeric(variable, key, default, cast):
            try:
                return cast(variable.get())
            except (tk.TclError, TypeError, ValueError):
                return cast(previous.get(key, default))

        try:
            channel = int(self.channel.get()) - 1
        except (tk.TclError, TypeError, ValueError):
            channel = int(previous.get("channel", 0))

        bpm = float(np.clip(numeric(self.bpm, "bpm", 120.0, float), 30.0, 240.0))

        return {
            "root": ROOTS[self.root_note.get()],
            "scale_name": self.scale.get(),
            "scale": SCALES[self.scale.get()],
            "octave": numeric(self.octave, "octave", 4, int),
            "bpm": bpm,
            "velocity": numeric(self.velocity, "velocity", 111, int),
            "channel": int(np.clip(channel, 0, 15)),
            "temperature": numeric(self.temperature, "temperature", 0.85, float),
            "novelty": numeric(self.novelty, "novelty", 0.65, float),
            "motif_memory": numeric(self.motif_memory, "motif_memory", 0.55, float),
            "rhythm_mode": self.rhythm_mode.get(),
            "gate_variation": numeric(self.gate_variation, "gate_variation", 0.45, float),
            "chords": bool(self.chords.get()),
        }

    @staticmethod
    def chord_notes(root_note, scale_name, scale, scale_index):
        if scale_name == "Minor pentatonic":
            return [root_note, root_note + 3, root_note + 7]
        if scale_name == "Major pentatonic":
            return [root_note, root_note + 4, root_note + 7]
        if scale_name == "Whole tone":
            return [root_note, root_note + 4, root_note + 8]
        if scale_name == "Chromatic":
            return [root_note, root_note + 3, root_note + 7]

        base = root_note - scale[scale_index]
        notes = []
        for offset in (0, 2, 4):
            position = scale_index + offset
            notes.append(base + scale[position % len(scale)] + 12 * (position // len(scale)))
        return notes

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.port.get():
            self.status.set("No MIDI output available")
            return
        try:
            self.output = mido.open_output(self.port.get())
        except Exception as exc:
            self.status.set(f"MIDI error: {exc}")
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self.run_notes, daemon=True)
        self.worker.start()
        self.status.set(f"Playing to {self.port.get()}")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def run_notes(self):
        active_notes = []
        active_channel = 0
        try:
            while not self.stop_event.is_set():
                with self.settings_lock:
                    settings = dict(self.current_settings)
                with self.generator_lock:
                    token, confidence = self.generator.next_token(
                        settings["temperature"], settings["novelty"], settings["motif_memory"]
                    )
                    duration_beats = self.generator.next_duration(
                        settings["rhythm_mode"], confidence, settings["motif_memory"]
                    )
                scale = settings["scale"]
                scale_index = int(round(token * (len(scale) - 1) / (DEGREES - 1)))
                midi_note = 12 * (settings["octave"] + 1) + settings["root"] + scale[scale_index]
                midi_note = int(np.clip(midi_note, 0, 127))
                notes = [midi_note]
                if settings["chords"]:
                    notes = self.chord_notes(midi_note, settings["scale_name"], scale, scale_index)
                active_notes = list(dict.fromkeys(int(np.clip(note, 0, 127)) for note in notes))
                active_channel = settings["channel"]
                chord_velocity = max(1, settings["velocity"] - (8 if len(active_notes) > 1 else 0))
                for note in active_notes:
                    self.send_midi(
                        mido.Message("note_on", note=note, velocity=chord_velocity, channel=settings["channel"])
                    )
                names = [NOTE_NAMES[note % 12] + str(note // 12 - 1) for note in active_notes]
                label = "Chord" if len(active_notes) > 1 else "Note"
                self.events.put(("note", f"{label}: {' '.join(names)}  p={confidence:.2f}"))
                variation = settings["gate_variation"]
                gate = float(np.clip(0.82 + self.generator.rng.uniform(-0.36, 0.13) * variation, 0.25, 0.96))
                self.wait_beats(duration_beats * gate)
                for note in active_notes:
                    self.send_midi(mido.Message("note_off", note=note, velocity=0, channel=settings["channel"]))
                active_notes = []
                self.wait_beats(duration_beats * (1.0 - gate))
        except Exception:
            self.events.put(("error", traceback.format_exc()))
        finally:
            if self.output is not None:
                for note in active_notes:
                    try:
                        self.send_midi(mido.Message("note_off", note=note, velocity=0, channel=active_channel))
                    except Exception:
                        pass
                try:
                    self.send_all_notes_off()
                except Exception:
                    pass
            self.events.put(("stopped", None))

    def stop(self):
        self.stop_event.set()

    def wait_beats(self, beats):
        """Wait in musical time so tempo edits take effect during the current note."""
        remaining = max(0.0, float(beats))
        previous_time = time.monotonic()
        while remaining > 0.0 and not self.stop_event.is_set():
            now = time.monotonic()
            with self.settings_lock:
                bpm = float(self.current_settings.get("bpm", 105.0))
            remaining -= (now - previous_time) * bpm / 60.0
            previous_time = now
            if remaining > 0.0:
                self.stop_event.wait(min(0.02, remaining * 60.0 / bpm))

    def send_midi(self, message):
        with self.midi_lock:
            if self.output is not None:
                self.output.send(message)

    def send_all_notes_off(self):
        with self.midi_lock:
            if self.output is not None:
                for channel in range(16):
                    self.output.send(mido.Message("control_change", control=123, value=0, channel=channel))

    def panic(self):
        self.send_all_notes_off()
        self.note_status.set("Note: --")

    def chords_changed(self):
        if not self.chords.get():
            self.panic()
            self.status.set("Chord notes released; melody mode")

    def reset_memory(self):
        with self.generator_lock:
            self.generator.reset_memory()
        self.status.set("Recurrent memory reset")

    def poll_events(self):
        with self.settings_lock:
            self.current_settings = self.snapshot()
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "note":
                    self.note_status.set(value)
                elif kind == "error":
                    final_line = value.strip().splitlines()[-1] if value.strip() else "unknown worker error"
                    self.status.set(f"Generator stopped: {final_line}")
                    print(value, flush=True)
                elif kind == "stopped":
                    self.panic()
                    if self.output is not None:
                        self.output.close()
                        self.output = None
                    if not self.status.get().startswith("Generator stopped:"):
                        self.status.set("Stopped")
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
        except queue.Empty:
            pass
        if (
            not self.closing
            and self.worker is not None
            and not self.worker.is_alive()
            and str(self.start_button.cget("state")) == "disabled"
        ):
            self.send_all_notes_off()
            if self.output is not None:
                self.output.close()
                self.output = None
            self.worker = None
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            if not self.status.get().startswith("Generator stopped:"):
                self.status.set("Stopped; ready to restart")
        self.root.after(50, self.poll_events)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.stop_event.set()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.status.set("Closing safely...")
        self.root.after(25, self.finish_close)

    def finish_close(self):
        if self.worker is not None and self.worker.is_alive():
            self.root.after(25, self.finish_close)
            return
        self.panic()
        if self.output is not None:
            self.output.close()
            self.output = None
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--rhythm-checkpoint", type=Path, default=RHYTHM_CHECKPOINT)
    parser.add_argument("--list-ports", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_ports:
        for name in mido.get_output_names():
            print(name)
        return
    if not args.checkpoint.exists():
        raise SystemExit(f"Missing {args.checkpoint}. Run: python3 midi_transfer_lab.py")
    if not args.rhythm_checkpoint.exists():
        raise SystemExit(f"Missing {args.rhythm_checkpoint}. Run: python3 midi_rhythm_learning_lab.py")
    root = tk.Tk()
    app = MidiApp(root, args.checkpoint, args.rhythm_checkpoint)

    def request_close(_signum, _frame):
        root.after(0, app.close)

    signal.signal(signal.SIGINT, request_close)
    signal.signal(signal.SIGTERM, request_close)
    root.mainloop()


if __name__ == "__main__":
    main()
