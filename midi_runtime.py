"""Small NumPy runtime for the learned MIDI pitch and rhythm policies."""

from pathlib import Path

import numpy as np


DEGREES = 7
STEPS = 64
MOTIF = 8
CHORDS = ((0, 2, 4), (3, 5, 0), (4, 6, 1))
PROGRESSION = (0, 0, 1, 2, 0, 1, 2, 0)
SECTION = (0, 1, 2, 2, 0, 1, 3, 0)
FEATURE_DIM = DEGREES + 8 + 3 + 4 + 1
RHYTHM_OBS_DIM = 8 + 4 + 3 + 4 + 1 + 7 + 1
DURATION_BEATS = (0.25, 0.5, 1.0, 2.0)


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


class NumpyGRUPolicy:
    """Inference-compatible implementation of a PyTorch GRUCell plus linear head."""

    def __init__(self, checkpoint):
        payload = np.load(Path(checkpoint))
        self.weight_ih = payload["weight_ih"]
        self.weight_hh = payload["weight_hh"]
        self.bias_ih = payload["bias_ih"]
        self.bias_hh = payload["bias_hh"]
        self.head_weight = payload["head_weight"]
        self.head_bias = payload["head_bias"]
        self.hidden_size = self.weight_hh.shape[1]

    def initial_state(self):
        return np.zeros(self.hidden_size, dtype=np.float32)

    def step(self, observation, hidden):
        observation = np.asarray(observation, dtype=np.float32)
        hidden = np.asarray(hidden, dtype=np.float32)
        input_gates = self.weight_ih @ observation + self.bias_ih
        hidden_gates = self.weight_hh @ hidden + self.bias_hh
        input_reset, input_update, input_new = np.split(input_gates, 3)
        hidden_reset, hidden_update, hidden_new = np.split(hidden_gates, 3)
        reset = _sigmoid(input_reset + hidden_reset)
        update = _sigmoid(input_update + hidden_update)
        candidate = np.tanh(input_new + reset * hidden_new)
        next_hidden = (1.0 - update) * candidate + update * hidden
        logits = self.head_weight @ next_hidden + self.head_bias
        return logits, next_hidden.astype(np.float32, copy=False)


def grounded_valence(previous_note, step):
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


def feature(previous_note, step, valence_enabled=True):
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)
    vector[int(previous_note)] = 1.0
    offset = DEGREES
    vector[offset + step % MOTIF] = 1.0
    offset += 8
    bar = min(7, step // MOTIF)
    vector[offset + PROGRESSION[bar]] = 1.0
    offset += 3
    vector[offset + SECTION[bar]] = 1.0
    if valence_enabled:
        vector[-1] = grounded_valence(previous_note, step)
    return vector


def rhythm_observation(step, pitch_token, confidence, previous_action, bar_elapsed):
    observation = np.zeros(RHYTHM_OBS_DIM, dtype=np.float32)
    offset = 0
    observation[offset + step % MOTIF] = 1.0
    offset += 8
    observation[offset + SECTION[min(7, step // MOTIF)]] = 1.0
    offset += 4
    observation[offset + PROGRESSION[min(7, step // MOTIF)]] = 1.0
    offset += 3
    observation[offset + int(previous_action)] = 1.0
    offset += 4
    observation[offset] = float(np.clip(bar_elapsed / 4.0, 0.0, 1.5))
    offset += 1
    observation[offset + int(pitch_token)] = 1.0
    observation[-1] = float(confidence)
    return observation


def softmax(logits, temperature=1.0):
    scaled = np.asarray(logits, dtype=np.float64) / max(0.05, float(temperature))
    scaled -= scaled.max()
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum()
