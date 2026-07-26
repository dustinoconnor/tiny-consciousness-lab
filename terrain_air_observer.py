"""Causally disconnected live observer for AIR-selected terrain memory."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from terrain_air_memory_lab import ACTION_INDEX, angular_action_error


class PassiveTerrainAirObserver:
    def __init__(self, checkpoint=None):
        self.enabled = checkpoint is not None
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.memory = []
        self.threshold = 0.12
        self.recalled_action = "none"
        self.distance = math.inf
        self.confidence = 0.0
        self.resonance = False
        self.agreement = False
        self.queries = 0
        self.resonances = 0
        self.agreements = 0
        self.action_influence = 0.0
        if not self.enabled:
            return
        payload = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
        if payload.get("format") != "terrain_air_memory_v1":
            raise ValueError("Unsupported terrain AIR checkpoint format")
        self.threshold = float(
            payload.get("protocol", {}).get("retrieval_threshold", 0.12)
        )
        for packet in payload.get("memory", []):
            rays = np.asarray(packet.get("rays", []), dtype=np.float64)
            clearance = np.asarray(
                packet.get("body_clearance", []), dtype=np.float64
            )
            if rays.size != 8 or clearance.size != 8:
                continue
            self.memory.append(
                {
                    "vector": np.r_[
                        np.clip(rays, 0.0, 1.0),
                        np.clip(clearance, 0.0, 1.0),
                        float(packet.get("hunger", 0.0)),
                    ],
                    "action": str(packet.get("action", "none")),
                    "utility": float(packet.get("utility", 0.0)),
                }
            )
        if not self.memory:
            raise ValueError("Terrain AIR checkpoint contains no valid packets")

    @property
    def agreement_rate(self):
        return self.agreements / max(self.resonances, 1)

    def update(self, body_state, hunger, active_direction):
        if not self.enabled:
            return
        rays = np.asarray(
            body_state.get("directional_rays", []), dtype=np.float64
        )
        clearance = np.asarray(
            body_state.get("directional_body_clearance", []), dtype=np.float64
        )
        if rays.size != 8 or clearance.size != 8:
            return
        query = np.r_[
            np.clip(rays, 0.0, 1.0),
            np.clip(clearance, 0.0, 1.0),
            max(0.0, min(1.0, float(hunger))),
        ]
        distances = [
            float(np.mean(np.abs(packet["vector"] - query)))
            for packet in self.memory
        ]
        index = int(np.argmin(distances))
        self.distance = distances[index]
        self.resonance = self.distance <= self.threshold
        self.recalled_action = (
            self.memory[index]["action"] if self.resonance else "none"
        )
        self.confidence = (
            max(0.0, 1.0 - self.distance / max(self.threshold, 1e-6))
            if self.resonance
            else 0.0
        )
        self.agreement = (
            self.resonance
            and self.recalled_action in ACTION_INDEX
            and active_direction in ACTION_INDEX
            and angular_action_error(self.recalled_action, active_direction) <= 1
        )
        self.queries += 1
        if self.resonance:
            self.resonances += 1
            self.agreements += int(self.agreement)
