"""Passive, hunger-gated episodic memory for Unity resource regions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


@dataclass
class ResourceRegion:
    x: float
    z: float
    rewards: int = 1
    failures: int = 0

    @property
    def confidence(self):
        reward_evidence = 1.0 - math.exp(-self.rewards / 2.0)
        return reward_evidence / (1.0 + self.failures)


class PassiveTerrainResourceMemory:
    """Records rewarded places and emits causally disconnected suggestions."""

    FORMAT = "terrain_resource_memory_v2"
    LEGACY_FORMAT = "terrain_resource_memory_v1"
    TYPED_FEATURES = ("red", "blue", "yellow")

    def __init__(
        self,
        path=None,
        hunger_gate=0.70,
        cell_size=12.0,
        arrival_radius=4.0,
        max_entries=128,
        distance_penalty=0.01,
        hz=5.0,
        reward_refractory_seconds=60.0,
        stale_refractory_seconds=30.0,
        control_mode="passive",
        max_guidance_weight=0.03,
        guidance_margin_threshold=0.05,
    ):
        self.path = Path(path).expanduser().resolve() if path else None
        self.enabled = self.path is not None
        self.hunger_gate = clamp(hunger_gate)
        self.cell_size = max(1.0, float(cell_size))
        self.arrival_radius = max(0.5, float(arrival_radius))
        self.max_entries = max(1, int(max_entries))
        self.distance_penalty = max(0.0, float(distance_penalty))
        self.hz = max(0.1, float(hz))
        self.reward_refractory_frames = max(
            1, int(round(float(reward_refractory_seconds) * self.hz))
        )
        self.stale_refractory_frames = max(
            1, int(round(float(stale_refractory_seconds) * self.hz))
        )
        if control_mode not in {"passive", "guided"}:
            raise ValueError("resource-memory control must be passive or guided")
        self.control_mode = str(control_mode)
        self.max_guidance_weight = clamp(max_guidance_weight)
        self.guidance_margin_threshold = max(
            0.0, float(guidance_margin_threshold)
        )
        self.entries: dict[tuple[int, int], ResourceRegion] = {}
        self.typed_entries: dict[tuple[str, int, int], ResourceRegion] = {}
        self.suppressed_frames: dict[tuple[int, int], int] = {}
        self.typed_suppressed_frames: dict[tuple[str, int, int], int] = {}
        self.query_refractory_frames = 0
        self.last_pickup_total = None
        self.last_typed_totals = None
        self.active = False
        self.recommendation = "none"
        self.active_feature = "none"
        self.target = None
        self.guidance_vector = (0.0, 0.0)
        self.distance = 0.0
        self.confidence = 0.0
        self.encodings = 0
        self.typed_encodings = 0
        self.typed_queries = 0
        self.typed_recommendations = 0
        self.queries = 0
        self.recommendations = 0
        self.recommendation_frames = 0
        self.pickups_after_recommendation = 0
        self.counterfactual_stale_arrivals = 0
        self.typed_counterfactual_stale_arrivals = 0
        self.action_influence = 0
        self.guidance_decisions = 0
        self.guidance_action_changes = 0
        self.last_unguided_action = "none"
        self.last_guided_action = "none"
        self.last_unguided_margin = math.inf
        self.last_margin_ambiguity = 0.0
        self.last_effective_guidance_weight = 0.0
        self.release_reason = "disabled" if not self.enabled else "waiting"
        self._active_key = None
        if self.enabled:
            self.load()

    def cell(self, x, z):
        return (
            int(math.floor(float(x) / self.cell_size)),
            int(math.floor(float(z) / self.cell_size)),
        )

    def load(self):
        if self.path is None or not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("format") not in {self.FORMAT, self.LEGACY_FORMAT}:
            raise ValueError("unsupported terrain resource-memory format")
        for raw in payload.get("regions", []):
            entry = ResourceRegion(
                x=float(raw["x"]),
                z=float(raw["z"]),
                rewards=max(1, int(raw.get("rewards", 1))),
                failures=max(0, int(raw.get("failures", 0))),
            )
            self.entries[self.cell(entry.x, entry.z)] = entry
        for raw in payload.get("typed_regions", []):
            feature = str(raw.get("feature", ""))
            if feature not in self.TYPED_FEATURES:
                raise ValueError("unsupported typed resource feature")
            entry = ResourceRegion(
                x=float(raw["x"]),
                z=float(raw["z"]),
                rewards=max(1, int(raw.get("rewards", 1))),
                failures=max(0, int(raw.get("failures", 0))),
            )
            cell_x, cell_z = self.cell(entry.x, entry.z)
            self.typed_entries[(feature, cell_x, cell_z)] = entry

    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        regions = sorted(
            self.entries.values(),
            key=lambda entry: (-entry.rewards, entry.x, entry.z),
        )[: self.max_entries]
        payload = {
            "format": self.FORMAT,
            "cell_size": self.cell_size,
            "regions": [
                {
                    "x": round(entry.x, 4),
                    "z": round(entry.z, 4),
                    "rewards": entry.rewards,
                    "failures": entry.failures,
                }
                for entry in regions
            ],
            "typed_regions": [
                {
                    "feature": feature,
                    "x": round(entry.x, 4),
                    "z": round(entry.z, 4),
                    "rewards": entry.rewards,
                    "failures": entry.failures,
                }
                for (feature, _cell_x, _cell_z), entry in sorted(
                    self.typed_entries.items(),
                    key=lambda item: (
                        item[0][0], -item[1].rewards, item[1].x, item[1].z
                    ),
                )[: self.max_entries]
            ],
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def record_reward(self, x, z, count=1):
        count = max(1, int(count))
        key = self.cell(x, z)
        entry = self.entries.get(key)
        if entry is None:
            entry = ResourceRegion(float(x), float(z), rewards=count)
            self.entries[key] = entry
        else:
            total = entry.rewards + count
            entry.x = (entry.x * entry.rewards + float(x) * count) / total
            entry.z = (entry.z * entry.rewards + float(z) * count) / total
            entry.rewards = total
            entry.failures = max(0, entry.failures - count)
        self.encodings += count
        if len(self.entries) > self.max_entries:
            weakest = min(
                self.entries,
                key=lambda candidate: (
                    self.entries[candidate].confidence,
                    self.entries[candidate].rewards,
                ),
            )
            del self.entries[weakest]
        self.save()

    def record_typed_reward(self, feature, x, z, count=1):
        feature = str(feature)
        if feature not in self.TYPED_FEATURES:
            raise ValueError("unsupported typed resource feature")
        count = max(1, int(count))
        cell_x, cell_z = self.cell(x, z)
        key = (feature, cell_x, cell_z)
        entry = self.typed_entries.get(key)
        if entry is None:
            entry = ResourceRegion(float(x), float(z), rewards=count)
            self.typed_entries[key] = entry
        else:
            total = entry.rewards + count
            entry.x = (entry.x * entry.rewards + float(x) * count) / total
            entry.z = (entry.z * entry.rewards + float(z) * count) / total
            entry.rewards = total
            entry.failures = max(0, entry.failures - count)
        self.typed_encodings += count
        if len(self.typed_entries) > self.max_entries:
            weakest = min(
                self.typed_entries,
                key=lambda candidate: (
                    self.typed_entries[candidate].confidence,
                    self.typed_entries[candidate].rewards,
                ),
            )
            del self.typed_entries[weakest]
        self.save()

    def best_region(self, x, z):
        candidates = []
        for key, entry in self.entries.items():
            if self.suppressed_frames.get(key, 0) > 0:
                continue
            distance = math.hypot(entry.x - x, entry.z - z)
            if distance < self.arrival_radius:
                continue
            score = entry.confidence - self.distance_penalty * distance
            candidates.append((score, -distance, key, entry))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))

    def best_typed_region(self, feature, x, z):
        candidates = []
        for key, entry in self.typed_entries.items():
            if key[0] != feature or self.typed_suppressed_frames.get(key, 0) > 0:
                continue
            distance = math.hypot(entry.x - x, entry.z - z)
            if distance < self.arrival_radius:
                continue
            score = entry.confidence - self.distance_penalty * distance
            candidates.append((score, -distance, key, entry))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))

    def release(self, reason):
        self.active = False
        self.recommendation = "none"
        self.active_feature = "none"
        self.target = None
        self.guidance_vector = (0.0, 0.0)
        self.distance = 0.0
        self.confidence = 0.0
        self._active_key = None
        self.release_reason = str(reason)

    def update(self, body_state, hunger, requested_feature=None):
        if not self.enabled or not isinstance(body_state, dict):
            return
        try:
            x = float(body_state.get("x", 0.0))
            z = float(body_state.get("z", 0.0))
            pickup_total = int(body_state.get("mushroom_pickups_total", 0) or 0)
            typed_totals = {
                "red": int(body_state.get("red_mushroom_pickups_total", 0) or 0),
                "blue": int(body_state.get("blue_mushroom_pickups_total", 0) or 0),
                "yellow": int(body_state.get("yellow_flower_pickups_total", 0) or 0),
            }
        except (TypeError, ValueError):
            self.release("invalid_telemetry")
            return
        self.query_refractory_frames = max(
            0, self.query_refractory_frames - 1
        )
        self.suppressed_frames = {
            key: remaining - 1
            for key, remaining in self.suppressed_frames.items()
            if remaining > 1
        }
        self.typed_suppressed_frames = {
            key: remaining - 1
            for key, remaining in self.typed_suppressed_frames.items()
            if remaining > 1
        }

        typed_gains = {feature: 0 for feature in self.TYPED_FEATURES}
        if self.last_pickup_total is None:
            self.last_pickup_total = pickup_total
            self.last_typed_totals = dict(typed_totals)
        elif pickup_total < self.last_pickup_total:
            self.last_pickup_total = pickup_total
            self.last_typed_totals = dict(typed_totals)
        else:
            prior = self.last_typed_totals or {
                feature: 0 for feature in self.TYPED_FEATURES
            }
            for feature in self.TYPED_FEATURES:
                typed_gained = max(0, typed_totals[feature] - prior.get(feature, 0))
                typed_gains[feature] = typed_gained
                if typed_gained:
                    self.record_typed_reward(feature, x, z, typed_gained)
            self.last_typed_totals = dict(typed_totals)

        if pickup_total > self.last_pickup_total:
            gained = pickup_total - self.last_pickup_total
            if self.active:
                self.pickups_after_recommendation += gained
            self.record_reward(x, z, gained)
            self.last_pickup_total = pickup_total
            self.query_refractory_frames = self.reward_refractory_frames
            self.release("reward_encoded")

        requested_feature = (
            str(requested_feature)
            if requested_feature in self.TYPED_FEATURES
            else None
        )
        if requested_feature is not None:
            if typed_gains[requested_feature] > 0:
                self.release("requested_feature_collected")
                return
            if bool(body_state.get(f"{requested_feature}_food_visible", False)):
                self.release("requested_feature_visible")
                return
            self.typed_queries += 1
            arrived = [
                (math.hypot(entry.x - x, entry.z - z), key, entry)
                for key, entry in self.typed_entries.items()
                if key[0] == requested_feature
                and self.typed_suppressed_frames.get(key, 0) <= 0
            ]
            if arrived:
                nearest_distance, nearest_key, nearest_entry = min(arrived)
                if nearest_distance < self.arrival_radius:
                    self.counterfactual_stale_arrivals += 1
                    self.typed_counterfactual_stale_arrivals += 1
                    self.typed_suppressed_frames[nearest_key] = (
                        self.stale_refractory_frames
                    )
                    nearest_entry.failures += 1
                    self.save()
            selected = self.best_typed_region(requested_feature, x, z)
            if selected is None:
                reason = (
                    "typed_arrived_without_visible_resource"
                    if arrived and nearest_distance < self.arrival_radius
                    else "no_typed_memory"
                )
                self.release(reason)
                return
            _score, _negative_distance, key, entry = selected
            dx = entry.x - x
            dz = entry.z - z
            distance = math.hypot(dx, dz)
            if not self.active or key != self._active_key:
                self.typed_recommendations += 1
                self.recommendations += 1
            self.active = True
            self.active_feature = requested_feature
            self._active_key = key
            self.recommendation = (
                f"typed_{requested_feature}_{key[1]}_{key[2]}"
            )
            self.target = (entry.x, entry.z)
            self.guidance_vector = (dx / distance, dz / distance)
            self.distance = distance
            self.confidence = entry.confidence
            self.recommendation_frames += 1
            self.release_reason = "typed_rule_recommendation"
            return

        if bool(body_state.get("food_visible", False)):
            self.release("visible_food_override")
            return
        if float(hunger) <= self.hunger_gate:
            self.release("below_hunger_gate")
            return
        if self.query_refractory_frames > 0:
            self.release("post_reward_refractory")
            return
        if not self.entries:
            self.release("no_memory")
            return

        arrived = [
            (math.hypot(entry.x - x, entry.z - z), key)
            for key, entry in self.entries.items()
            if self.suppressed_frames.get(key, 0) <= 0
        ]
        if arrived:
            nearest_distance, nearest_key = min(arrived)
            if nearest_distance < self.arrival_radius:
                self.counterfactual_stale_arrivals += 1
                self.suppressed_frames[nearest_key] = (
                    self.stale_refractory_frames
                )
                self.query_refractory_frames = max(
                    1, self.stale_refractory_frames // 3
                )
                self.release("arrived_without_visible_food")
                return

        self.queries += 1
        selected = self.best_region(x, z)
        if selected is None:
            if self.active:
                self.counterfactual_stale_arrivals += 1
            self.release("arrived_without_visible_food")
            return
        _score, _negative_distance, key, entry = selected
        dx = entry.x - x
        dz = entry.z - z
        distance = math.hypot(dx, dz)
        if not self.active or key != self._active_key:
            self.recommendations += 1
        self.active = True
        self._active_key = key
        self.recommendation = f"resource_{key[0]}_{key[1]}"
        self.target = (entry.x, entry.z)
        self.guidance_vector = (dx / distance, dz / distance)
        self.distance = distance
        self.confidence = entry.confidence
        self.recommendation_frames += 1
        self.release_reason = "passive_recommendation"
