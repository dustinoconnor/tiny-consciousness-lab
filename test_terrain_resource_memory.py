import json
import tempfile
import unittest
from pathlib import Path

from analyze_resource_memory_unity import summarize
from terrain_resource_memory import PassiveTerrainResourceMemory


def packet(
    x=0.0,
    z=0.0,
    pickups=0,
    food_visible=False,
    red=0,
    blue=0,
    yellow=0,
    **updates,
):
    body = {
        "x": x,
        "z": z,
        "mushroom_pickups_total": pickups,
        "food_visible": food_visible,
        "red_mushroom_pickups_total": red,
        "blue_mushroom_pickups_total": blue,
        "yellow_flower_pickups_total": yellow,
    }
    body.update(updates)
    return body


def test_pickup_encodes_and_persists_coarse_region(tmp_path):
    path = tmp_path / "resource_memory.json"
    memory = PassiveTerrainResourceMemory(path)
    memory.update(packet(x=10.0, z=20.0, pickups=0), hunger=0.2)
    memory.update(packet(x=10.5, z=20.5, pickups=1), hunger=0.2)

    assert memory.encodings == 1
    assert len(memory.entries) == 1
    payload = json.loads(path.read_text())
    assert payload["format"] == memory.FORMAT
    assert len(payload["regions"]) == 1


def test_pickups_are_persisted_as_separate_typed_locations(tmp_path):
    path = tmp_path / "resource_memory.json"
    memory = PassiveTerrainResourceMemory(path)
    memory.update(packet(), hunger=0.2)
    memory.update(
        packet(x=10.0, z=10.0, pickups=1, red=1), hunger=0.2
    )
    memory.update(
        packet(x=10.0, z=10.0, pickups=2, red=1, yellow=1),
        hunger=0.2,
    )

    assert memory.typed_encodings == 2
    assert {key[0] for key in memory.typed_entries} == {"red", "yellow"}
    payload = json.loads(path.read_text())
    assert payload["format"] == "terrain_resource_memory_v2"
    assert {item["feature"] for item in payload["typed_regions"]} == {
        "red",
        "yellow",
    }


def test_typed_recall_returns_only_requested_feature_without_hunger_gate(tmp_path):
    memory = PassiveTerrainResourceMemory(tmp_path / "memory.json")
    memory.record_typed_reward("blue", 5.0, 0.0)
    memory.record_typed_reward("yellow", 30.0, 0.0)

    memory.update(packet(), hunger=0.1, requested_feature="yellow")

    assert memory.active
    assert memory.active_feature == "yellow"
    assert memory.target == (30.0, 0.0)
    assert memory.recommendation.startswith("typed_yellow_")
    assert memory.typed_queries == 1
    assert memory.typed_recommendations == 1


def test_visible_requested_feature_releases_typed_recall(tmp_path):
    memory = PassiveTerrainResourceMemory(tmp_path / "memory.json")
    memory.record_typed_reward("yellow", 30.0, 0.0)

    memory.update(
        packet(yellow_food_visible=True),
        hunger=0.9,
        requested_feature="yellow",
    )

    assert not memory.active
    assert memory.release_reason == "requested_feature_visible"


class TypedStaleMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "memory.json"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_arrival_suppresses_missing_resource_and_records_failure(self):
        memory = PassiveTerrainResourceMemory(
            self.path,
            hz=1.0,
            stale_refractory_seconds=3.0,
        )
        memory.record_typed_reward("yellow", 30.0, 30.0)

        memory.update(
            packet(x=30.5, z=30.5),
            hunger=0.1,
            requested_feature="yellow",
        )

        key = ("yellow", 2, 2)
        self.assertFalse(memory.active)
        self.assertEqual(
            memory.release_reason,
            "typed_arrived_without_visible_resource",
        )
        self.assertEqual(memory.typed_counterfactual_stale_arrivals, 1)
        self.assertEqual(memory.counterfactual_stale_arrivals, 1)
        self.assertEqual(memory.typed_suppressed_frames[key], 3)
        self.assertEqual(memory.typed_entries[key].failures, 1)
        self.assertEqual(
            json.loads(self.path.read_text())["typed_regions"][0]["failures"],
            1,
        )

    def test_stale_arrival_falls_through_to_next_matching_memory(self):
        memory = PassiveTerrainResourceMemory(self.path)
        memory.record_typed_reward("yellow", 30.0, 30.0)
        memory.record_typed_reward("yellow", 60.0, 30.0)

        memory.update(
            packet(x=30.5, z=30.5),
            hunger=0.1,
            requested_feature="yellow",
        )

        self.assertTrue(memory.active)
        self.assertEqual(memory.active_feature, "yellow")
        self.assertEqual(memory.target, (60.0, 30.0))
        self.assertEqual(memory.typed_counterfactual_stale_arrivals, 1)

    def test_pickup_does_not_mark_freshly_collected_location_stale(self):
        memory = PassiveTerrainResourceMemory(self.path)
        memory.record_typed_reward("yellow", 30.0, 30.0)
        memory.update(
            packet(x=20.0, z=20.0, pickups=0, yellow=0),
            hunger=0.1,
            requested_feature="yellow",
        )

        memory.update(
            packet(x=30.0, z=30.0, pickups=1, yellow=1),
            hunger=0.1,
            requested_feature="yellow",
        )

        self.assertFalse(memory.active)
        self.assertEqual(memory.release_reason, "requested_feature_collected")
        self.assertEqual(memory.typed_counterfactual_stale_arrivals, 0)
        self.assertTrue(
            all(entry.failures == 0 for entry in memory.typed_entries.values())
        )


def test_legacy_untyped_memory_remains_loadable(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "format": "terrain_resource_memory_v1",
                "regions": [
                    {"x": 12.0, "z": 18.0, "rewards": 2, "failures": 0}
                ],
            }
        )
    )

    memory = PassiveTerrainResourceMemory(path)

    assert len(memory.entries) == 1
    assert memory.typed_entries == {}


def test_recall_requires_hunger_and_absent_visible_food(tmp_path):
    memory = PassiveTerrainResourceMemory(tmp_path / "memory.json")
    memory.record_reward(30.0, 30.0)

    memory.update(packet(x=5.0, z=5.0), hunger=0.54)
    assert not memory.active
    assert memory.release_reason == "below_hunger_gate"

    memory.update(
        packet(x=5.0, z=5.0, food_visible=True),
        hunger=0.90,
    )
    assert not memory.active
    assert memory.release_reason == "visible_food_override"

    memory.update(packet(x=5.0, z=5.0), hunger=0.90)
    assert memory.active
    assert memory.recommendation != "none"


def test_passive_recommendation_has_no_action_influence(tmp_path):
    memory = PassiveTerrainResourceMemory(tmp_path / "memory.json")
    memory.record_reward(30.0, 30.0)

    memory.update(packet(x=5.0, z=5.0), hunger=0.90)

    assert memory.guidance_vector[0] > 0.0
    assert memory.guidance_vector[1] > 0.0
    assert memory.action_influence == 0


def test_pickup_after_recommendation_is_counted(tmp_path):
    memory = PassiveTerrainResourceMemory(tmp_path / "memory.json")
    memory.record_reward(30.0, 30.0)
    memory.update(packet(x=5.0, z=5.0, pickups=0), hunger=0.90)

    memory.update(packet(x=29.0, z=29.0, pickups=1), hunger=0.20)

    assert memory.pickups_after_recommendation == 1
    assert memory.encodings == 2
    assert not memory.active


def test_pickup_starts_post_reward_refractory(tmp_path):
    memory = PassiveTerrainResourceMemory(
        tmp_path / "memory.json",
        hz=1.0,
        reward_refractory_seconds=3.0,
    )
    memory.update(packet(x=30.0, z=30.0, pickups=0), hunger=0.2)
    memory.update(packet(x=30.0, z=30.0, pickups=1), hunger=0.9)
    memory.update(packet(x=5.0, z=5.0, pickups=1), hunger=0.9)

    assert not memory.active
    assert memory.release_reason == "post_reward_refractory"


def test_empty_arrival_temporarily_suppresses_region(tmp_path):
    memory = PassiveTerrainResourceMemory(
        tmp_path / "memory.json",
        hz=1.0,
        stale_refractory_seconds=3.0,
    )
    memory.record_reward(30.0, 30.0)

    memory.update(packet(x=30.5, z=30.5), hunger=0.9)

    assert memory.counterfactual_stale_arrivals == 1
    assert not memory.active
    assert memory.release_reason == "arrived_without_visible_food"


def test_passive_analyzer_verifies_causal_separation():
    rows = [
        {
            "time": 0.0,
            "resource_memory_enabled": True,
            "resource_memory_active": False,
            "resource_memory_regions": 0,
            "resource_memory_encodings": 0,
            "resource_memory_queries": 0,
            "resource_memory_recommendations": 0,
            "resource_memory_pickups_after_recommendation": 0,
            "resource_memory_stale_arrivals": 0,
            "resource_memory_action_influence": 0,
            "resource_memory_release_reason": "waiting",
            "mushroom_pickups_total": 0,
            "hunger": 0.2,
            "food_visible": False,
        },
        {
            "time": 1.0,
            "resource_memory_enabled": True,
            "resource_memory_active": True,
            "resource_memory_regions": 1,
            "resource_memory_encodings": 1,
            "resource_memory_queries": 1,
            "resource_memory_recommendations": 1,
            "resource_memory_pickups_after_recommendation": 0,
            "resource_memory_stale_arrivals": 0,
            "resource_memory_action_influence": 0,
            "resource_memory_release_reason": "passive_recommendation",
            "mushroom_pickups_total": 1,
            "hunger": 0.8,
            "food_visible": False,
        },
    ]

    result = summarize(rows)

    assert result["checks"]["passive_integration_valid"]
    assert result["recommendation_transitions_observed"] == 1
