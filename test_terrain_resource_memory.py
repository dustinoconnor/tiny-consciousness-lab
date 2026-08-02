import json

from analyze_resource_memory_unity import summarize
from terrain_resource_memory import PassiveTerrainResourceMemory


def packet(x=0.0, z=0.0, pickups=0, food_visible=False):
    return {
        "x": x,
        "z": z,
        "mushroom_pickups_total": pickups,
        "food_visible": food_visible,
    }


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
