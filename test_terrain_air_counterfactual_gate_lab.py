import unittest

import numpy as np

from terrain_air_counterfactual_gate_lab import (
    ActionConditionedKnn,
    intervention_needed,
)
from terrain_air_memory_lab import TerrainPacket


def packet(action, utility, ray):
    return TerrainPacket(
        time=0.0,
        step=0,
        rays=np.full(8, ray),
        body_clearance=np.ones(8),
        hunger=0.4,
        action=action,
        features=np.zeros(6),
        utility=utility,
        useful=int(utility > 0.35),
        source_index=0,
    )


class TerrainAirCounterfactualGateTests(unittest.TestCase):
    def test_action_conditioned_model_separates_outcomes(self):
        packets = (
            [packet("up", 0.8, 0.7) for _ in range(30)]
            + [packet("left", -0.2, 0.7) for _ in range(30)]
        )
        model = ActionConditionedKnn(packets, neighbors=12)
        query = packet("up", 0.0, 0.7)
        self.assertGreater(
            model.estimate(query, "up")["mean"],
            model.estimate(query, "left")["mean"],
        )

    def test_necessity_requires_grounded_wedge_trap_or_orbit(self):
        self.assertFalse(intervention_needed({}))
        self.assertTrue(intervention_needed({"physics_wedge_seconds": 1.1}))
        self.assertTrue(
            intervention_needed({"orbit_path": 5.0, "orbit_efficiency": 0.2})
        )
        self.assertFalse(
            intervention_needed({"orbit_path": 5.0, "orbit_efficiency": 0.8})
        )


if __name__ == "__main__":
    unittest.main()
