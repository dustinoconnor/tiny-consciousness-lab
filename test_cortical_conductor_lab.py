import unittest

import numpy as np

from cortical_conductor_lab import (
    CONTEXTS,
    LearnedConductor,
    ProtocolEntry,
    delayed_report,
    frozen_specialist_proposals,
    protocol_context,
)


class CorticalConductorLabTests(unittest.TestCase):
    def test_protocol_context_lesion_and_intervention_are_distinct(self):
        cue = 1
        self.assertEqual(protocol_context("protocol_intact", cue, 0, 2), cue)
        self.assertEqual(
            protocol_context("protocol_lesion", cue, 0, 2), len(CONTEXTS)
        )
        self.assertNotEqual(protocol_context("protocol_scrambled", cue, 0, 2), cue)
        self.assertEqual(protocol_context("oracle_conductor", cue, 0, 2), 2)

    def test_reward_learning_prefers_successful_specialist(self):
        conductor = LearnedConductor(4)
        for _ in range(30):
            conductor.update(0, 2, 0.8)
            conductor.update(0, 0, -0.4)
        self.assertEqual(conductor.choose(0), 2)

    def test_delayed_report_reconstructs_protocol(self):
        entries = [
            ProtocolEntry(
                step=index,
                attended_context="familiar",
                selected_specialist="episodic",
                specialist_confidence=0.8,
                predicted_action=1,
                executed_action=1,
                correct_action=1,
                outcome="success",
                reward=0.9,
                valence=1.0,
            )
            for index in range(3)
        ]
        context, specialist, valence = delayed_report(
            "protocol_intact", entries, 0
        )
        self.assertEqual(context, CONTEXTS.index("familiar"))
        self.assertEqual(specialist, "episodic")
        self.assertEqual(valence, 1.0)

    def test_frozen_specialists_return_valid_actions_and_confidences(self):
        proposals, confidences = frozen_specialist_proposals(
            0, 2, np.random.default_rng(9)
        )
        self.assertEqual(proposals.shape, (3,))
        self.assertEqual(confidences.shape, (3,))
        self.assertTrue(np.all((proposals >= 0) & (proposals < 4)))
        self.assertTrue(np.all((confidences >= 0.0) & (confidences <= 1.0)))
