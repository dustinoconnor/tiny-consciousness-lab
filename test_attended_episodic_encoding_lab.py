import unittest

import numpy as np

from attended_episodic_encoding_lab import (
    ATTENTION_FEATURES,
    CONDITIONS,
    MEMORY_BUDGET,
    PACKETS_PER_EPISODE,
    LearnedAttentionGate,
    make_episode,
    retrieve,
    run_benchmark,
    select_packets,
    train_gates,
)


class AttendedEpisodicEncodingLabTests(unittest.TestCase):
    def test_episode_has_fixed_packet_structure(self):
        episode = make_episode(np.random.default_rng(4), transformed=True)
        self.assertEqual(len(episode.packets), PACKETS_PER_EPISODE)
        self.assertEqual(sum(packet.useful for packet in episode.packets), 2)
        self.assertEqual(
            episode.packets[0].attention_features.shape,
            (len(ATTENTION_FEATURES),),
        )

    def test_all_conditions_receive_equal_capacity(self):
        rng = np.random.default_rng(7)
        gates = train_gates(8, episodes=80)
        episode = make_episode(rng)
        for condition in CONDITIONS:
            memory = select_packets(condition, episode, gates, rng)
            self.assertEqual(len(memory), MEMORY_BUDGET)

    def test_retrieval_rejects_distant_unknown(self):
        rng = np.random.default_rng(10)
        episode = make_episode(rng)
        useful = [packet for packet in episode.packets if packet.useful]
        recalled, _ = retrieve(useful, episode.query_rays)
        unknown, _ = retrieve(useful, episode.unknown_query)
        self.assertIsNotNone(recalled)
        self.assertIsNone(unknown)

    def test_feature_lesion_zeroes_model_input(self):
        gate = LearnedAttentionGate(disabled_features=("signed_valence",))
        episode = make_episode(np.random.default_rng(12))
        matrix = gate.vectors(episode.packets)
        valence_index = ATTENTION_FEATURES.index("signed_valence")
        self.assertTrue(np.all(matrix[:, valence_index] == 0.0))

    def test_quick_benchmark_supports_attention_advantage(self):
        payload = run_benchmark(
            seed_count=3,
            train_episodes=180,
            evaluation_episodes=80,
        )
        self.assertTrue(payload["criteria"]["equal_memory_capacity"])
        self.assertGreater(
            payload["summary"]["learned_attention"]["retrieval_success"],
            payload["summary"]["random_capacity_match"]["retrieval_success"],
        )
        self.assertGreater(
            payload["summary"]["learned_attention"]["retrieval_success"],
            payload["summary"]["attention_scrambled"]["retrieval_success"],
        )


if __name__ == "__main__":
    unittest.main()
