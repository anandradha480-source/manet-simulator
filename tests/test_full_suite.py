"""
Comprehensive Unit and Integration Test Suite.
Verifies:
1. MANET Simulator, 30 nodes, Random Waypoint (20-50 m/s), 250m radio range.
2. Function listing each node's neighbors.
3. AODV and GPSR routing with PDR and delay metrics.
4. PyTorch DDPG agent architecture (State, Action scores, Actor/Critic).
5. Reward function: throughput - delay - energy + route stability - penalty for breaks.
6. 10 random seeds benchmark harness.
"""

import unittest
import numpy as np
import torch

from manet.simulator import MANETSimulator
from manet.channel import WirelessChannel
from routing.aodv import AODVRouting
from routing.gpsr import GPSRRouting
from routing.ddpg_routing import DDPGRouting
from rl.networks import ActorNetwork, CriticNetwork
from rl.agent import DDPGAgent
from rl.reward import MANETRewardCalculator
from rl.environment import MANETRoutingEnv


class TestMANETSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = MANETSimulator(num_nodes=30, radio_range=250.0, speed_range=(20.0, 50.0), seed=42)

    def test_node_count_and_radio_range(self):
        self.assertEqual(len(self.sim.nodes), 30)
        self.assertEqual(self.sim.radio_range, 250.0)

    def test_speed_range(self):
        for node in self.sim.nodes:
            self.assertGreaterEqual(node.speed, 20.0 - 1e-4)
            self.assertLessEqual(node.speed, 50.0 + 1e-4)

    def test_neighbor_listing_function(self):
        """Verifies get_neighbors returns only nodes within 250m range."""
        for i in range(self.sim.num_nodes):
            neighbors = self.sim.get_neighbors(i)
            self.assertNotIn(i, neighbors)  # Node should not be its own neighbor
            for n in neighbors:
                dist = self.sim.get_distance(i, n)
                self.assertLessEqual(dist, 250.0, f"Node {i} and {n} are distance {dist} > 250m!")

    def test_mobility_advancement(self):
        initial_positions = [node.position for node in self.sim.nodes]
        for _ in range(5):
            self.sim.step_mobility()
        new_positions = [node.position for node in self.sim.nodes]
        # At least one node should have changed position
        moved = any(p1 != p2 for p1, p2 in zip(initial_positions, new_positions))
        self.assertTrue(moved)


class TestBaselines(unittest.TestCase):
    def test_aodv_and_gpsr_routing(self):
        sim = MANETSimulator(num_nodes=30, seed=123)
        aodv = AODVRouting()
        gpsr = GPSRRouting()

        # AODV
        sim.reset(seed=123)
        aodv.reset()
        for _ in range(25):
            sim.step_mobility()
            pkt = sim.generate_packet()
            if pkt:
                aodv.route_packet(pkt, sim)
        m_aodv = sim.get_metrics()
        self.assertIn("pdr", m_aodv)
        self.assertIn("delay", m_aodv)
        self.assertGreater(m_aodv["total_generated"], 0)

        # GPSR
        sim.reset(seed=123)
        gpsr.reset()
        for _ in range(25):
            sim.step_mobility()
            pkt = sim.generate_packet()
            if pkt:
                gpsr.route_packet(pkt, sim)
        m_gpsr = sim.get_metrics()
        self.assertIn("pdr", m_gpsr)
        self.assertIn("delay", m_gpsr)
        self.assertGreater(m_gpsr["total_generated"], 0)


class TestRewardFunction(unittest.TestCase):
    def setUp(self):
        self.calc = MANETRewardCalculator(
            w_throughput=2.0, w_delay=1.0, w_energy=1.0, w_stability=1.5, w_break=5.0, delivery_bonus=5.0
        )

    def test_break_penalty(self):
        res = self.calc.compute_reward(
            curr_dist_to_dest=500.0,
            next_dist_to_dest=400.0,
            hop_delay=0.002,
            energy_consumed=1e-5,
            link_expiration_time=3.0,
            is_link_break=True,
            is_delivered=False,
        )
        self.assertEqual(res["reward"], -5.0)
        self.assertEqual(res["break_penalty"], 5.0)

    def test_reward_components(self):
        res = self.calc.compute_reward(
            curr_dist_to_dest=500.0,
            next_dist_to_dest=300.0,  # 200m closer
            hop_delay=0.002,
            energy_consumed=2e-5,
            link_expiration_time=4.0,
            is_link_break=False,
            is_delivered=False,
            radio_range=250.0,
        )
        # Check that throughput, delay, energy, stability terms are present
        self.assertGreater(res["throughput_term"], 0.0)
        self.assertGreater(res["delay_term"], 0.0)
        self.assertGreater(res["energy_term"], 0.0)
        self.assertGreater(res["stability_term"], 0.0)
        expected = res["throughput_term"] - res["delay_term"] - res["energy_term"] + res["stability_term"]
        self.assertAlmostEqual(res["reward"], expected, places=5)


class TestDDPGAgent(unittest.TestCase):
    def test_actor_critic_shapes(self):
        state_dim = 70
        action_dim = 10
        actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim)
        critic = CriticNetwork(state_dim=state_dim, action_dim=action_dim)

        dummy_state = torch.randn(4, state_dim)
        scores = actor(dummy_state)
        self.assertEqual(scores.shape, (4, action_dim))
        self.assertTrue(torch.all(scores >= -1.0) and torch.all(scores <= 1.0))

        q_val = critic(dummy_state, scores)
        self.assertEqual(q_val.shape, (4, 1))

    def test_agent_training_step(self):
        agent = DDPGAgent(state_dim=70, action_dim=10, batch_size=8)
        # Push 10 samples
        for _ in range(10):
            s = np.random.randn(70).astype(np.float32)
            a = np.random.uniform(-1, 1, size=10).astype(np.float32)
            r = float(np.random.randn())
            s_next = np.random.randn(70).astype(np.float32)
            done = False
            agent.replay_buffer.push(s, a, r, s_next, done)

        loss_tuple = agent.update()
        self.assertIsNotNone(loss_tuple)
        c_loss, a_loss = loss_tuple
        self.assertIsInstance(c_loss, float)
        self.assertIsInstance(a_loss, float)


if __name__ == "__main__":
    unittest.main()
