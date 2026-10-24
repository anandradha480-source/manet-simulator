"""
Main CLI entry point for MANET Routing with DDPG, AODV, and GPSR.
Supports simulation demo, agent training, 10-seed benchmarking, and plotting.
"""

import argparse
import sys
from train import train
from evaluate import evaluate_protocols
from plot_results import plot_benchmark_results
from manet.simulator import MANETSimulator
from routing.aodv import AODVRouting
from routing.gpsr import GPSRRouting


def run_demo():
    """Runs a quick interactive simulation demonstration of MANET with AODV and GPSR."""
    print("=================================================================")
    print("      MANET Simulator Demo (30 Nodes, RWP 20-50 m/s, 250m Range) ")
    print("=================================================================")
    sim = MANETSimulator(num_nodes=30, seed=42)
    print(f"Total Nodes: {sim.num_nodes}")
    print(f"Radio Range: {sim.radio_range} m")
    print(f"Speed Range: {sim.speed_range} m/s")

    for node_id in [0, 5, 12]:
        neighbors = sim.get_neighbors(node_id)
        print(f"Node {node_id:2d} | Position: ({sim.nodes[node_id].x:.1f}, {sim.nodes[node_id].y:.1f}) | Neighbors ({len(neighbors)}): {neighbors}")

    print("\nDemonstrating 30 mobility steps with AODV and GPSR routing...")
    aodv = AODVRouting()
    gpsr = GPSRRouting()

    sim.reset(seed=42)
    aodv.reset()
    for _ in range(30):
        sim.step_mobility()
        pkt = sim.generate_packet()
        if pkt:
            aodv.route_packet(pkt, sim)
    m_aodv = sim.get_metrics()
    print(f"[AODV] Delivered: {int(m_aodv['total_delivered'])}/{int(m_aodv['total_generated'])} | PDR: {m_aodv['pdr']:.1f}% | Delay: {m_aodv['delay']*1000:.2f} ms")

    sim.reset(seed=42)
    gpsr.reset()
    for _ in range(30):
        sim.step_mobility()
        pkt = sim.generate_packet()
        if pkt:
            gpsr.route_packet(pkt, sim)
    m_gpsr = sim.get_metrics()
    print(f"[GPSR] Delivered: {int(m_gpsr['total_delivered'])}/{int(m_gpsr['total_generated'])} | PDR: {m_gpsr['pdr']:.1f}% | Delay: {m_gpsr['delay']*1000:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="MANET Simulator with DDPG, AODV, and GPSR Routing")
    parser.add_argument("--demo", action="store_true", help="Run a quick MANET simulator demo")
    parser.add_argument("--train", action="store_true", help="Train the DDPG routing agent")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=50, help="Steps per training episode")
    parser.add_argument("--eval", action="store_true", help="Evaluate protocols across 10 random seeds")
    parser.add_argument("--plot", action="store_true", help="Generate publication plots from benchmark results")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline: train -> eval -> plot")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.demo:
        run_demo()
        if len(sys.argv) == 1:
            print("\nTip: Run with --all to execute training, evaluation over 10 seeds, and plotting.")
        return

    if args.all or args.train:
        train(num_episodes=args.episodes, steps_per_episode=args.steps)

    if args.all or args.eval:
        evaluate_protocols()

    if args.all or args.plot:
        plot_benchmark_results()


if __name__ == "__main__":
    main()
