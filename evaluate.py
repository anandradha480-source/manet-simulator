"""
Evaluation benchmark: compares DDPG, AODV, and GPSR across 10 random seeds.
Measures: Packet Delivery Ratio (PDR), End-to-End Delay, Energy Consumption, and Link Breaks.
"""

import os
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch

from manet.simulator import MANETSimulator
from routing.aodv import AODVRouting
from routing.gpsr import GPSRRouting
from routing.ddpg_routing import DDPGRouting
from rl.agent import DDPGAgent


def evaluate_protocols(
    seeds: List[int] = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    num_packets_per_seed: int = 60,
    model_path: str = "models/best_ddpg.pt",
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Evaluates DDPG, AODV, and GPSR across 10 random seeds on identical MANET traffic conditions.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Initialize DDPG Agent
    state_dim = 10 * 7  # 10 candidate neighbors * 7 features
    action_dim = 10
    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim)
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded trained DDPG agent from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Running with untrained agent.")

    ddpg_router = DDPGRouting(agent=agent, max_neighbors=10)
    aodv_router = AODVRouting(route_lifetime=3.0)
    gpsr_router = GPSRRouting()

    protocols = {
        "DDPG (Ours)": ddpg_router,
        "AODV": aodv_router,
        "GPSR": gpsr_router,
    }

    records = []

    print(f"\n=== Evaluating Protocols Across {len(seeds)} Random Seeds ===")
    print(f"{'Seed':>5} | {'Protocol':<12} | {'PDR (%)':>8} | {'Delay (ms)':>10} | {'Energy (J)':>10} | {'Breaks':>7} | {'Delivered':>9}")
    print("-" * 75)

    for seed in seeds:
        for proto_name, proto in protocols.items():
            # Create simulator with current seed
            sim = MANETSimulator(num_nodes=30, seed=seed)
            proto.reset()

            # Warmup mobility so topology is well-mixed
            for _ in range(20):
                sim.step_mobility()

            # Generate and route packets
            for _ in range(num_packets_per_seed):
                sim.step_mobility()
                pkt = sim.generate_packet()
                if pkt is not None:
                    proto.route_packet(pkt, sim)

            m = sim.get_metrics()
            pdr = m["pdr"]
            delay_ms = m["delay"] * 1000.0  # convert seconds to ms
            energy = m["total_energy"]
            breaks = m["link_breaks"]
            delivered = m["total_delivered"]
            generated = m["total_generated"]

            records.append({
                "seed": seed,
                "protocol": proto_name,
                "pdr": pdr,
                "delay_ms": delay_ms,
                "energy_joules": energy,
                "link_breaks": breaks,
                "delivered": delivered,
                "generated": generated,
            })

            print(f"{seed:>5} | {proto_name:<12} | {pdr:>7.2f}% | {delay_ms:>9.2f}ms | {energy:>9.4f}J | {int(breaks):>7} | {int(delivered):>4}/{int(generated):<4}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(results_dir, "benchmark_10_seeds.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRaw benchmark results saved to: {csv_path}")

    # Compute summary statistics
    summary = df.groupby("protocol").agg({
        "pdr": ["mean", "std"],
        "delay_ms": ["mean", "std"],
        "energy_joules": ["mean", "std"],
        "link_breaks": ["mean", "std"],
    }).reset_index()

    summary_json_path = os.path.join(results_dir, "benchmark_summary.json")
    # Convert multi-level column df to clean dict
    summary_dict = {}
    for proto in df["protocol"].unique():
        proto_df = df[df["protocol"] == proto]
        summary_dict[proto] = {
            "pdr_mean": float(proto_df["pdr"].mean()),
            "pdr_std": float(proto_df["pdr"].std()),
            "delay_mean_ms": float(proto_df["delay_ms"].mean()),
            "delay_std_ms": float(proto_df["delay_ms"].std()),
            "energy_mean_j": float(proto_df["energy_joules"].mean()),
            "energy_std_j": float(proto_df["energy_joules"].std()),
            "breaks_mean": float(proto_df["link_breaks"].mean()),
            "breaks_std": float(proto_df["link_breaks"].std()),
        }

    with open(summary_json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"Benchmark summary saved to: {summary_json_path}")
    print("\n=== Aggregate Performance Summary (Mean +/- Std over 10 Seeds) ===")
    for proto, stats in summary_dict.items():
        print(f"[{proto}]")
        print(f"  PDR:          {stats['pdr_mean']:.2f}% +/- {stats['pdr_std']:.2f}%")
        print(f"  Delay:        {stats['delay_mean_ms']:.2f} ms +/- {stats['delay_std_ms']:.2f} ms")
        print(f"  Energy:       {stats['energy_mean_j']:.4f} J +/- {stats['energy_std_j']:.4f} J")
        print(f"  Link Breaks:  {stats['breaks_mean']:.2f} +/- {stats['breaks_std']:.2f}")

    return df


if __name__ == "__main__":
    evaluate_protocols()
