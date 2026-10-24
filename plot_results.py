"""
Plotting and Visualization Script for MANET Routing Benchmarks.
Generates publication-quality figures comparing DDPG, AODV, and GPSR across 10 random seeds.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_plotting_style():
    """Sets clean, publication-ready style for Matplotlib."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "figure.autolayout": True,
    })


def plot_benchmark_results(
    csv_path: str = "results/benchmark_10_seeds.csv",
    training_history_path: str = "results/training_history.json",
    output_dir: str = "plots",
):
    """
    Reads benchmark data and generates comparative figures across the 10 seeds.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_plotting_style()

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run evaluate.py first.")
        return

    df = pd.read_csv(csv_path)
    protocols = ["DDPG (Ours)", "AODV", "GPSR"]
    # Filter only available protocols
    protocols = [p for p in protocols if p in df["protocol"].unique()]
    colors = {"DDPG (Ours)": "#1f77b4", "AODV": "#ff7f0e", "GPSR": "#2ca02c"}

    # -------------------------------------------------------------
    # 1. Packet Delivery Ratio (PDR) Comparison Plot
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    pdr_means = [df[df["protocol"] == p]["pdr"].mean() for p in protocols]
    pdr_stds = [df[df["protocol"] == p]["pdr"].std() for p in protocols]
    bar_colors = [colors[p] for p in protocols]

    bars = ax1.bar(protocols, pdr_means, yerr=pdr_stds, capsize=6, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("Packet Delivery Ratio (%)")
    ax1.set_title("Mean PDR across 10 Random Seeds")
    ax1.set_ylim(0, 105)

    # Label values on bars
    for bar, mean, std in zip(bars, pdr_means, pdr_stds):
        height = bar.get_height()
        ax1.annotate(
            f"{mean:.1f}%\n(±{std:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height / 2),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white"
        )

    # Boxplot distribution
    data_pdr = [df[df["protocol"] == p]["pdr"].values for p in protocols]
    bplot = ax2.boxplot(data_pdr, tick_labels=protocols, patch_artist=True)
    for patch, p in zip(bplot["boxes"], protocols):
        patch.set_facecolor(colors[p])
        patch.set_alpha(0.7)
    # Scatter points for individual seeds
    for idx, p in enumerate(protocols):
        y = df[df["protocol"] == p]["pdr"].values
        x = np.random.normal(idx + 1, 0.04, size=len(y))
        ax2.scatter(x, y, alpha=0.9, color="darkred", s=35, zorder=3)

    ax2.set_ylabel("PDR (%)")
    ax2.set_title("PDR Distribution Across Seeds (Individual Points in Red)")
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    pdr_file = os.path.join(output_dir, "pdr_comparison.png")
    plt.savefig(pdr_file, dpi=300)
    plt.close()
    print(f"Saved: {pdr_file}")

    # -------------------------------------------------------------
    # 2. End-to-End Delay Comparison Plot
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    delay_means = [df[df["protocol"] == p]["delay_ms"].mean() for p in protocols]
    delay_stds = [df[df["protocol"] == p]["delay_ms"].std() for p in protocols]

    bars = ax1.bar(protocols, delay_means, yerr=delay_stds, capsize=6, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("End-to-End Delay (ms)")
    ax1.set_title("Mean Delay across 10 Random Seeds (Lower is Better)")

    for bar, mean, std in zip(bars, delay_means, delay_stds):
        ax1.annotate(
            f"{mean:.2f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    data_delay = [df[df["protocol"] == p]["delay_ms"].values for p in protocols]
    bplot = ax2.boxplot(data_delay, tick_labels=protocols, patch_artist=True)
    for patch, p in zip(bplot["boxes"], protocols):
        patch.set_facecolor(colors[p])
        patch.set_alpha(0.7)

    for idx, p in enumerate(protocols):
        y = df[df["protocol"] == p]["delay_ms"].values
        x = np.random.normal(idx + 1, 0.04, size=len(y))
        ax2.scatter(x, y, alpha=0.9, color="darkblue", s=35, zorder=3)

    ax2.set_ylabel("Delay (ms)")
    ax2.set_title("Delay Distribution Across 10 Seeds")

    plt.tight_layout()
    delay_file = os.path.join(output_dir, "delay_comparison.png")
    plt.savefig(delay_file, dpi=300)
    plt.close()
    print(f"Saved: {delay_file}")

    # -------------------------------------------------------------
    # 3. Energy Consumption Comparison Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    energy_means = [df[df["protocol"] == p]["energy_joules"].mean() for p in protocols]
    energy_stds = [df[df["protocol"] == p]["energy_joules"].std() for p in protocols]

    bars = ax.bar(protocols, energy_means, yerr=energy_stds, capsize=6, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Total Network Energy Consumed (Joules)")
    ax.set_title("Energy Consumption across 10 Random Seeds (Lower is Better)")

    for bar, mean in zip(bars, energy_means):
        ax.annotate(
            f"{mean:.4f} J",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    plt.tight_layout()
    energy_file = os.path.join(output_dir, "energy_comparison.png")
    plt.savefig(energy_file, dpi=300)
    plt.close()
    print(f"Saved: {energy_file}")

    # -------------------------------------------------------------
    # 4. Link Breaks & Stability Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    breaks_means = [df[df["protocol"] == p]["link_breaks"].mean() for p in protocols]
    breaks_stds = [df[df["protocol"] == p]["link_breaks"].std() for p in protocols]

    bars = ax.bar(protocols, breaks_means, yerr=breaks_stds, capsize=6, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Average Link Breaks per Episode")
    ax.set_title("Link Break Robustness across 10 Random Seeds")

    for bar, mean in zip(bars, breaks_means):
        ax.annotate(
            f"{mean:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    plt.tight_layout()
    breaks_file = os.path.join(output_dir, "link_breaks_comparison.png")
    plt.savefig(breaks_file, dpi=300)
    plt.close()
    print(f"Saved: {breaks_file}")

    # -------------------------------------------------------------
    # 5. DDPG Training Dynamics Plot
    # -------------------------------------------------------------
    if os.path.exists(training_history_path):
        with open(training_history_path, "r") as f:
            history = json.load(f)

        episodes = np.arange(1, len(history["episode_rewards"]) + 1)
        rewards = np.array(history["episode_rewards"])

        fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(14, 5))

        # Reward curve with running mean
        ax_r.plot(episodes, rewards, color="#1f77b4", alpha=0.35, label="Raw Episode Reward")
        if len(rewards) >= 5:
            # 5-episode moving average
            window = min(10, len(rewards))
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax_r.plot(episodes[window - 1:], moving_avg, color="#0d47a1", linewidth=2.2, label=f"{window}-Ep Moving Avg")

        ax_r.set_xlabel("Episode")
        ax_r.set_ylabel("Episode Reward")
        ax_r.set_title("DDPG Training Reward Progression")
        ax_r.legend()

        # Critic and Actor Loss
        if "critic_losses" in history and any(history["critic_losses"]):
            c_loss = history["critic_losses"]
            a_loss = history["actor_losses"]
            ax_l.plot(episodes[:len(c_loss)], c_loss, color="#d62728", label="Critic Loss (MSE)", alpha=0.85)
            ax_l.plot(episodes[:len(a_loss)], a_loss, color="#2ca02c", label="Actor Policy Loss", alpha=0.85)
            ax_l.set_xlabel("Episode")
            ax_l.set_ylabel("Loss")
            ax_l.set_title("Critic & Actor Loss Progression")
            ax_l.legend()

        plt.tight_layout()
        train_file = os.path.join(output_dir, "ddpg_training_curve.png")
        plt.savefig(train_file, dpi=300)
        plt.close()
        print(f"Saved: {train_file}")

    # -------------------------------------------------------------
    # 6. Overall Publication Dashboard (2x2 Grid)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) PDR
    axes[0, 0].bar(protocols, pdr_means, yerr=pdr_stds, capsize=6, color=bar_colors, alpha=0.85, edgecolor="black")
    axes[0, 0].set_ylabel("Packet Delivery Ratio (%)")
    axes[0, 0].set_title("(a) Packet Delivery Ratio (Higher is Better)", fontweight="bold")
    axes[0, 0].set_ylim(0, 105)
    for b, m in zip(axes[0, 0].patches, pdr_means):
        axes[0, 0].annotate(f"{m:.1f}%", (b.get_x() + b.get_width() / 2, m / 2), ha="center", va="center", color="white", fontweight="bold")

    # (b) Delay
    axes[0, 1].bar(protocols, delay_means, yerr=delay_stds, capsize=6, color=bar_colors, alpha=0.85, edgecolor="black")
    axes[0, 1].set_ylabel("End-to-End Delay (ms)")
    axes[0, 1].set_title("(b) End-to-End Delay (Lower is Better)", fontweight="bold")
    for b, m in zip(axes[0, 1].patches, delay_means):
        axes[0, 1].annotate(f"{m:.2f} ms", (b.get_x() + b.get_width() / 2, m), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")

    # (c) Energy
    axes[1, 0].bar(protocols, energy_means, yerr=energy_stds, capsize=6, color=bar_colors, alpha=0.85, edgecolor="black")
    axes[1, 0].set_ylabel("Energy Consumed (J)")
    axes[1, 0].set_title("(c) Network Energy Consumption (Lower is Better)", fontweight="bold")
    for b, m in zip(axes[1, 0].patches, energy_means):
        axes[1, 0].annotate(f"{m:.4f} J", (b.get_x() + b.get_width() / 2, m), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")

    # (d) Link Breaks
    axes[1, 1].bar(protocols, breaks_means, yerr=breaks_stds, capsize=6, color=bar_colors, alpha=0.85, edgecolor="black")
    axes[1, 1].set_ylabel("Average Link Breaks")
    axes[1, 1].set_title("(d) Link Break Frequency (Lower is Better)", fontweight="bold")
    for b, m in zip(axes[1, 1].patches, breaks_means):
        axes[1, 1].annotate(f"{m:.1f}", (b.get_x() + b.get_width() / 2, m), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")

    plt.suptitle("MANET Routing Performance Benchmark (10 Random Seeds, 30 Nodes, RWP 20-50 m/s)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    dashboard_file = os.path.join(output_dir, "overall_dashboard.png")
    plt.savefig(dashboard_file, dpi=300)
    plt.close()
    print(f"Saved: {dashboard_file}")


if __name__ == "__main__":
    plot_benchmark_results()
