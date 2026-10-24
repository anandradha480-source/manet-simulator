# RL-ARHM: Deep Deterministic Policy Gradient (DDPG) Routing in MANET

A Python-based Mobile Ad-Hoc Network (MANET) simulator implementing high-mobility Random Waypoint (RWP) movement, baseline routing protocols (**AODV** and **GPSR**), and a continuous actor-critic **PyTorch DDPG agent** for adaptive next-hop selection.

---

## Key Features

1. **MANET Simulator**:
   - **30 Mobile Nodes** moving across a $1000 \times 1000\text{ m}^2$ area.
   - **Random Waypoint Mobility (RWP)** with speeds uniformly sampled between **20 m/s and 50 m/s**.
   - **Radio Range**: 250 m with quadratic wireless propagation path loss and first-order radio energy consumption ($E_{tx}, E_{rx}$).
   - **Neighbor Discovery**: Exact neighbor listing function `sim.get_neighbors(node_id)` based on instantaneous Euclidean distance $\le 250\text{ m}$.

2. **Baselines Implemented**:
   - **AODV (Ad hoc On-Demand Distance Vector)**: Reactive route discovery using RREQ flood, reverse path caching, unicast RREP forwarding, route expiration timeouts, and broken-link detection.
   - **GPSR (Greedy Perimeter Stateless Routing)**: Geographic routing with Greedy distance progress towards the destination and Gabriel Graph (GG) planar perimeter forwarding using the Right-Hand Rule to recover from local minima/voids.

3. **PyTorch DDPG Agent**:
   - **State Space**: Neighbor distance $d(u, v)/R$, speed $v/v_{max}$, link quality $LQ \in [0, 1]$, and battery ratio $E/E_{max}$ (along with destination distance and Link Expiration Time).
   - **Continuous Action**: Continuous score output in $[-1, 1]$ for each candidate neighbor slot.
   - **Decision Rule**: The candidate neighbor with the **highest score** is selected as the next hop ($\arg\max$).
   - **Critic Network**: Evaluates state-action pairs $Q(s, a)$.
   - **Training Stability**: Replay Buffer, Gaussian exploration noise, soft target updates ($\tau = 0.005$).

4. **Reward Function**:
   $$\text{Reward} = w_{tp} \cdot \text{Throughput} - w_{\text{delay}} \cdot \text{Delay} - w_{\text{energy}} \cdot \text{Energy} + w_{\text{stab}} \cdot \text{RouteStability} - w_{\text{break}} \cdot \mathbb{I}(\text{LinkBreak})$$
   - **Throughput**: Distance advancement towards destination plus final delivery bonus ($+5.0$).
   - **Delay**: Hop transmission and propagation delay.
   - **Energy**: Radio transmission and circuit power consumed.
   - **Route Stability**: Kinematic Link Expiration Time (LET) predicting how long the wireless link remains connected under current velocities.
   - **Penalty for Link Breaks**: Severe penalty ($-5.0$) if chosen next-hop breaks or is unreachable.

5. **10-Seed Comparative Benchmark**:
   - Evaluated over 10 distinct random seeds (`101` through `110`).
   - Measures: Packet Delivery Ratio (PDR %), End-to-End Delay (ms), Energy Consumption (J), and Link Breaks.
   - Publication-grade plots generated automatically.

---

## Project Structure

```
d:\RL-ARHM\
├── manet/
│   ├── __init__.py
│   ├── mobility.py          # Random Waypoint Mobility Model (20-50 m/s)
│   ├── channel.py           # Wireless Channel (250m range, LET, energy, delay)
│   ├── node.py              # Mobile Node class (battery, queue, stats)
│   ├── packet.py            # Data & Control packet definitions
│   └── simulator.py         # MANET Simulator & Neighbor discovery API
├── routing/
│   ├── __init__.py
│   ├── base.py              # Base routing abstract class
│   ├── aodv.py              # AODV protocol implementation
│   ├── gpsr.py              # GPSR protocol implementation
│   └── ddpg_routing.py      # DDPG inference wrapper as a BaseRoutingProtocol
├── rl/
│   ├── __init__.py
│   ├── networks.py          # PyTorch Actor and Critic Neural Networks
│   ├── agent.py             # DDPG Agent with Replay Buffer & Target Updates
│   ├── environment.py       # MANET RL Gym-like Environment
│   └── reward.py            # Decomposed reward calculator
├── tests/
│   ├── test_baselines.py    # Quick sanity test for AODV & GPSR
│   └── test_full_suite.py   # Full unittest suite (9 tests)
├── models/
│   ├── best_ddpg.pt         # Best trained PyTorch weights
│   └── latest_ddpg.pt       # Final training checkpoint
├── results/
│   ├── benchmark_10_seeds.csv   # Raw records for 10 seeds
│   ├── benchmark_summary.json  # Mean +/- Std across seeds
│   └── training_history.json   # Training rewards and loss logs
├── plots/
│   ├── pdr_comparison.png       # PDR comparison bar + boxplot
│   ├── delay_comparison.png     # End-to-end delay comparison
│   ├── energy_comparison.png    # Energy consumption comparison
│   ├── link_breaks_comparison.png # Link break robustness
│   ├── ddpg_training_curve.png  # Training reward & loss curve
│   └── overall_dashboard.png    # 4-panel publication dashboard
├── train.py                 # Training script
├── evaluate.py              # 10-seed benchmarking script
├── plot_results.py          # Visualization script
├── main.py                  # Master CLI entrypoint
└── requirements.txt         # Dependencies
```

---

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Interactive Simulation Demo
```bash
python main.py --demo
```

### 3. Run Unit Tests
```bash
python -m unittest tests/test_full_suite.py
```

### 4. Train DDPG Agent
```bash
python main.py --train --episodes 120 --steps 50
```

### 5. Benchmark Across 10 Random Seeds
```bash
python main.py --eval
```

### 6. Generate Figures
```bash
python main.py --plot
```

Or execute the complete end-to-end pipeline in one command:
```bash
python main.py --all
```

---

## Evaluation Results (10 Random Seeds)

| Protocol | Packet Delivery Ratio (PDR) | End-to-End Delay | Energy Consumed | Link Breaks |
| :--- | :---: | :---: | :---: | :---: |
| **DDPG (Ours)** | **90.17% ± 6.69%** | **2.10 ms ± 0.00 ms** | **3.3031 J ± 0.67 J** | **0.00 ± 0.00** |
| **AODV** | 94.17% ± 6.39% | 2.10 ms ± 0.00 ms | 2.3660 J ± 0.41 J | 0.00 ± 0.00 |
| **GPSR** | 91.83% ± 5.74% | 2.10 ms ± 0.00 ms | 3.3862 J ± 0.84 J | 0.00 ± 0.00 |
