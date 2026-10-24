"""
Interactive Desktop GUI for MANET Routing Simulator.
Visualizes 30 nodes moving with Random Waypoint mobility (20-50 m/s),
wireless communication links within 250m range, and live packet routing
for DDPG, AODV, and GPSR.
"""

import sys
import os
import time
import math
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manet.simulator import MANETSimulator
from routing.aodv import AODVRouting
from routing.gpsr import GPSRRouting
from routing.ddpg_routing import DDPGRouting
from rl.agent import DDPGAgent


class MANETVisualizerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MANET Simulator & DDPG Routing Dashboard (30 Nodes, RWP 20-50 m/s)")
        self.root.geometry("1200x820")
        self.root.minsize(1000, 700)

        # Simulator setup
        self.sim = MANETSimulator(num_nodes=30, radio_range=250.0, speed_range=(20.0, 50.0), seed=42)
        
        # Load DDPG agent
        state_dim = 10 * 7
        action_dim = 10
        self.agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim)
        model_path = os.path.join(os.path.dirname(__file__), "models", "best_ddpg.pt")
        if os.path.exists(model_path):
            self.agent.load(model_path)
            self.agent_loaded = True
        else:
            self.agent_loaded = False

        # Routing protocols
        self.protocols = {
            "DDPG (Ours)": DDPGRouting(agent=self.agent, max_neighbors=10),
            "AODV": AODVRouting(route_lifetime=3.0),
            "GPSR": GPSRRouting(),
        }
        self.current_protocol_name = "DDPG (Ours)"

        # Animation state
        self.is_running = False
        self.auto_packet = True
        self.selected_node = None
        self.active_paths = []  # list of tuples: (path_nodes, color, lifetime)
        self.scale = 0.65  # canvas pixels per meter (1000m -> 650px)
        self.canvas_offset = 30
        self.step_delay_ms = 40

        self._build_ui()
        self._draw_network()

    def _build_ui(self):
        # Main layout: Left = Canvas, Right = Control Panel & Stats
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left Canvas Frame
        self.canvas_frame = ttk.LabelFrame(self.paned, text="MANET 2D Simulation Terrain (1000m x 1000m, Range=250m)")
        self.paned.add(self.canvas_frame, weight=3)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#1a1c24", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Right Control Panel Frame
        self.right_frame = ttk.Frame(self.paned, width=380)
        self.paned.add(self.right_frame, weight=1)

        # Protocol Selection
        proto_group = ttk.LabelFrame(self.right_frame, text="Active Routing Protocol")
        proto_group.pack(fill=tk.X, padx=5, pady=5)

        self.proto_var = tk.StringVar(value=self.current_protocol_name)
        for name in ["DDPG (Ours)", "AODV", "GPSR"]:
            rb = ttk.Radiobutton(
                proto_group, text=name, value=name, variable=self.proto_var,
                command=self._on_protocol_changed
            )
            rb.pack(anchor=tk.W, padx=10, pady=3)

        # Simulation Controls
        ctrl_group = ttk.LabelFrame(self.right_frame, text="Simulation Controls")
        ctrl_group.pack(fill=tk.X, padx=5, pady=5)

        btn_row1 = ttk.Frame(ctrl_group)
        btn_row1.pack(fill=tk.X, padx=5, pady=3)
        self.btn_play = ttk.Button(btn_row1, text="▶ Start", command=self._toggle_run)
        self.btn_play.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.btn_step = ttk.Button(btn_row1, text="⏭ Step", command=self._single_step)
        self.btn_step.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.btn_reset = ttk.Button(btn_row1, text="🔄 Reset", command=self._reset_sim)
        self.btn_reset.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # Traffic Injection
        pkt_group = ttk.LabelFrame(self.right_frame, text="Packet Traffic")
        pkt_group.pack(fill=tk.X, padx=5, pady=5)

        self.auto_pkt_var = tk.BooleanVar(value=True)
        cb_auto = ttk.Checkbutton(pkt_group, text="Auto-generate Packets (CBR Traffic)", variable=self.auto_pkt_var)
        cb_auto.pack(anchor=tk.W, padx=10, pady=3)

        btn_send = ttk.Button(pkt_group, text="⚡ Inject Single Random Packet", command=self._inject_packet)
        btn_send.pack(fill=tk.X, padx=10, pady=4)

        # Speed slider
        speed_frame = ttk.Frame(ctrl_group)
        speed_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT, padx=3)
        self.speed_slider = ttk.Scale(speed_frame, from_=10, to_=120, value=40, command=self._on_speed_changed)
        self.speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Live Metrics
        stats_group = ttk.LabelFrame(self.right_frame, text="Live Network Performance")
        stats_group.pack(fill=tk.X, padx=5, pady=5)

        self.lbl_pdr = ttk.Label(stats_group, text="PDR: 0.0%", font=("Arial", 11, "bold"))
        self.lbl_pdr.pack(anchor=tk.W, padx=10, pady=2)
        self.lbl_delay = ttk.Label(stats_group, text="Avg Delay: 0.00 ms")
        self.lbl_delay.pack(anchor=tk.W, padx=10, pady=2)
        self.lbl_pkts = ttk.Label(stats_group, text="Delivered: 0 / 0")
        self.lbl_pkts.pack(anchor=tk.W, padx=10, pady=2)
        self.lbl_energy = ttk.Label(stats_group, text="Total Energy: 0.00 J")
        self.lbl_energy.pack(anchor=tk.W, padx=10, pady=2)
        self.lbl_breaks = ttk.Label(stats_group, text="Link Breaks: 0")
        self.lbl_breaks.pack(anchor=tk.W, padx=10, pady=2)

        # Node Inspector
        insp_group = ttk.LabelFrame(self.right_frame, text="Selected Node Inspector")
        insp_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.txt_inspector = tk.Text(insp_group, height=9, bg="#232733", fg="#e0e0e0", font=("Consolas", 9), relief=tk.FLAT)
        self.txt_inspector.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_inspector.insert(tk.END, "Click any node on the terrain\nto view its real-time neighbors,\ncoordinates, speed, and state.")

        # Bottom Buttons (Open Plots)
        bot_frame = ttk.Frame(self.right_frame)
        bot_frame.pack(fill=tk.X, padx=5, pady=5)
        btn_plots = ttk.Button(bot_frame, text="📊 Open Benchmark Plots", command=self._open_plots_folder)
        btn_plots.pack(fill=tk.X, pady=2)

    def _on_protocol_changed(self):
        self.current_protocol_name = self.proto_var.get()
        self._update_status_display()

    def _on_speed_changed(self, val):
        self.step_delay_ms = int(140 - float(val))

    def _toggle_run(self):
        self.is_running = not self.is_running
        self.btn_play.config(text="⏸ Pause" if self.is_running else "▶ Start")
        if self.is_running:
            self._simulation_loop()

    def _reset_sim(self):
        self.is_running = False
        self.btn_play.config(text="▶ Start")
        self.sim.reset(seed=int(time.time()) % 1000)
        for p in self.protocols.values():
            p.reset()
        self.active_paths.clear()
        self.selected_node = None
        self._update_status_display()
        self._draw_network()

    def _single_step(self):
        self._advance_simulation()
        self._draw_network()

    def _inject_packet(self):
        pkt = self.sim.generate_packet()
        if pkt is not None:
            proto = self.protocols[self.current_protocol_name]
            delivered = proto.route_packet(pkt, self.sim)
            color = "#00d4ff" if self.current_protocol_name == "DDPG (Ours)" else ("#ff9900" if self.current_protocol_name == "AODV" else "#00ff66")
            self.active_paths.append((list(pkt.path_history), color, 6))
        self._update_status_display()
        self._draw_network()

    def _advance_simulation(self):
        self.sim.step_mobility()
        if self.auto_pkt_var.get():
            pkt = self.sim.generate_packet()
            if pkt is not None:
                proto = self.protocols[self.current_protocol_name]
                delivered = proto.route_packet(pkt, self.sim)
                color = "#00d4ff" if self.current_protocol_name == "DDPG (Ours)" else ("#ff9900" if self.current_protocol_name == "AODV" else "#00ff66")
                self.active_paths.append((list(pkt.path_history), color, 5))

        # Fade active paths
        surviving_paths = []
        for path, col, life in self.active_paths:
            if life > 1:
                surviving_paths.append((path, col, life - 1))
        self.active_paths = surviving_paths

        self._update_status_display()
        if self.selected_node is not None:
            self._inspect_node(self.selected_node)

    def _simulation_loop(self):
        if not self.is_running:
            return
        self._advance_simulation()
        self._draw_network()
        self.root.after(self.step_delay_ms, self._simulation_loop)

    def _update_status_display(self):
        m = self.sim.get_metrics()
        pdr_color = "#4caf50" if m["pdr"] >= 80 else ("#ff9800" if m["pdr"] >= 50 else "#f44336")
        self.lbl_pdr.config(text=f"PDR: {m['pdr']:.1f}%", foreground=pdr_color)
        self.lbl_delay.config(text=f"Avg Delay: {m['delay']*1000:.2f} ms")
        self.lbl_pkts.config(text=f"Delivered: {int(m['total_delivered'])} / {int(m['total_generated'])}")
        self.lbl_energy.config(text=f"Total Energy: {m['total_energy']:.4f} J")
        self.lbl_breaks.config(text=f"Link Breaks: {int(m['link_breaks'])}")

    def _on_canvas_click(self, event):
        # Find closest node to click
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        area_w, area_h = self.sim.area_size

        closest_node = None
        min_dist = float("inf")

        for node in self.sim.nodes:
            cx = self.canvas_offset + (node.x / area_w) * (cw - 2 * self.canvas_offset)
            cy = self.canvas_offset + (node.y / area_h) * (ch - 2 * self.canvas_offset)
            d = math.hypot(event.x - cx, event.y - cy)
            if d < min_dist and d < 25:
                min_dist = d
                closest_node = node.node_id

        self.selected_node = closest_node
        if closest_node is not None:
            self._inspect_node(closest_node)
        self._draw_network()

    def _inspect_node(self, node_id: int):
        node = self.sim.nodes[node_id]
        neighbors = self.sim.get_neighbors(node_id)
        details = self.sim.get_neighbor_details(node_id)

        self.txt_inspector.delete("1.0", tk.END)
        info = (
            f"=== Node {node_id} Details ===\n"
            f"Position: ({node.x:.1f} m, {node.y:.1f} m)\n"
            f"Velocity: ({node.velocity[0]:.1f}, {node.velocity[1]:.1f}) m/s\n"
            f"Speed:    {node.speed:.1f} m/s [20-50 m/s]\n"
            f"Battery:  {node.battery_ratio*100:.1f}%\n"
            f"Active Neighbors ({len(neighbors)}):\n"
        )
        for d in details[:5]:
            v_id = d["neighbor_id"]
            dist = d["distance"]
            lq = d["link_quality"]
            let = d["let"]
            info += f"  * Node {v_id:2d}: {dist:.0f}m | LQ:{lq:.2f} | LET:{let:.1f}s\n"
        if len(details) > 5:
            info += f"  ... and {len(details)-5} more neighbors\n"

        self.txt_inspector.insert(tk.END, info)

    def _draw_network(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        area_w, area_h = self.sim.area_size
        eff_w = cw - 2 * self.canvas_offset
        eff_h = ch - 2 * self.canvas_offset

        def to_canvas(x, y):
            cx = self.canvas_offset + (x / area_w) * eff_w
            cy = self.canvas_offset + (y / area_h) * eff_h
            return cx, cy

        # Draw grid lines
        for gx in range(0, 1001, 200):
            cx1, cy1 = to_canvas(gx, 0)
            cx2, cy2 = to_canvas(gx, 1000)
            self.canvas.create_line(cx1, cy1, cx2, cy2, fill="#252836", dash=(2, 4))
        for gy in range(0, 1001, 200):
            cx1, cy1 = to_canvas(0, gy)
            cx2, cy2 = to_canvas(1000, gy)
            self.canvas.create_line(cx1, cy1, cx2, cy2, fill="#252836", dash=(2, 4))

        # Terrain boundary
        bx1, by1 = to_canvas(0, 0)
        bx2, by2 = to_canvas(1000, 1000)
        self.canvas.create_rectangle(bx1, by1, bx2, by2, outline="#3b4252", width=2)

        # Draw wireless links (range <= 250m)
        all_neighbors = self.sim.get_all_neighbors()
        drawn_links = set()
        for u, neighbors in all_neighbors.items():
            ux, uy = to_canvas(self.sim.nodes[u].x, self.sim.nodes[u].y)
            for v in neighbors:
                if (u, v) not in drawn_links and (v, u) not in drawn_links:
                    drawn_links.add((u, v))
                    vx, vy = to_canvas(self.sim.nodes[v].x, self.sim.nodes[v].y)
                    self.canvas.create_line(ux, uy, vx, vy, fill="#2e3447", width=1)

        # Draw selected node radio range circle (250m)
        if self.selected_node is not None:
            sn = self.sim.nodes[self.selected_node]
            scx, scy = to_canvas(sn.x, sn.y)
            rx_rad = (self.sim.radio_range / area_w) * eff_w
            ry_rad = (self.sim.radio_range / area_h) * eff_h
            self.canvas.create_oval(
                scx - rx_rad, scy - ry_rad, scx + rx_rad, scy + ry_rad,
                outline="#00e5ff", width=1, dash=(4, 4)
            )

        # Draw active routing paths
        for path, col, life in self.active_paths:
            width = min(4, life + 1)
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                ux, uy = to_canvas(self.sim.nodes[u].x, self.sim.nodes[u].y)
                vx, vy = to_canvas(self.sim.nodes[v].x, self.sim.nodes[v].y)
                self.canvas.create_line(ux, uy, vx, vy, fill=col, width=width, arrow=tk.LAST)

        # Draw nodes
        r = 8
        for node in self.sim.nodes:
            nx, ny = to_canvas(node.x, node.y)
            is_sel = (node.node_id == self.selected_node)
            fill_col = "#ffcc00" if is_sel else ("#00bcd4" if node.is_alive() else "#555555")
            outline_col = "#ffffff" if is_sel else "#111111"
            out_width = 2 if is_sel else 1

            self.canvas.create_oval(
                nx - r, ny - r, nx + r, ny + r,
                fill=fill_col, outline=outline_col, width=out_width
            )

            # Node label
            self.canvas.create_text(nx, ny - 13, text=str(node.node_id), fill="#cccccc", font=("Arial", 8, "bold"))

            # Velocity direction indicator
            vx, vy = node.velocity
            v_mag = math.hypot(vx, vy)
            if v_mag > 1e-3:
                dx = (vx / v_mag) * 14
                dy = (vy / v_mag) * 14
                self.canvas.create_line(nx, ny, nx + dx, ny + dy, fill="#ff5252", width=1.5)

        # Legend
        self.canvas.create_text(
            self.canvas_offset + 10, self.canvas_offset + 12,
            text=f"Nodes: 30 | Speed: 20-50 m/s | Radio: 250m | Protocol: {self.current_protocol_name}",
            anchor=tk.W, fill="#88c0d0", font=("Arial", 9, "bold")
        )

    def _open_plots_folder(self):
        plots_dir = os.path.join(os.path.dirname(__file__), "plots")
        if os.path.exists(plots_dir):
            if sys.platform == "win32":
                os.startfile(plots_dir)
            else:
                messagebox.showinfo("Plots Location", f"Plots stored in:\n{plots_dir}")


def main():
    root = tk.Tk()
    app = MANETVisualizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
