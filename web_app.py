"""
Interactive Web Application for MANET Routing Simulator & DDPG Dashboard.
Zero external web dependencies (uses standard library http.server).
Serves real-time interactive HTML5 canvas animation, protocol switcher,
live metrics telemetry, and benchmark plots.
"""

import sys
import os
import json
import time
import math
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manet.simulator import MANETSimulator
from routing.aodv import AODVRouting
from routing.gpsr import GPSRRouting
from routing.ddpg_routing import DDPGRouting
from rl.agent import DDPGAgent


class SimulationManager:
    """Manages the live MANET simulation state for the web server."""

    def __init__(self):
        self.sim = MANETSimulator(num_nodes=30, radio_range=250.0, speed_range=(20.0, 50.0), seed=42)
        state_dim = 10 * 7
        action_dim = 10
        self.agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim)
        model_path = os.path.join(os.path.dirname(__file__), "models", "best_ddpg.pt")
        if os.path.exists(model_path):
            self.agent.load(model_path)

        self.protocols = {
            "DDPG": DDPGRouting(agent=self.agent, max_neighbors=10),
            "AODV": AODVRouting(route_lifetime=3.0),
            "GPSR": GPSRRouting(),
        }
        self.current_proto_name = "DDPG"
        self.active_paths = []

    def reset(self, seed: int = 42):
        self.sim.reset(seed=seed)
        for p in self.protocols.values():
            p.reset()
        self.active_paths.clear()

    def set_protocol(self, proto_name: str):
        if proto_name in self.protocols:
            self.current_proto_name = proto_name

    def step(self, inject_packet: bool = True):
        self.sim.step_mobility()
        new_path = None
        if inject_packet:
            pkt = self.sim.generate_packet()
            if pkt is not None:
                proto = self.protocols[self.current_proto_name]
                delivered = proto.route_packet(pkt, self.sim)
                new_path = {
                    "path": list(pkt.path_history),
                    "delivered": delivered,
                    "protocol": self.current_proto_name,
                    "ttl": 8,
                }
                self.active_paths.append(new_path)

        # Age paths
        surviving = []
        for p in self.active_paths:
            if p["ttl"] > 1:
                p["ttl"] -= 1
                surviving.append(p)
        self.active_paths = surviving

        return self.get_state()

    def get_state(self):
        nodes_data = []
        for n in self.sim.nodes:
            nodes_data.append({
                "id": n.node_id,
                "x": round(float(n.x), 1),
                "y": round(float(n.y), 1),
                "vx": round(float(n.velocity[0]), 1),
                "vy": round(float(n.velocity[1]), 1),
                "speed": round(float(n.speed), 1),
                "battery": round(float(n.battery_ratio) * 100, 1),
                "neighbors": self.sim.get_neighbors(n.node_id),
            })

        metrics = self.sim.get_metrics()
        return {
            "nodes": nodes_data,
            "metrics": {
                "pdr": round(float(metrics["pdr"]), 1),
                "delay_ms": round(float(metrics["delay"] * 1000.0), 2),
                "total_generated": int(metrics["total_generated"]),
                "total_delivered": int(metrics["total_delivered"]),
                "total_energy": round(float(metrics["total_energy"]), 4),
                "link_breaks": int(metrics["link_breaks"]),
            },
            "active_protocol": self.current_proto_name,
            "radio_range": self.sim.radio_range,
            "area_size": self.sim.area_size,
            "active_paths": self.active_paths,
        }


sim_manager = SimulationManager()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MANET Simulator & DDPG Routing Dashboard</title>
<style>
  :root {
    --bg-dark: #0f111a;
    --panel-bg: #1a1c28;
    --card-bg: #232738;
    --accent-blue: #00b4d8;
    --accent-green: #06d6a0;
    --accent-orange: #f77f00;
    --text-primary: #f8f9fa;
    --text-secondary: #adb5bd;
    --border-color: #2e344a;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg-dark); color: var(--text-primary); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  header { background: var(--panel-bg); border-bottom: 1px solid var(--border-color); padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; }
  header h1 { font-size: 1.25rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; }
  header .badge { background: rgba(0, 180, 216, 0.15); color: var(--accent-blue); padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; border: 1px solid var(--accent-blue); }
  .main-container { display: flex; flex: 1; height: calc(100vh - 60px); overflow: hidden; }
  .canvas-panel { flex: 1; padding: 16px; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; }
  canvas { background: #12141f; border: 2px solid var(--border-color); border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
  .sidebar { width: 420px; background: var(--panel-bg); border-left: 1px solid var(--border-color); display: flex; flex-direction: column; overflow-y: auto; padding: 16px; gap: 16px; }
  .card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 14px; }
  .card-title { font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 10px; display: flex; justify-content: space-between; }
  .btn-group { display: flex; gap: 8px; }
  button { flex: 1; padding: 8px 12px; border: none; border-radius: 6px; font-weight: 600; cursor: pointer; transition: all 0.2s; font-size: 0.85rem; }
  .btn-primary { background: var(--accent-blue); color: #fff; }
  .btn-primary:hover { background: #0096c7; }
  .btn-secondary { background: #343a4e; color: #fff; }
  .btn-secondary:hover { background: #444b66; }
  .btn-proto { background: #2a2e42; color: var(--text-secondary); border: 1px solid var(--border-color); }
  .btn-proto.active { background: var(--accent-blue); color: #fff; border-color: var(--accent-blue); box-shadow: 0 0 12px rgba(0, 180, 216, 0.4); }
  .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .metric-box { background: rgba(0,0,0,0.2); padding: 10px; border-radius: 6px; border-left: 3px solid var(--accent-blue); }
  .metric-val { font-size: 1.3rem; font-weight: 700; color: #fff; margin-top: 2px; }
  .metric-lbl { font-size: 0.72rem; color: var(--text-secondary); text-transform: uppercase; }
  .node-inspector { font-family: monospace; font-size: 0.8rem; background: #12141f; padding: 10px; border-radius: 6px; border: 1px solid var(--border-color); min-height: 100px; white-space: pre-wrap; color: #ced4da; }
  .tabs { display: flex; gap: 6px; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }
  .tab-btn { background: transparent; color: var(--text-secondary); padding: 6px 12px; border-radius: 4px; font-size: 0.85rem; }
  .tab-btn.active { background: #2a2e42; color: #fff; font-weight: 700; }
  .gallery-img { width: 100%; border-radius: 6px; margin-top: 8px; border: 1px solid var(--border-color); }
</style>
</head>
<body>
<header>
  <h1><span>🛰️</span> MANET Simulator & DDPG Routing Dashboard</h1>
  <div style="display:flex; align-items:center; gap: 12px;">
    <span class="badge">30 Nodes | RWP 20-50 m/s | 250m Radio Range</span>
  </div>
</header>

<div class="main-container">
  <div class="canvas-panel">
    <canvas id="simCanvas" width="680" height="680"></canvas>
  </div>

  <div class="sidebar">
    <div class="card">
      <div class="card-title">Routing Protocol</div>
      <div class="btn-group">
        <button class="btn-proto active" id="btnProtoDDPG" onclick="setProtocol('DDPG')">DDPG (Ours)</button>
        <button class="btn-proto" id="btnProtoAODV" onclick="setProtocol('AODV')">AODV</button>
        <button class="btn-proto" id="btnProtoGPSR" onclick="setProtocol('GPSR')">GPSR</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Simulation Controls</div>
      <div class="btn-group" style="margin-bottom: 10px;">
        <button class="btn-primary" id="btnPlay" onclick="togglePlay()">▶ Start Simulation</button>
        <button class="btn-secondary" onclick="stepSim()">⏭ Step</button>
        <button class="btn-secondary" onclick="resetSim()">🔄 Reset</button>
      </div>
      <button class="btn-secondary" style="width:100%" onclick="injectPacket()">⚡ Inject Single Packet</button>
    </div>

    <div class="card">
      <div class="card-title">Live Performance Telemetry</div>
      <div class="metrics-grid">
        <div class="metric-box" style="border-color: var(--accent-green);">
          <div class="metric-lbl">Packet Delivery Ratio</div>
          <div class="metric-val" id="metricPDR">0.0%</div>
        </div>
        <div class="metric-box" style="border-color: var(--accent-blue);">
          <div class="metric-lbl">End-to-End Delay</div>
          <div class="metric-val" id="metricDelay">0.00 ms</div>
        </div>
        <div class="metric-box" style="border-color: var(--accent-orange);">
          <div class="metric-lbl">Delivered / Total</div>
          <div class="metric-val" id="metricPkts">0 / 0</div>
        </div>
        <div class="metric-box" style="border-color: #9d4edd;">
          <div class="metric-lbl">Energy Consumed</div>
          <div class="metric-val" id="metricEnergy">0.000 J</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Selected Node Details</div>
      <div class="node-inspector" id="inspector">Click on any node in the canvas to inspect its real-time 1-hop neighbors, velocity, and battery.</div>
    </div>

    <div class="card">
      <div class="card-title">Benchmark Visualizations</div>
      <div class="tabs">
        <button class="tab-btn active" onclick="showPlot('overall_dashboard.png', this)">Dashboard</button>
        <button class="tab-btn" onclick="showPlot('pdr_comparison.png', this)">PDR</button>
        <button class="tab-btn" onclick="showPlot('delay_comparison.png', this)">Delay</button>
        <button class="tab-btn" onclick="showPlot('energy_comparison.png', this)">Energy</button>
        <button class="tab-btn" onclick="showPlot('ddpg_training_curve.png', this)">Training</button>
      </div>
      <img id="plotView" class="gallery-img" src="/plots/overall_dashboard.png" alt="Benchmark Plot">
    </div>
  </div>
</div>

<script>
let state = null;
let isPlaying = false;
let playInterval = null;
let selectedNodeId = null;
const canvas = document.getElementById('simCanvas');
const ctx = canvas.getContext('2d');
const scale = canvas.width / 1000.0;

function fetchState() {
  fetch('/api/state').then(r => r.json()).then(data => {
    state = data;
    render();
    updateMetrics();
  });
}

function setProtocol(proto) {
  fetch('/api/protocol', {method: 'POST', body: JSON.stringify({protocol: proto})})
    .then(r => r.json()).then(data => {
      document.querySelectorAll('.btn-proto').forEach(b => b.classList.remove('active'));
      document.getElementById('btnProto' + proto).classList.add('active');
      fetchState();
    });
}

function togglePlay() {
  isPlaying = !isPlaying;
  const btn = document.getElementById('btnPlay');
  if (isPlaying) {
    btn.textContent = '⏸ Pause Simulation';
    btn.classList.replace('btn-primary', 'btn-secondary');
    playInterval = setInterval(() => {
      fetch('/api/step', {method: 'POST'}).then(r => r.json()).then(data => {
        state = data;
        render();
        updateMetrics();
      });
    }, 60);
  } else {
    btn.textContent = '▶ Start Simulation';
    btn.classList.replace('btn-secondary', 'btn-primary');
    clearInterval(playInterval);
  }
}

function stepSim() {
  fetch('/api/step', {method: 'POST'}).then(r => r.json()).then(data => {
    state = data;
    render();
    updateMetrics();
  });
}

function injectPacket() {
  fetch('/api/step', {method: 'POST'}).then(r => r.json()).then(data => {
    state = data;
    render();
    updateMetrics();
  });
}

function resetSim() {
  fetch('/api/reset', {method: 'POST'}).then(r => r.json()).then(data => {
    state = data;
    selectedNodeId = null;
    render();
    updateMetrics();
  });
}

function updateMetrics() {
  if (!state) return;
  const m = state.metrics;
  document.getElementById('metricPDR').textContent = m.pdr.toFixed(1) + '%';
  document.getElementById('metricDelay').textContent = m.delay_ms.toFixed(2) + ' ms';
  document.getElementById('metricPkts').textContent = m.total_delivered + ' / ' + m.total_generated;
  document.getElementById('metricEnergy').textContent = m.total_energy.toFixed(4) + ' J';

  if (selectedNodeId !== null) {
    const node = state.nodes.find(n => n.id === selectedNodeId);
    if (node) {
      document.getElementById('inspector').textContent =
        `=== Node ${node.id} Details ===\n` +
        `Position:  (${node.x}m, ${node.y}m)\n` +
        `Velocity:  (${node.vx}, ${node.vy}) m/s\n` +
        `Speed:     ${node.speed} m/s [20-50 m/s]\n` +
        `Battery:   ${node.battery}%\n` +
        `Neighbors: [${node.neighbors.join(', ')}] (${node.neighbors.length} in 250m)`;
    }
  }
}

canvas.addEventListener('click', (e) => {
  if (!state) return;
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) / scale;
  const my = (e.clientY - rect.top) / scale;

  let clicked = null;
  let minDist = 30;
  for (const n of state.nodes) {
    const d = Math.hypot(n.x - mx, n.y - my);
    if (d < minDist) {
      minDist = d;
      clicked = n.id;
    }
  }
  selectedNodeId = clicked;
  updateMetrics();
  render();
});

function render() {
  if (!state) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw Grid
  ctx.strokeStyle = '#1b1e2e';
  ctx.lineWidth = 1;
  for (let x = 0; x <= 1000; x += 200) {
    ctx.beginPath();
    ctx.moveTo(x * scale, 0);
    ctx.lineTo(x * scale, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y <= 1000; y += 200) {
    ctx.beginPath();
    ctx.moveTo(0, y * scale);
    ctx.lineTo(canvas.width, y * scale);
    ctx.stroke();
  }

  // Draw Links within 250m
  ctx.strokeStyle = 'rgba(46, 52, 74, 0.6)';
  ctx.lineWidth = 1;
  for (const n of state.nodes) {
    for (const vId of n.neighbors) {
      if (n.id < vId) {
        const v = state.nodes[vId];
        ctx.beginPath();
        ctx.moveTo(n.x * scale, n.y * scale);
        ctx.lineTo(v.x * scale, v.y * scale);
        ctx.stroke();
      }
    }
  }

  // Selected node radio range halo
  if (selectedNodeId !== null) {
    const sel = state.nodes.find(n => n.id === selectedNodeId);
    if (sel) {
      ctx.strokeStyle = 'rgba(0, 229, 255, 0.4)';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 6]);
      ctx.beginPath();
      ctx.arc(sel.x * scale, sel.y * scale, 250 * scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // Draw Active Packet Paths
  if (state.active_paths) {
    for (const p of state.active_paths) {
      const col = p.protocol === 'DDPG' ? '#00e5ff' : (p.protocol === 'AODV' ? '#ff9e00' : '#00f5d4');
      ctx.strokeStyle = col;
      ctx.lineWidth = 3;
      ctx.shadowColor = col;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      for (let i = 0; i < p.path.length; i++) {
        const nid = p.path[i];
        const n = state.nodes[nid];
        if (i === 0) ctx.moveTo(n.x * scale, n.y * scale);
        else ctx.lineTo(n.x * scale, n.y * scale);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
  }

  // Draw Nodes
  for (const n of state.nodes) {
    const isSel = (n.id === selectedNodeId);
    const nx = n.x * scale;
    const ny = n.y * scale;

    // Velocity arrow
    const vMag = Math.hypot(n.vx, n.vy);
    if (vMag > 0.1) {
      ctx.strokeStyle = '#ef476f';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(nx, ny);
      ctx.lineTo(nx + (n.vx / vMag) * 14, ny + (n.vy / vMag) * 14);
      ctx.stroke();
    }

    // Node body
    ctx.fillStyle = isSel ? '#ffd166' : '#06d6a0';
    ctx.beginPath();
    ctx.arc(nx, ny, isSel ? 7 : 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#12141f';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Node ID text
    ctx.fillStyle = '#adb5bd';
    ctx.font = '10px monospace';
    ctx.fillText(n.id, nx - 4, ny - 9);
  }
}

function showPlot(imgName, btn) {
  document.getElementById('plotView').src = '/plots/' + imgName;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

fetchState();
</script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/" or url.path == "/index.html":
            self._set_headers("text/html; charset=utf-8")
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif url.path == "/api/state":
            self._set_headers("application/json")
            data = sim_manager.get_state()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif url.path.startswith("/plots/"):
            fname = os.path.basename(url.path)
            fpath = os.path.join(os.path.dirname(__file__), "plots", fname)
            if os.path.exists(fpath):
                self._set_headers("image/png")
                with open(fpath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Plot not found")
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        url = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

        if url.path == "/api/step":
            self._set_headers("application/json")
            data = sim_manager.step(inject_packet=True)
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif url.path == "/api/reset":
            self._set_headers("application/json")
            sim_manager.reset(seed=int(time.time()) % 1000)
            data = sim_manager.get_state()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif url.path == "/api/protocol":
            payload = json.loads(post_data.decode("utf-8"))
            proto = payload.get("protocol", "DDPG")
            sim_manager.set_protocol(proto)
            self._set_headers("application/json")
            data = sim_manager.get_state()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        # Silence standard HTTP access logging to keep console clean
        pass


def run_web_server(port: int = 5000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"=== MANET Interactive Web Dashboard running at http://localhost:{port} ===")
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    run_web_server(port)
